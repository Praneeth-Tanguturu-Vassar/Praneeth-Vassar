"""
inference_pipeline.py
---------------------
End-to-end inference pipeline for FLAIR-1 LULC segmentation.

Usage (command-line):
    python inference_pipeline.py \
        --ckpt  /path/to/weights.ckpt \
        --input /path/to/image_dir_or_csv \
        --output /path/to/output_dir \
        [--provider HuggingFace] \
        [--hf_model openmmlab/upernet-swin-small] \
        [--channels 1 2 3 4 5] \
        [--batch_size 4] \
        [--device cuda]

Usage (Python API):
    from inference_pipeline import FLAIRInferencePipeline

    pipeline = FLAIRInferencePipeline(
        ckpt_path="weights.ckpt",
        model_provider="HuggingFace",
    )
    pipeline.run(image_paths=["img1.tif", "img2.tif"], output_dir="preds/")
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import rasterio
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Local architecture definition (same folder)
from Architecture import (
    FLAIR_ModelFactory,
    FLAIR_SegmentationPredict,
    CLASS_NAMES,
    DEFAULT_N_CLASSES,
    DEFAULT_N_CHANNELS,
    DEFAULT_HF_MODEL,
    NORM_MEANS,
    NORM_STDS,
)


# ---------------------------------------------------------------------------
# Normalisation helper (mirrors data_loader.py::norm)
# ---------------------------------------------------------------------------

def normalize(img: np.ndarray, means: list, stds: list) -> np.ndarray:
    """Per-channel z-score normalisation (custom mode from FLAIR config)."""
    img = img.astype(np.float64)
    for i in range(img.shape[0]):
        img[i] = (img[i] - means[i]) / stds[i]
    return img


# ---------------------------------------------------------------------------
# Minimal inference Dataset  (no mask required)
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    Reads multi-channel GeoTIFF images and returns normalised float tensors.

    Parameters
    ----------
    image_paths : list[str]
        Paths to input GeoTIFF files.
    channels : list[int]
        1-indexed band numbers to read (default [1,2,3,4,5]).
    means / stds : list[float]
        Per-channel normalisation statistics.
    """

    def __init__(
        self,
        image_paths: list,
        channels: list = None,
        means: list = None,
        stds: list = None,
    ):
        self.image_paths = image_paths
        self.channels    = channels or list(range(1, DEFAULT_N_CHANNELS + 1))
        self.means       = means    or NORM_MEANS
        self.stds        = stds     or NORM_STDS

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with rasterio.open(path) as src:
            img = src.read(self.channels).astype(np.float32)   # (C, H, W)
        img = normalize(img, self.means, self.stds)
        return {
            "img": torch.as_tensor(img, dtype=torch.float32),
            "id":  path,
        }


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, seg_module: FLAIR_SegmentationPredict):
    """
    Load weights into seg_module from a .ckpt (Lightning) or .pth/.pt file.

    Lightning checkpoints store the full module state under the key 'state_dict'
    with keys prefixed 'model.'.  Plain PyTorch files store the raw state dict.
    """
    print(f"\n[Checkpoint] Loading from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    ext = Path(ckpt_path).suffix.lower()
    if ext == ".ckpt":
        # PyTorch Lightning checkpoint
        state_dict = checkpoint.get("state_dict", checkpoint)
    elif ext in (".pth", ".pt"):
        # Raw PyTorch weights
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint extension '{ext}'. "
            "Expected .ckpt, .pth, or .pt"
        )

    # Remove 'criterion' weights (class-weight tensors stored during training)
    state_dict = {k: v for k, v in state_dict.items() if "criterion" not in k}

    missing, unexpected = seg_module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [Warning] Missing keys  ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [Warning] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    print("[Checkpoint] Weights loaded successfully.\n")


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def save_prediction(pred: np.ndarray, src_path: str, output_dir: str, georef: bool = True):
    """
    Save a 2D prediction array (H, W), uint8, as a GeoTIFF (georeferenced)
    or plain TIFF.

    pred values are 0-indexed class IDs.  The file is saved as
    PRED_<original_filename> in output_dir.
    """
    fname      = Path(src_path).name
    out_name   = "PRED_" + fname
    out_path   = os.path.join(output_dir, out_name)
    pred_uint8 = pred.astype(np.uint8)

    if georef:
        with rasterio.open(src_path, "r") as src:
            meta         = src.profile.copy()
            meta["count"]    = 1
            meta["dtype"]    = "uint8"
            meta["compress"] = "lzw"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(pred_uint8, 1)
    else:
        Image.fromarray(pred_uint8).save(out_path, compression="tiff_lzw")

    return out_path


# ---------------------------------------------------------------------------
# Main Pipeline class
# ---------------------------------------------------------------------------

class FLAIRInferencePipeline:
    """
    High-level inference pipeline for FLAIR-1.

    Parameters
    ----------
    ckpt_path : str
        Path to the model checkpoint (.ckpt / .pth / .pt).
    model_provider : str
        'HuggingFace' (default) or 'SegmentationModelsPytorch'.
    hf_model_name : str
        HuggingFace model identifier (used when model_provider='HuggingFace').
    smp_encoder_decoder : str
        '<encoder>_<arch>' for SMP, e.g. 'resnet50_unet'.
    n_channels : int
        Number of input bands (default 5).
    n_classes : int
        Number of output classes (default 19).
    channels : list[int]
        1-indexed band indices to read from each GeoTIFF (default [1..5]).
    norm_means / norm_stds : list[float]
        Per-channel normalisation statistics.
    use_metadata : bool
        Whether the model uses the auxiliary MetadataMLP branch (default False).
    batch_size : int
        DataLoader batch size (default 4).
    num_workers : int
        DataLoader workers (default 4).
    device : str
        'cuda', 'cpu', or 'auto' (default 'auto').
    georef_output : bool
        Write georeferenced GeoTIFFs (True) or plain TIFFs (False).
    """

    def __init__(
        self,
        ckpt_path: str,
        model_provider: str = "HuggingFace",
        hf_model_name: str = DEFAULT_HF_MODEL,
        smp_encoder_decoder: str = None,
        n_channels: int = DEFAULT_N_CHANNELS,
        n_classes: int = DEFAULT_N_CLASSES,
        channels: list = None,
        norm_means: list = None,
        norm_stds: list = None,
        use_metadata: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
        device: str = "auto",
        georef_output: bool = True,
    ):
        self.ckpt_path           = ckpt_path
        self.model_provider      = model_provider
        self.hf_model_name       = hf_model_name
        self.smp_encoder_decoder = smp_encoder_decoder
        self.n_channels          = n_channels
        self.n_classes           = n_classes
        self.channels            = channels  or list(range(1, n_channels + 1))
        self.norm_means          = norm_means or NORM_MEANS
        self.norm_stds           = norm_stds  or NORM_STDS
        self.use_metadata        = use_metadata
        self.batch_size          = batch_size
        self.num_workers         = num_workers
        self.georef_output       = georef_output

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.seg_module = self._build_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> FLAIR_SegmentationPredict:
        """Instantiate the model and load checkpoint weights."""
        model = FLAIR_ModelFactory(
            model_provider      = self.model_provider,
            n_channels          = self.n_channels,
            n_classes           = self.n_classes,
            use_metadata        = self.use_metadata,
            hf_model_name       = self.hf_model_name,
            smp_encoder_decoder = self.smp_encoder_decoder,
        )
        seg_module = FLAIR_SegmentationPredict(
            model        = model,
            num_classes  = self.n_classes,
            use_metadata = self.use_metadata,
        )
        load_checkpoint(self.ckpt_path, seg_module)
        seg_module.eval()
        seg_module.to(self.device)
        return seg_module

    def _build_dataloader(self, image_paths: list) -> DataLoader:
        dataset = InferenceDataset(
            image_paths = image_paths,
            channels    = self.channels,
            means       = self.norm_means,
            stds        = self.norm_stds,
        )
        return DataLoader(
            dataset,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            drop_last   = False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_batch(self, batch: dict) -> dict:
        """
        Run a single pre-loaded batch through the model.

        Parameters
        ----------
        batch : dict with keys 'img' (tensor) and 'id' (list of paths)

        Returns
        -------
        batch augmented with 'preds' key: (B, H, W) int64 tensor on CPU
        """
        img = batch["img"].to(self.device)
        met = batch.get("mtd", None)
        if met is not None:
            met = met.to(self.device)

        with torch.no_grad():
            logits = self.seg_module(img, met)
            proba  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(proba, dim=1).cpu()   # (B, H, W)

        batch["preds"] = preds
        return batch

    def run(
        self,
        image_paths: list,
        output_dir: str,
    ) -> list:
        """
        Run full inference on a list of GeoTIFF paths and save predictions.

        Parameters
        ----------
        image_paths : list[str]  paths to input GeoTIFFs
        output_dir  : str        directory where PRED_*.tif files are saved

        Returns
        -------
        List of output file paths.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        loader     = self._build_dataloader(image_paths)
        out_paths  = []
        total      = len(image_paths)

        print(f"[Inference] Running on {total} image(s) → {output_dir}")
        print(f"[Inference] Device: {self.device} | Batch size: {self.batch_size}\n")

        for batch_idx, batch in enumerate(loader):
            batch   = self.predict_batch(batch)
            preds   = batch["preds"].numpy()      # (B, H, W)
            ids     = batch["id"]

            for pred, src_path in zip(preds, ids):
                out = save_prediction(pred, src_path, output_dir, georef=self.georef_output)
                out_paths.append(out)
                print(f"  [{len(out_paths):>4d}/{total}] Saved → {Path(out).name}")

        print(f"\n[Inference] Done. {len(out_paths)} prediction(s) written to {output_dir}")
        return out_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _collect_image_paths(input_path: str) -> list:
    """
    Accept either:
    - a directory  → all .tif / .tiff files inside
    - a .csv file  → first column = image paths
    - a single file path
    """
    p = Path(input_path)
    if p.is_dir():
        paths = sorted(p.glob("**/*.tif")) + sorted(p.glob("**/*.tiff"))
        return [str(x) for x in paths]
    if p.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(p, header=None)
        return df.iloc[:, 0].tolist()
    return [str(p)]


def main():
    parser = argparse.ArgumentParser(
        description="FLAIR-1 LULC inference pipeline"
    )
    parser.add_argument("--ckpt",       required=True,  help="Path to .ckpt / .pth / .pt weight file")
    parser.add_argument("--input",      required=True,  help="Image dir, CSV, or single GeoTIFF path")
    parser.add_argument("--output",     required=True,  help="Output directory for prediction GeoTIFFs")
    parser.add_argument("--provider",   default="HuggingFace",
                        choices=["HuggingFace", "SegmentationModelsPytorch"],
                        help="Model backend (default: HuggingFace)")
    parser.add_argument("--hf_model",   default=DEFAULT_HF_MODEL,
                        help=f"HuggingFace model name (default: {DEFAULT_HF_MODEL})")
    parser.add_argument("--smp_model",  default=None,
                        help="SMP encoder_decoder string, e.g. resnet50_unet")
    parser.add_argument("--channels",   nargs="+", type=int, default=list(range(1, DEFAULT_N_CHANNELS + 1)),
                        help="1-indexed band numbers to read (default: 1 2 3 4 5)")
    parser.add_argument("--n_classes",  type=int, default=DEFAULT_N_CLASSES,
                        help=f"Number of output classes (default: {DEFAULT_N_CLASSES})")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--device",     default="auto", help="cuda / cpu / auto")
    parser.add_argument("--no_georef",  action="store_true",
                        help="Write plain TIFFs instead of georeferenced GeoTIFFs")
    args = parser.parse_args()

    image_paths = _collect_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No .tif images found under: {args.input}")

    pipeline = FLAIRInferencePipeline(
        ckpt_path           = args.ckpt,
        model_provider      = args.provider,
        hf_model_name       = args.hf_model,
        smp_encoder_decoder = args.smp_model,
        n_channels          = len(args.channels),
        n_classes           = args.n_classes,
        channels            = args.channels,
        batch_size          = args.batch_size,
        num_workers         = args.num_workers,
        device              = args.device,
        georef_output       = not args.no_georef,
    )

    pipeline.run(image_paths=image_paths, output_dir=args.output)


if __name__ == "__main__":
    main()
