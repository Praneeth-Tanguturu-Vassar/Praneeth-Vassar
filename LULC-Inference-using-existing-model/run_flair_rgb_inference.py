"""
run_flair_rgb_inference.py
--------------------------
End-to-end inference pipeline:
  1.  Download  resnet34-UNet weights from HuggingFace
  2.  Build     SMP model  (resnet34 + UNet, 3-ch RGB, 19 classes)
  3.  Load      weights into the model
  4.  Scan      input directory for GeoTIFF images
  5.  Inference  tile-by-tile (512x512 sliding window)  with resume support
  6.  Save      georeferenced 19-class prediction  ->  PRED19_<name>.tif
  7.  Reclassify 19 classes -> 3 classes
                   0 = Water bodies
                   1 = Land
                   2 = Vegetation
  8.  Save      georeferenced 3-class map  ->  PRED3_<name>.tif
  9.  Print     aggregated pixel counts at the end (not per image)
  10. Delete    downloaded weights file

NOTE: The file is named "15cl" because only 15 of 19 FLAIR classes had
      non-zero training weight. The model architecture itself outputs 19
      classes -- building the model with 15 causes a size mismatch error.

RESUME: If the process is killed, re-running will:
  [SKIP]  -- skip images whose outputs are already valid
  [ADOPT] -- recover outputs that exist but were not counted yet
  [REDO]  -- redo images whose outputs are corrupted / wrong size
  [DONE]  -- freshly processed this run

Run:
    python run_flair_rgb_inference.py
"""

import os
import sys
import requests
import numpy as np
import torch
import torch.nn as nn
import rasterio
from pathlib import Path
from skimage import img_as_float
import segmentation_models_pytorch as smp


# ============================================================
# 0.  CONFIGURATION
# ============================================================

WEIGHTS_URL   = (
    "https://huggingface.co/IGNF/FLAIR-INC_rgb_15cl_resnet34-unet"
    "/resolve/main/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth"
)
WEIGHTS_LOCAL = "flair_rgb_unet_weights.pth"

INPUT_DIR  = "/media/vassarml/HDD/Praneeth/IN/Assam/Lakhimpur/Lakhimpur_Whole_Sat_Data/Lakhimpur"
OUTPUT_DIR = "/media/vassarml/HDD/Praneeth/IN/Assam/Lakhimpur/output/LULC"

PATCH_SIZE    = 512   # model's native patch size
BATCH_SIZE    = 4     # tiles per forward pass
N_CLASSES     = 19    # checkpoint has 19 output neurons ("15cl" = 15 weighted, not 15 neurons)
RGB_BANDS     = [1, 2, 3]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Progress file saved inside OUTPUT_DIR after every completed image
PROGRESS_FILE = "_progress.npz"


# ============================================================
# 1.  CLASS DEFINITIONS
# ============================================================

# All 19 FLAIR classes (0-indexed = model output value)
# Classes 14, 15, 16, 18 had weight=0 during training ("inactive")
FLAIR_19 = {
    0:  "building",
    1:  "pervious surface",
    2:  "impervious surface",
    3:  "bare soil",
    4:  "water",
    5:  "coniferous",
    6:  "deciduous",
    7:  "brushwood",
    8:  "vineyard",
    9:  "herbaceous vegetation",
    10: "agricultural land",
    11: "plowed land",
    12: "swimming pool",
    13: "snow",
    14: "clear cut",      # weight=0 during training
    15: "mixed",          # weight=0 during training
    16: "ligneous",       # weight=0 during training
    17: "greenhouse",
    18: "other",          # weight=0 during training
}

THREE_CLASS = {
    0: "Water bodies",
    1: "Land",
    2: "Vegetation",
}

# FLAIR 19 classes -> 3-class (0=Water, 1=Land, 2=Vegetation)
REMAP_19_TO_3 = {
    0:  1,   # building            -> Land
    1:  1,   # pervious surface    -> Land
    2:  1,   # impervious surface  -> Land
    3:  1,   # bare soil           -> Land
    4:  0,   # water               -> Water bodies
    5:  2,   # coniferous          -> Vegetation
    6:  2,   # deciduous           -> Vegetation
    7:  2,   # brushwood           -> Vegetation
    8:  2,   # vineyard            -> Vegetation
    9:  2,   # herbaceous veg      -> Vegetation
    10: 2,   # agricultural land   -> Vegetation
    11: 1,   # plowed land         -> Land
    12: 0,   # swimming pool       -> Water bodies
    13: 1,   # snow                -> Land
    14: 2,   # clear cut           -> Vegetation
    15: 2,   # mixed               -> Vegetation
    16: 2,   # ligneous            -> Vegetation
    17: 1,   # greenhouse          -> Land
    18: 1,   # other               -> Land
}


# ============================================================
# STEP 1 : DOWNLOAD WEIGHTS
# ============================================================

def download_weights(url: str, local_path: str):
    if os.path.exists(local_path):
        print(f"\n  [STEP 1] Weights already downloaded: {local_path}  (skipping download)")
        return

    print("\n" + "=" * 60)
    print("STEP 1 : Downloading model weights")
    print(f"  URL   : {url}")
    print(f"  Saving: {local_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total      = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024   # 1 MB

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                mb_done  = downloaded / (1024 ** 2)
                mb_total = total / (1024 ** 2) if total else "?"
                print(f"\r  Progress: {mb_done:.1f} / {mb_total} MB", end="", flush=True)

    print(f"\n  Download complete  ({downloaded / (1024**2):.1f} MB)")
    print("=" * 60)


# ============================================================
# STEP 2 : BUILD MODEL
# ============================================================

def build_model() -> nn.Module:
    print("\n" + "=" * 60)
    print("STEP 2 : Building ResNet34-UNet model")
    print(f"  Encoder    : resnet34")
    print(f"  Decoder    : UNet")
    print(f"  In channels: {len(RGB_BANDS)}  (RGB)")
    print(f"  Classes    : {N_CLASSES}")

    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = len(RGB_BANDS),
        classes         = N_CLASSES,    # 19 -- must match checkpoint segmentation head
    )
    print("  Model built successfully.")
    print("=" * 60)
    return model


# ============================================================
# STEP 3 : LOAD WEIGHTS
# ============================================================

def load_weights(model: nn.Module, ckpt_path: str) -> nn.Module:
    print("\n" + "=" * 60)
    print("STEP 3 : Loading weights into model")
    print(f"  Checkpoint: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    def strip_prefix(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    model_state = model.state_dict()

    for prefix in ["", "model.seg_model.", "seg_model.", "model."]:
        candidate = strip_prefix(state_dict, prefix) if prefix else dict(state_dict)
        filtered  = {
            k: v for k, v in candidate.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        skipped = [
            k for k in candidate
            if k in model_state and candidate[k].shape != model_state[k].shape
        ]

        if len(filtered) > 0:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"  Prefix used     : '{prefix}'")
            print(f"  Keys matched    : {len(filtered)}")
            if skipped:
                print(f"  Shape-skipped   : {skipped}")
            if missing:
                print(f"  Missing keys    : {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys : {len(unexpected)}")
            print("  Weights loaded successfully.")
            break
    else:
        print("  [WARNING] Could not match any keys -- check checkpoint format.")

    model.eval()
    model.to(DEVICE)
    print(f"  Device: {DEVICE}")
    print("=" * 60)
    return model


# ============================================================
# STEP 4 : SCAN INPUT DIRECTORY
# ============================================================

def scan_images(input_dir: str) -> list:
    print("\n" + "=" * 60)
    print("STEP 4 : Scanning input directory for GeoTIFF images")
    print(f"  Directory: {input_dir}")

    exts  = {".tif", ".tiff", ".TIF", ".TIFF"}
    paths = sorted([
        str(p) for p in Path(input_dir).rglob("*")
        if p.suffix in exts
    ])

    if not paths:
        print("  [ERROR] No GeoTIFF images found. Exiting.")
        sys.exit(1)

    print(f"  Found {len(paths)} image(s).")
    print("=" * 60)
    return paths


# ============================================================
# HELPERS : normalisation / tiling / reclassification
# ============================================================

def normalize_image(img: np.ndarray) -> np.ndarray:
    return img_as_float(img).astype(np.float32)


def pad_to_multiple(arr: np.ndarray, patch: int):
    C, H, W = arr.shape
    ph = (patch - H % patch) % patch
    pw = (patch - W % patch) % patch
    if ph > 0 or pw > 0:
        arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    return arr, (H, W)


def tile_image(arr: np.ndarray, patch: int):
    _, H, W = arr.shape
    for r in range(0, H, patch):
        for c in range(0, W, patch):
            yield r, c, arr[:, r:r + patch, c:c + patch]


def predict_tiles(model: nn.Module, tiles: list) -> list:
    preds = []
    for i in range(0, len(tiles), BATCH_SIZE):
        batch_np = np.stack(tiles[i:i + BATCH_SIZE], axis=0)
        batch_t  = torch.as_tensor(batch_np, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = model(batch_t)
            pred   = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        preds.extend(pred.cpu().numpy())
    return preds


def reconstruct_prediction(tile_preds: list,
                            tile_coords: list,
                            full_shape: tuple,
                            patch: int) -> np.ndarray:
    H, W   = full_shape
    output = np.zeros((H, W), dtype=np.uint8)
    for (r, c), pred in zip(tile_coords, tile_preds):
        output[r:r + patch, c:c + patch] = pred[:H - r, :W - c]
    return output


def reclassify_19_to_3(pred19: np.ndarray) -> np.ndarray:
    lut = np.array([REMAP_19_TO_3[i] for i in range(N_CLASSES)], dtype=np.uint8)
    return lut[pred19]


def save_geotiff(arr: np.ndarray, src_profile: dict, out_path: str):
    profile             = src_profile.copy()
    profile["count"]    = 1
    profile["dtype"]    = "uint8"
    profile["compress"] = "lzw"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


# ============================================================
# RESUME HELPERS
# ============================================================

def load_progress(output_dir: str):
    """
    Load cumulative pixel counts and set of completed image stems.
    Returns (counts_19, counts_3, completed_stems).
    """
    path = os.path.join(output_dir, PROGRESS_FILE)
    if os.path.exists(path):
        try:
            data            = np.load(path, allow_pickle=True)
            counts_19       = data["counts_19"].astype(np.int64)
            counts_3        = data["counts_3"].astype(np.int64)
            completed_stems = set(data["completed_stems"].tolist())
            print(f"  [Resume] Progress file found -- {len(completed_stems)} image(s) already done.")
            return counts_19, counts_3, completed_stems
        except Exception as e:
            print(f"  [Resume] Progress file unreadable ({e}), starting fresh.")

    return (
        np.zeros(N_CLASSES,        dtype=np.int64),
        np.zeros(len(THREE_CLASS), dtype=np.int64),
        set(),
    )


def save_progress(output_dir: str,
                  counts_19: np.ndarray,
                  counts_3: np.ndarray,
                  completed_stems: set):
    """Atomically save progress after each completed image."""
    path = os.path.join(output_dir, PROGRESS_FILE)
    tmp  = path + ".tmp"
    np.savez(tmp,
             counts_19       = counts_19,
             counts_3        = counts_3,
             completed_stems = np.array(list(completed_stems)))
    os.replace(tmp, path)   # atomic -- safe against mid-write kills


def validate_geotiff(path: str, expected_h: int, expected_w: int) -> tuple:
    """
    Returns (is_valid: bool, reason: str).
    Checks: file exists, rasterio can open it, dimensions match,
    top-left 64x64 window is readable (catches partial writes).
    """
    if not os.path.exists(path):
        return False, "file missing"
    try:
        with rasterio.open(path) as src:
            if src.height != expected_h or src.width != expected_w:
                return False, (
                    f"size mismatch -- expected {expected_w}x{expected_h}, "
                    f"got {src.width}x{src.height}"
                )
            win_h = min(64, src.height)
            win_w = min(64, src.width)
            src.read(1, window=rasterio.windows.Window(0, 0, win_w, win_h))
        return True, "ok"
    except Exception as e:
        return False, f"rasterio error: {e}"


# ============================================================
# STEP 5-9 : INFERENCE LOOP  (with resume + validation)
# ============================================================

def run_inference(model: nn.Module, image_paths: list, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_images = len(image_paths)

    # Load previous progress if any
    total_pixels_19, total_pixels_3, completed_stems = load_progress(output_dir)

    skipped = 0
    redone  = 0
    fresh   = 0

    print(f"\n  Processing {n_images} image(s) on device: {DEVICE}\n")

    for img_idx, img_path in enumerate(image_paths, start=1):
        name       = Path(img_path).stem
        out19_path = os.path.join(output_dir, f"PRED19_{name}.tif")
        out3_path  = os.path.join(output_dir, f"PRED3_{name}.tif")

        # Peek at input dimensions without reading pixels
        with rasterio.open(img_path) as src:
            H_orig, W_orig = src.height, src.width

        # ------------------------------------------------------------------
        # Case 1: Already in progress file -- validate outputs
        # ------------------------------------------------------------------
        if name in completed_stems:
            v19, r19 = validate_geotiff(out19_path, H_orig, W_orig)
            v3,  r3  = validate_geotiff(out3_path,  H_orig, W_orig)

            if v19 and v3:
                print(f"  [SKIP]  [{img_idx}/{n_images}] {Path(img_path).name}", flush=True)
                skipped += 1
                continue

            # Outputs corrupted / missing -- remove from completed and redo
            reason = r19 if not v19 else r3
            tag    = "PRED19" if not v19 else "PRED3"
            print(f"  [REDO]  [{img_idx}/{n_images}] {Path(img_path).name}"
                  f"  ({tag}: {reason})", flush=True)
            completed_stems.discard(name)
            redone += 1

        # ------------------------------------------------------------------
        # Case 2: Not in progress file -- check if outputs exist anyway
        #         (handles deleted progress file with existing outputs)
        # ------------------------------------------------------------------
        else:
            v19, _ = validate_geotiff(out19_path, H_orig, W_orig)
            v3,  _ = validate_geotiff(out3_path,  H_orig, W_orig)

            if v19 and v3:
                # Outputs exist and valid -- adopt them, recover counts
                with rasterio.open(out19_path) as src:
                    pred19_existing = src.read(1)
                with rasterio.open(out3_path) as src:
                    pred3_existing  = src.read(1)

                for cls_id in range(N_CLASSES):
                    total_pixels_19[cls_id] += int(np.sum(pred19_existing == cls_id))
                for cls_id in range(len(THREE_CLASS)):
                    total_pixels_3[cls_id]  += int(np.sum(pred3_existing  == cls_id))

                completed_stems.add(name)
                save_progress(output_dir, total_pixels_19, total_pixels_3, completed_stems)
                print(f"  [ADOPT] [{img_idx}/{n_images}] {Path(img_path).name}"
                      f"  (outputs exist, counts recovered)", flush=True)
                skipped += 1
                continue

            fresh += 1

        # ------------------------------------------------------------------
        # Full inference
        # ------------------------------------------------------------------
        with rasterio.open(img_path) as src:
            profile       = src.profile
            n_bands       = src.count
            bands_to_read = RGB_BANDS[:min(len(RGB_BANDS), n_bands)]
            img_raw       = src.read(bands_to_read)

        img_norm = normalize_image(img_raw)
        img_pad, (H_orig, W_orig) = pad_to_multiple(img_norm, PATCH_SIZE)

        all_tiles, all_coords = [], []
        for r, c, tile in tile_image(img_pad, PATCH_SIZE):
            all_tiles.append(tile)
            all_coords.append((r, c))

        tile_preds = predict_tiles(model, all_tiles)
        pred19     = reconstruct_prediction(tile_preds, all_coords, (H_orig, W_orig), PATCH_SIZE)

        save_geotiff(pred19, profile, out19_path)

        pred3 = reclassify_19_to_3(pred19)
        save_geotiff(pred3, profile, out3_path)

        # Accumulate pixel counts
        for cls_id in range(N_CLASSES):
            total_pixels_19[cls_id] += int(np.sum(pred19 == cls_id))
        for cls_id in range(len(THREE_CLASS)):
            total_pixels_3[cls_id]  += int(np.sum(pred3  == cls_id))

        # Persist progress -- atomic write, safe against kill
        completed_stems.add(name)
        save_progress(output_dir, total_pixels_19, total_pixels_3, completed_stems)

        print(f"  [DONE]  [{img_idx}/{n_images}] {Path(img_path).name}", flush=True)

    # ------------------------------------------------------------------
    # Final aggregated summary
    # ------------------------------------------------------------------
    grand_total_19 = int(total_pixels_19.sum())
    grand_total_3  = int(total_pixels_3.sum())
    inactive       = {14, 15, 16, 18}

    print("\n" + "=" * 70)
    print("FINAL SUMMARY  --  Aggregated across all images")
    print("=" * 70)
    print(f"\n  Total images          : {n_images}")
    print(f"  Freshly processed     : {fresh}")
    print(f"  Re-done (corrupt)     : {redone}")
    print(f"  Skipped (valid)       : {skipped}")
    print(f"  Total pixels          : {grand_total_19:,}")
    print(f"  Output directory      : {output_dir}")

    print(f"\n  {'─'*68}")
    print(f"  19-Class Breakdown")
    print(f"  {'─'*68}")
    print(f"  {'Cls':>3}  {'Name':<25}  {'Active':>6}  {'Pixels':>16}  {'Percent':>8}")
    print(f"  {'─'*68}")
    for cls_id, cls_name in FLAIR_19.items():
        count   = int(total_pixels_19[cls_id])
        percent = 100.0 * count / grand_total_19 if grand_total_19 > 0 else 0.0
        active  = "no" if cls_id in inactive else "yes"
        print(f"  {cls_id:>3}  {cls_name:<25}  {active:>6}  {count:>16,}  {percent:>7.3f}%")
    print(f"  {'':>3}  {'TOTAL':<25}  {'':>6}  {grand_total_19:>16,}  {100.000:>7.3f}%")

    print(f"\n  {'─'*68}")
    print(f"  3-Class Breakdown  (after reclassification)")
    print(f"  {'─'*68}")
    print(f"  {'Cls':>3}  {'Name':<20}  {'Pixels':>16}  {'Percent':>8}")
    print(f"  {'─'*68}")
    for cls_id, cls_name in THREE_CLASS.items():
        count   = int(total_pixels_3[cls_id])
        percent = 100.0 * count / grand_total_3 if grand_total_3 > 0 else 0.0
        print(f"  {cls_id:>3}  {cls_name:<20}  {count:>16,}  {percent:>7.3f}%")
    print(f"  {'':>3}  {'TOTAL':<20}  {grand_total_3:>16,}  {100.000:>7.3f}%")
    print("\n" + "=" * 70)


# ============================================================
# STEP 10 : DELETE WEIGHTS
# ============================================================

def delete_weights(local_path: str):
    print("\n" + "=" * 60)
    print("STEP 10 : Deleting downloaded weights file")
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"  Deleted: {local_path}")
    else:
        print(f"  File not found (already removed?): {local_path}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "#" * 60)
    print("#  FLAIR-INC RGB Inference Pipeline")
    print("#  Model : resnet34-UNet  |  19 classes  |  RGB input")
    print("#" * 60)

    download_weights(WEIGHTS_URL, WEIGHTS_LOCAL)
    model       = build_model()
    model       = load_weights(model, WEIGHTS_LOCAL)
    image_paths = scan_images(INPUT_DIR)
    run_inference(model, image_paths, OUTPUT_DIR)
    delete_weights(WEIGHTS_LOCAL)

    print("\n" + "#" * 60)
    print("#  Pipeline finished successfully.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
