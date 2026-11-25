"""
Architecture.py
---------------
Standalone model architecture for FLAIR-1 Land Use / Land Cover segmentation.

The FLAIR-1 model is a semantic segmentation network that supports two backends:
  1. HuggingFace  : AutoModelForSemanticSegmentation (default: openmmlab/upernet-swin-small)
  2. SegmentationModelsPytorch (smp): any encoder-decoder pair supported by smp.create_model()

Optional MetadataMLP fuses 45-dim aerial metadata into the encoder feature space
(only used when model_provider == 'SegmentationModelsPytorch' and use_metadata == True).

Default configuration (from flair-1-config.yaml):
  - model_provider  : HuggingFace
  - org_model       : openmmlab/upernet-swin-small
  - in_channels     : 5  (aerial bands 1–5)
  - num_classes     : 19
  - use_metadata    : False
  - norm_type       : custom
  - norm_means      : [105.08, 110.87, 101.82, 106.38, 53.26]
  - norm_stds       : [52.17,  45.38,  44.00,  39.69,  79.30]

19 LULC classes (1-indexed in masks, 0-indexed after -1 offset):
  0  building            1  pervious surface    2  impervious surface
  3  bare soil           4  water               5  coniferous
  6  deciduous           7  brushwood           8  vineyard
  9  herbaceous veg     10  agricultural land  11  plowed land
  12 swimming pool      13 snow               14  clear cut
  15 mixed              16 ligneous           17  greenhouse
  18 other
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import AutoModelForSemanticSegmentation, AutoConfig
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Constants mirroring flair-1-config.yaml defaults
# ---------------------------------------------------------------------------

DEFAULT_HF_MODEL      = "openmmlab/upernet-swin-small"
DEFAULT_N_CHANNELS    = 5
DEFAULT_N_CLASSES     = 19

NORM_MEANS = [105.08, 110.87, 101.82, 106.38, 53.26]
NORM_STDS  = [52.17,  45.38,  44.00,  39.69,  79.30]

# Class index → name (0-indexed, i.e. mask_value - 1)
CLASS_NAMES = {
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
    14: "clear cut",
    15: "mixed",
    16: "ligneous",
    17: "greenhouse",
    18: "other",
}


# ---------------------------------------------------------------------------
# MetadataMLP
# ---------------------------------------------------------------------------

class MetadataMLP(nn.Module):
    """
    Lightweight MLP that encodes 45-dimensional aerial metadata into a
    16-dimensional feature vector, which is then injected into the last
    encoder feature map of the SMP model.

    Input  : (B, 45) – encoded coordinates, altitude, camera type, datetime
    Output : (B, 16)

    Architecture:
        Linear(45 → 64) → Dropout(0.4) → ReLU
        Linear(64 → 32) → Dropout(0.4) → ReLU
        Linear(32 → 16) → Dropout(0.4) → ReLU
    """

    def __init__(self):
        super().__init__()
        self.enc_mlp = nn.Sequential(
            nn.Linear(45, 64),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(0.4),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_mlp(x)


# ---------------------------------------------------------------------------
# FLAIR_ModelFactory
# ---------------------------------------------------------------------------

class FLAIR_ModelFactory(nn.Module):
    """
    Factory that wraps either a HuggingFace or SegmentationModelsPytorch
    segmentation model, with an optional metadata branch.

    Parameters
    ----------
    model_provider : str
        'HuggingFace' or 'SegmentationModelsPytorch'
    n_channels : int
        Number of input image channels (default 5).
    n_classes : int
        Number of output segmentation classes (default 19).
    use_metadata : bool
        If True and model_provider == 'SegmentationModelsPytorch', the
        MetadataMLP branch is activated (default False).
    hf_model_name : str
        HuggingFace model identifier (default 'openmmlab/upernet-swin-small').
    smp_encoder_decoder : str
        '<encoder>_<architecture>' string for smp.create_model()
        e.g. 'resnet50_unet' (only used for SegmentationModelsPytorch).

    Forward
    -------
    x   : (B, C, H, W) image tensor
    met : (B, 45) metadata tensor or '' / None when not used

    Returns
    -------
    logits : (B, num_classes, H, W)
    """

    def __init__(
        self,
        model_provider: str = "HuggingFace",
        n_channels: int = DEFAULT_N_CHANNELS,
        n_classes: int = DEFAULT_N_CLASSES,
        use_metadata: bool = False,
        hf_model_name: str = DEFAULT_HF_MODEL,
        smp_encoder_decoder: str = None,
    ):
        super().__init__()

        self.model_provider = model_provider
        self.use_metadata   = use_metadata

        # ---- HuggingFace path ------------------------------------------------
        if model_provider == "HuggingFace":
            cfg = AutoConfig.from_pretrained(
                hf_model_name,
                num_labels=n_classes,
            )
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(
                hf_model_name,
                config=cfg,
                ignore_mismatched_sizes=True,
            )

        # ---- SegmentationModelsPytorch path ----------------------------------
        elif model_provider == "SegmentationModelsPytorch":
            if smp_encoder_decoder is None:
                raise ValueError(
                    "smp_encoder_decoder must be provided for "
                    "SegmentationModelsPytorch, e.g. 'resnet50_unet'"
                )
            encoder, architecture = smp_encoder_decoder.split("_", 1)
            self.seg_model = smp.create_model(
                arch=architecture,
                encoder_name=encoder,
                classes=n_classes,
                in_channels=n_channels,
            )
            if use_metadata:
                self.enc = MetadataMLP()

        else:
            raise ValueError(
                f"Unknown model_provider '{model_provider}'. "
                "Choose 'HuggingFace' or 'SegmentationModelsPytorch'."
            )

    def forward(
        self,
        x: torch.Tensor,
        met=None,
    ) -> torch.Tensor:

        if self.model_provider == "SegmentationModelsPytorch":

            if self.use_metadata and met is not None and met != "":
                # Inject metadata into last encoder feature map
                feats     = self.seg_model.encoder(x)
                x_enc     = self.enc(met)                              # (B, 16)
                x_enc     = x_enc.unsqueeze(1).unsqueeze(-1)           # (B, 1, 16, 1)
                x_enc     = x_enc.repeat(1, 512, 1, 16)               # (B, 512, 16, 16)
                feats[-1] = torch.add(feats[-1], x_enc)
                output    = self.seg_model.decoder(*feats)
                output    = self.seg_model.segmentation_head(output)
            else:
                output = self.seg_model(x)

        elif self.model_provider == "HuggingFace":
            output = self.seg_model(x).logits   # (B, num_classes, H/4, W/4) → upsampled by trainer

        return output


# ---------------------------------------------------------------------------
# Lightning Wrapper for Inference (Predict stage)
# ---------------------------------------------------------------------------

class FLAIR_SegmentationPredict(pl.LightningModule):
    """
    PyTorch Lightning module for inference only.

    Wraps FLAIR_ModelFactory and exposes predict_step() compatible with
    pl.Trainer.predict().

    Parameters
    ----------
    model : FLAIR_ModelFactory
    num_classes : int
    use_metadata : bool
    """

    def __init__(
        self,
        model: FLAIR_ModelFactory,
        num_classes: int = DEFAULT_N_CLASSES,
        use_metadata: bool = False,
    ):
        super().__init__()
        self.model        = model
        self.num_classes  = num_classes
        self.use_metadata = use_metadata

    def forward(self, x: torch.Tensor, met=None) -> torch.Tensor:
        return self.model(x, met)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img = batch["img"]
        met = batch["mtd"] if self.use_metadata else ""
        logits = self.forward(img, met)                 # (B, C, H, W)
        proba  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(proba, dim=1)             # (B, H, W)  0-indexed
        batch["preds"] = preds
        return batch
