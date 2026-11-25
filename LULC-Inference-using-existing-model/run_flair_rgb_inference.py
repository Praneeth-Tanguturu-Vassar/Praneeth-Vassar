"""
run_flair_rgb_inference.py
--------------------------
End-to-end inference pipeline:
  1.  Download  resnet34-UNet weights from HuggingFace
  2.  Build     SMP model  (resnet34 + UNet, 3-ch RGB, 15 classes)
  3.  Load      weights into the model
  4.  Scan      input directory for GeoTIFF images
  5.  Inference  tile-by-tile (512×512 sliding window)
  6.  Save      georeferenced 15-class prediction  →  PRED15_<name>.tif
  7.  Print     per-class pixel counts  (15 FLAIR-INC classes)
  8.  Reclassify 15 classes  →  3 classes
                   0 = Water bodies
                   1 = Land
                   2 = Vegetation
  9.  Save      georeferenced 3-class map  →  PRED3_<name>.tif
  10. Print     per-class pixel counts  (3 classes)
  11. Delete    downloaded weights file

Run:
    python inference/run_flair_rgb_inference.py
"""

import os
import sys
import requests
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import from_bounds
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

PATCH_SIZE  = 512          # model's native patch size
BATCH_SIZE  = 4            # tiles per forward pass
N_CLASSES   = 15
RGB_BANDS   = [1, 2, 3]    # 1-indexed rasterio bands

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1.  CLASS DEFINITIONS
# ============================================================

# FLAIR-INC 15 classes  (0-indexed = model output value)
FLAIR_15 = {
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
    14: "greenhouse",
}

# 3-class remapping
THREE_CLASS = {
    0: "Water bodies",
    1: "Land",
    2: "Vegetation",
}

# FLAIR-INC (0-14)  →  3-class (0=Water, 1=Land, 2=Vegetation)
REMAP_15_TO_3 = {
    0:  1,   # building            → Land
    1:  1,   # pervious surface    → Land
    2:  1,   # impervious surface  → Land
    3:  1,   # bare soil           → Land
    4:  0,   # water               → Water bodies
    5:  2,   # coniferous          → Vegetation
    6:  2,   # deciduous           → Vegetation
    7:  2,   # brushwood           → Vegetation
    8:  2,   # vineyard            → Vegetation
    9:  2,   # herbaceous veg      → Vegetation
    10: 2,   # agricultural land   → Vegetation
    11: 1,   # plowed land         → Land
    12: 0,   # swimming pool       → Water bodies
    13: 1,   # snow                → Land
    14: 1,   # greenhouse          → Land
}


# ============================================================
# STEP 1 : DOWNLOAD WEIGHTS
# ============================================================

def download_weights(url: str, local_path: str):
    print("\n" + "=" * 60)
    print("STEP 1 : Downloading model weights")
    print(f"  URL   : {url}")
    print(f"  Saving: {local_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024   # 1 MB

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                mb_done  = downloaded / (1024 ** 2)
                mb_total = total      / (1024 ** 2) if total else "?"
                print(f"\r  Progress: {mb_done:.1f} / {mb_total} MB", end="", flush=True)

    print(f"\n  Download complete  ({downloaded / (1024**2):.1f} MB)")
    print("=" * 60)


# ============================================================
# STEP 2 : BUILD MODEL
# ============================================================

def build_model() -> nn.Module:
    print("\n" + "=" * 60)
    print("STEP 2 : Building ResNet34-UNet model")
    print(f"  Encoder   : resnet34")
    print(f"  Decoder   : UNet")
    print(f"  In channels: {len(RGB_BANDS)}  (RGB)")
    print(f"  Classes   : {N_CLASSES}")

    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,         # we load our own weights
        in_channels     = len(RGB_BANDS),
        classes         = N_CLASSES,
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

    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Handle Lightning .ckpt wrapped inside 'state_dict' key
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip common prefixes added by Lightning / FLAIR wrappers
    def strip_prefix(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    # Try stripping known wrapper prefixes until we get a clean match
    for prefix in ["", "model.seg_model.", "seg_model.", "model."]:
        candidate = strip_prefix(state_dict, prefix) if prefix else state_dict
        missing, unexpected = model.load_state_dict(candidate, strict=False)
        matched = len(state_dict) - len(unexpected)
        if matched > 0:
            print(f"  Prefix used   : '{prefix}' (matched {matched} keys)")
            if missing:
                print(f"  Missing keys  : {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            print("  Weights loaded successfully.")
            break
    else:
        print("  [WARNING] Could not match any keys – check checkpoint format.")

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

    exts   = {".tif", ".tiff", ".TIF", ".TIFF"}
    paths  = sorted([
        str(p) for p in Path(input_dir).rglob("*")
        if p.suffix in exts
    ])

    if not paths:
        print("  [ERROR] No GeoTIFF images found. Exiting.")
        sys.exit(1)

    print(f"  Found {len(paths)} image(s):")
    for p in paths:
        print(f"    - {Path(p).name}")
    print("=" * 60)
    return paths


# ============================================================
# HELPERS : tiling / normalisation / reclassification
# ============================================================

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] using skimage img_as_float (handles uint8 & uint16)."""
    return img_as_float(img).astype(np.float32)


def pad_to_multiple(arr: np.ndarray, patch: int):
    """
    Pad a (C, H, W) array so H and W are multiples of `patch`.
    Returns padded array and original (H, W).
    """
    C, H, W  = arr.shape
    orig_hw  = (H, W)
    ph = (patch - H % patch) % patch
    pw = (patch - W % patch) % patch
    if ph > 0 or pw > 0:
        arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    return arr, orig_hw


def tile_image(arr: np.ndarray, patch: int):
    """
    Yield (row_start, col_start, tile_array) for each patch tile.
    arr shape: (C, H, W).
    """
    _, H, W = arr.shape
    for r in range(0, H, patch):
        for c in range(0, W, patch):
            yield r, c, arr[:, r:r+patch, c:c+patch]


def predict_tiles(model: nn.Module, tiles: list) -> np.ndarray:
    """
    Run model on a list of (C, H, W) float32 numpy tiles.
    Returns list of (H, W) int64 prediction arrays.
    """
    preds = []
    for i in range(0, len(tiles), BATCH_SIZE):
        batch_np   = np.stack(tiles[i:i + BATCH_SIZE], axis=0)   # (B, C, H, W)
        batch_t    = torch.as_tensor(batch_np, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = model(batch_t)                               # (B, 15, H, W)
            pred   = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # (B, H, W)
        preds.extend(pred.cpu().numpy())
    return preds


def reconstruct_prediction(tile_preds: list,
                           tile_coords: list,
                           full_shape: tuple,
                           patch: int) -> np.ndarray:
    """Stitch tile predictions back into the full image shape."""
    H, W   = full_shape
    output = np.zeros((H, W), dtype=np.uint8)
    for (r, c), pred in zip(tile_coords, tile_preds):
        output[r:r+patch, c:c+patch] = pred[:H-r, :W-c]
    return output


def reclassify_15_to_3(pred15: np.ndarray) -> np.ndarray:
    """Map 15-class prediction to 3-class map using REMAP_15_TO_3 LUT."""
    lut   = np.array([REMAP_15_TO_3[i] for i in range(N_CLASSES)], dtype=np.uint8)
    return lut[pred15]


def print_pixel_stats_15(pred15: np.ndarray):
    total = pred15.size
    print("\n  --- 15-Class Pixel Counts ---")
    print(f"  {'Class':>2}  {'Name':<25}  {'Pixels':>12}  {'Percent':>8}")
    print(f"  {'-'*55}")
    for cls_id, cls_name in FLAIR_15.items():
        count   = int(np.sum(pred15 == cls_id))
        percent = 100.0 * count / total
        print(f"  {cls_id:>2}  {cls_name:<25}  {count:>12,}  {percent:>7.3f}%")
    print(f"  {'':>2}  {'TOTAL':<25}  {total:>12,}  {100.0:>7.3f}%")


def print_pixel_stats_3(pred3: np.ndarray):
    total = pred3.size
    print("\n  --- 3-Class Pixel Counts (after reclassification) ---")
    print(f"  {'Class':>5}  {'Name':<20}  {'Pixels':>12}  {'Percent':>8}")
    print(f"  {'-'*50}")
    for cls_id, cls_name in THREE_CLASS.items():
        count   = int(np.sum(pred3 == cls_id))
        percent = 100.0 * count / total
        print(f"  {cls_id:>5}  {cls_name:<20}  {count:>12,}  {percent:>7.3f}%")
    print(f"  {'':>5}  {'TOTAL':<20}  {total:>12,}  {100.0:>7.3f}%")


def save_geotiff(arr: np.ndarray,
                 src_profile: dict,
                 out_path: str):
    """Write a 2D uint8 array as a single-band georeferenced GeoTIFF."""
    profile         = src_profile.copy()
    profile["count"]    = 1
    profile["dtype"]    = "uint8"
    profile["compress"] = "lzw"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


# ============================================================
# STEP 5-10 : INFERENCE LOOP
# ============================================================

def run_inference(model: nn.Module, image_paths: list, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_images = len(image_paths)

    for img_idx, img_path in enumerate(image_paths, start=1):
        name = Path(img_path).stem

        print("\n" + "=" * 60)
        print(f"STEP 5-10 : Inference  [{img_idx}/{n_images}]  {Path(img_path).name}")
        print("=" * 60)

        # ---- Read image --------------------------------------------------
        print(f"\n  [5a] Reading image: {img_path}")
        with rasterio.open(img_path) as src:
            profile = src.profile
            H_orig, W_orig = src.height, src.width
            n_bands        = src.count
            print(f"       Size     : {W_orig} × {H_orig} px")
            print(f"       Bands    : {n_bands}")
            print(f"       CRS      : {src.crs}")
            print(f"       Dtype    : {src.dtypes[0]}")

            # Read RGB; fall back gracefully if fewer bands
            bands_to_read = RGB_BANDS[:min(len(RGB_BANDS), n_bands)]
            img_raw = src.read(bands_to_read)   # (C, H, W)

        # ---- Normalize --------------------------------------------------
        print(f"  [5b] Normalising to [0,1]  (img_as_float scaling)")
        img_norm = normalize_image(img_raw)     # (C, H, W) float32

        # ---- Pad to patch multiple --------------------------------------
        img_pad, (H_orig, W_orig) = pad_to_multiple(img_norm, PATCH_SIZE)
        _, Hp, Wp = img_pad.shape
        n_tiles   = (Hp // PATCH_SIZE) * (Wp // PATCH_SIZE)
        print(f"  [5c] Padded size: {Wp} × {Hp}  ({n_tiles} tiles of {PATCH_SIZE}×{PATCH_SIZE})")

        # ---- Tile and run inference -------------------------------------
        print(f"  [5d] Running tile inference on device: {DEVICE}")
        all_tiles  = []
        all_coords = []
        for r, c, tile in tile_image(img_pad, PATCH_SIZE):
            all_tiles.append(tile)
            all_coords.append((r, c))

        tile_preds = predict_tiles(model, all_tiles)
        print(f"       {len(tile_preds)} tiles processed.")

        # ---- Reconstruct full prediction (15-class) ---------------------
        print("  [5e] Reconstructing full prediction map (15 classes)")
        pred15 = reconstruct_prediction(tile_preds, all_coords, (H_orig, W_orig), PATCH_SIZE)

        # ---- Save 15-class GeoTIFF --------------------------------------
        out15_path = os.path.join(output_dir, f"PRED15_{name}.tif")
        print(f"  [6]  Saving 15-class prediction → {out15_path}")
        save_geotiff(pred15, profile, out15_path)

        # ---- Print 15-class pixel stats ---------------------------------
        print(f"\n  [7]  Pixel statistics for: {Path(img_path).name}")
        print_pixel_stats_15(pred15)

        # ---- Reclassify 15 → 3 -----------------------------------------
        print("\n  [8]  Reclassifying 15 classes → 3 classes")
        print("       Mapping:")
        for flair_id, three_id in REMAP_15_TO_3.items():
            print(f"         {flair_id:>2} {FLAIR_15[flair_id]:<25} → {three_id} {THREE_CLASS[three_id]}")
        pred3 = reclassify_15_to_3(pred15)

        # ---- Save 3-class GeoTIFF ---------------------------------------
        out3_path = os.path.join(output_dir, f"PRED3_{name}.tif")
        print(f"\n  [9]  Saving 3-class prediction  → {out3_path}")
        save_geotiff(pred3, profile, out3_path)

        # ---- Print 3-class pixel stats ----------------------------------
        print(f"\n  [10] 3-Class statistics for: {Path(img_path).name}")
        print_pixel_stats_3(pred3)

    print("\n" + "=" * 60)
    print(f"Inference complete. All outputs saved to: {output_dir}")
    print("=" * 60)


# ============================================================
# STEP 11 : DELETE WEIGHTS
# ============================================================

def delete_weights(local_path: str):
    print("\n" + "=" * 60)
    print("STEP 11 : Deleting downloaded weights file")
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
    print("#  Model : resnet34-UNet  |  15 classes  |  RGB input")
    print("#" * 60)

    # Step 1 : Download weights
    download_weights(WEIGHTS_URL, WEIGHTS_LOCAL)

    # Step 2 : Build model
    model = build_model()

    # Step 3 : Load weights
    model = load_weights(model, WEIGHTS_LOCAL)

    # Step 4 : Scan input directory
    image_paths = scan_images(INPUT_DIR)

    # Steps 5-10 : Inference + save + stats + reclassify
    run_inference(model, image_paths, OUTPUT_DIR)

    # Step 11 : Delete weights
    delete_weights(WEIGHTS_LOCAL)

    print("\n" + "#" * 60)
    print("#  Pipeline finished successfully.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
