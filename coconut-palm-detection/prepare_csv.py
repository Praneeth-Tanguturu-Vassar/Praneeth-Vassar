# =============================================================================
# prepare_csv.py  —  Slice large RGB tiles into 512×512 sub-tiles
#                    and generate a bounding-box annotation CSV for training
# =============================================================================
#
# WHAT THIS DOES
# --------------
# 1. Loads the annotation shapefile (tree polygons in CRS coordinates)
# 2. Builds a spatial index (R-tree) over all polygons — fast lookup per window
# 3. For each large RGB tile (.tif / .jp2):
#      a. Slides a 512×512 window across the full image (no overlap)
#      b. For each window, queries which annotation polygons intersect it
#      c. Clips each polygon to the window boundary
#      d. Converts the clipped polygon bounds from CRS → pixel coordinates
#      e. Saves the 512×512 sub-tile as a .tif file (skips if already saved)
#      f. Writes one CSV row per annotation: image_name_1, bbox_x, bbox_y,
#         bbox_width, bbox_height, label
# 4. Saves the final CSV
#
# NOTE ON DATA SOURCES
# --------------------
# The Lakhimpur GT shapefile (revised_palm_added.shp) is from Assam (93°E, 27°N)
# and does NOT geographically overlap with the Pulicherla RGB tiles (78°E, 13°N).
# Therefore this script uses the Pulicherla-specific Tree.shp which covers the
# same area as the RGB tiles and contains 1,250,233 tree polygon annotations.
# All polygons are labelled "Tree" → assigned label=1 (palm) in the CSV.
# If coconut-specific annotations become available, add them with label=2.
#
# HOW TO RUN
# ----------
#   cd COCOPALM_TRAINER/
#   python prepare_csv.py
#
# OUTPUTS
# -------
#   OUTPUT_TILES_DIR/   — 512×512 .tif sub-tile images for training
#   OUTPUT_CSV          — annotation CSV for DetectionDataset
#
# =============================================================================

# =============================================================================
# CONFIGURATION — edit paths before running
# =============================================================================

RGB_FOLDER      = '/home/vassarlabs/Downloads/AP/Blocks/pulicherla/gl_data/Pulicherla_rgb'
#   Folder containing large satellite RGB tiles (.tif / .jp2)

SHAPEFILE       = '/home/vassarlabs/Downloads/AP/Blocks/pulicherla/Tree_geojson/Tree.shp'
#   Annotation shapefile: tree polygon polygons in EPSG:3857
#   Columns used: 'class' (value: "Tree"), 'geometry'

OUTPUT_TILES_DIR = '/home/vassarlabs/Downloads/AP/Blocks/pulicherla/gl_data/tiles_512/'
#   512×512 sub-tile images will be saved here

OUTPUT_CSV       = '/home/vassarlabs/Downloads/AP/Blocks/pulicherla/gl_data/annotations.csv'
#   Final annotation CSV consumed by training/dataset.py

# --- Settings ----------------------------------------------------------------
TILE_SIZE    = 512    # sliding-window size in pixels
MIN_BOX_PX   = 5     # skip annotation boxes smaller than this (noise filter)

# Class label mapping: shapefile 'class' value → integer label for training
#   1 = palm  (default for "Tree")
#   2 = coconut  (add when coconut-specific annotations are available)
LABEL_MAP = {
    'Tree':    1,
    'palm':    1,
    'coconut': 2,
}
DEFAULT_LABEL = 1   # fallback if class value not in LABEL_MAP

# =============================================================================
# END CONFIGURATION
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.transform
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from shapely.geometry import box as shapely_box


def prepare_csv():

    os.makedirs(OUTPUT_TILES_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load annotations shapefile and build spatial index
    # -------------------------------------------------------------------------
    print("Loading annotation shapefile …")
    ann_gdf = gpd.read_file(SHAPEFILE)
    print(f"  Loaded {len(ann_gdf):,} annotations  CRS: {ann_gdf.crs}")
    ann_gdf = ann_gdf[ann_gdf.geometry.notna()].reset_index(drop=True)
    print(f"  Valid geometries: {len(ann_gdf):,}")

    # Build spatial index (R-tree) — used to quickly find polygons per tile window
    print("  Building spatial index …")
    ann_sindex = ann_gdf.sindex
    print("  Spatial index ready.\n")

    # -------------------------------------------------------------------------
    # Step 2: Collect all RGB tile files
    # -------------------------------------------------------------------------
    rgb_files = sorted([
        f for f in os.listdir(RGB_FOLDER)
        if f.lower().endswith('.tif') or f.lower().endswith('.jp2')
    ])
    print(f"Found {len(rgb_files)} RGB tile files in {RGB_FOLDER}\n")

    rows = []          # accumulates CSV rows
    tiles_saved = 0    # sub-tiles written to disk
    tiles_skipped = 0  # sub-tiles with no annotations (skipped)

    # -------------------------------------------------------------------------
    # Step 3: Process each RGB tile
    # -------------------------------------------------------------------------
    for file_idx, rgb_file in enumerate(rgb_files):
        tile_path = os.path.join(RGB_FOLDER, rgb_file)
        tile_stem = os.path.splitext(rgb_file)[0]

        print(f"[{file_idx+1}/{len(rgb_files)}] {rgb_file}")

        with rasterio.open(tile_path) as src:
            tile_crs       = src.crs
            tile_w         = src.width
            tile_h         = src.height
            tile_transform = src.transform
            tile_bounds    = src.bounds

            # Reproject annotations to tile CRS if needed
            if str(ann_gdf.crs) != str(tile_crs):
                print(f"  Reprojecting annotations from {ann_gdf.crs} → {tile_crs}")
                ann_local = ann_gdf.to_crs(tile_crs)
                ann_sindex_local = ann_local.sindex
            else:
                ann_local        = ann_gdf
                ann_sindex_local = ann_sindex

            # Pre-filter: only keep annotations whose bbox touches this tile
            tile_box         = shapely_box(*tile_bounds)
            candidate_idx    = list(ann_sindex_local.query(tile_box, predicate='intersects'))
            ann_tile         = ann_local.iloc[candidate_idx].copy()

            if ann_tile.empty:
                print(f"  No annotations overlap this tile — skipping all windows.")
                continue

            ann_tile_sindex  = ann_tile.sindex
            windows_in_tile  = 0
            annots_in_tile   = 0

            # -----------------------------------------------------------------
            # Step 4: Slide 512×512 windows across the tile
            # -----------------------------------------------------------------
            for row_off in range(0, tile_h, TILE_SIZE):
                for col_off in range(0, tile_w, TILE_SIZE):

                    win_w = min(TILE_SIZE, tile_w - col_off)
                    win_h = min(TILE_SIZE, tile_h - row_off)

                    win    = Window(col_off, row_off, win_w, win_h)
                    win_tf = win_transform(win, tile_transform)

                    # CRS bounding box of this window
                    xmin, ymin, xmax, ymax = rasterio.transform.array_bounds(
                        win_h, win_w, win_tf
                    )
                    win_box = shapely_box(xmin, ymin, xmax, ymax)

                    # Fast R-tree lookup: candidate annotations touching this window
                    cand_idx  = list(ann_tile_sindex.query(win_box, predicate='intersects'))
                    ann_win   = ann_tile.iloc[cand_idx]
                    ann_win   = ann_win[ann_win.intersects(win_box)]

                    if ann_win.empty:
                        # No trees in this window — skip saving (saves disk space)
                        tiles_skipped += 1
                        continue

                    # ---------------------------------------------------------
                    # Step 5: Save the 512×512 sub-tile image
                    # ---------------------------------------------------------
                    sub_name = f"{tile_stem}_r{row_off}_c{col_off}.tif"
                    sub_path = os.path.join(OUTPUT_TILES_DIR, sub_name)

                    if not os.path.exists(sub_path):
                        data = src.read(window=win)
                        # Ensure exactly 3 bands (RGB)
                        if data.shape[0] > 3:
                            data = data[:3]
                        elif data.shape[0] < 3:
                            data = np.pad(data,
                                          ((0, 3 - data.shape[0]), (0, 0), (0, 0)),
                                          mode='constant')
                        # Pad to full TILE_SIZE for edge tiles
                        if win_w < TILE_SIZE or win_h < TILE_SIZE:
                            padded = np.zeros(
                                (3, TILE_SIZE, TILE_SIZE), dtype=data.dtype
                            )
                            padded[:, :win_h, :win_w] = data
                            data = padded

                        profile = src.profile.copy()
                        profile.update(
                            width=TILE_SIZE, height=TILE_SIZE,
                            transform=win_tf, count=3
                        )
                        with rasterio.open(sub_path, 'w', **profile) as dst:
                            dst.write(data)
                        tiles_saved += 1

                    # ---------------------------------------------------------
                    # Step 6: Convert annotation polygons → pixel-space bboxes
                    # ---------------------------------------------------------
                    for _, ann_row in ann_win.iterrows():
                        # Clip polygon to window boundary
                        geom = ann_row.geometry.intersection(win_box)
                        if geom.is_empty:
                            continue

                        # Get geo-space bounds of clipped polygon
                        gx1, gy1, gx2, gy2 = geom.bounds

                        # Convert CRS coords → pixel offsets within the sub-tile
                        # rasterio's inverse transform: (col, row) = ~transform * (x, y)
                        # Note: y increases downward in pixel space, upward in CRS
                        col1, row1 = ~win_tf * (gx1, gy2)   # top-left corner
                        col2, row2 = ~win_tf * (gx2, gy1)   # bottom-right corner

                        px1 = max(0, int(col1))
                        py1 = max(0, int(row1))
                        px2 = min(TILE_SIZE, int(col2))
                        py2 = min(TILE_SIZE, int(row2))

                        bbox_w = px2 - px1
                        bbox_h = py2 - py1

                        # Skip boxes that are too small (noise / boundary clips)
                        if bbox_w < MIN_BOX_PX or bbox_h < MIN_BOX_PX:
                            continue

                        # Map class name to label integer
                        cls_name = ann_row.get('class', 'Tree')
                        label    = LABEL_MAP.get(str(cls_name), DEFAULT_LABEL)

                        rows.append({
                            'image_name_1': sub_name,
                            'bbox_x':       px1,
                            'bbox_y':       py1,
                            'bbox_width':   bbox_w,
                            'bbox_height':  bbox_h,
                            'label':        label,
                        })
                        annots_in_tile += 1

            print(f"  Windows: {windows_in_tile}  Annotations written: {annots_in_tile}")

    # -------------------------------------------------------------------------
    # Step 7: Write CSV
    # -------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*55)
    print("DONE")
    print("="*55)
    print(f"  Sub-tiles saved    : {tiles_saved}")
    print(f"  Windows skipped    : {tiles_skipped}  (no annotations)")
    print(f"  CSV rows           : {len(df):,}")
    print(f"  Sub-tiles folder   : {OUTPUT_TILES_DIR}")
    print(f"  CSV saved to       : {OUTPUT_CSV}")
    print()
    print("Label distribution:")
    print(df['label'].value_counts().to_string())
    print()
    print("Next step — run the training pipeline:")
    print("  Edit run_pipeline.py:")
    print(f"    TRAIN_IMAGE_DIR = '{OUTPUT_TILES_DIR}'")
    print(f"    TRAIN_CSV_FILE  = '{OUTPUT_CSV}'")
    print("  Then: python run_pipeline.py")


if __name__ == '__main__':
    prepare_csv()
