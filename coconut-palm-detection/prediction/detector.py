# detector.py — sliding-window Faster R-CNN inference over satellite rasters
#
# CoconutPalmDetector:
#   _load_model()      — loads Faster R-CNN checkpoint (Lightning .ckpt or plain state dict)
#   _run_detection()   — single-tile forward pass, returns (boxes, scores, labels)
#   _remove_duplicates — NMS across tile boundaries to remove cross-tile duplicates
#   process()          — iterates over all rasters in a folder using multiprocessing
#
# process_single_aoi() — per-raster worker function (called by Pool.map):
#   1. Slide 512-px tiles with overlap across the full raster
#   2. For each tile: run detection, filter by score threshold + boundary buffer
#   3. Project pixel-space boxes back to raster CRS coordinates
#   4. NMS to remove duplicates
#   5. postprocessor.quick_fix() — KD-tree label correction
#   6. cluster_extractor.extract_clusters() — plantation density filter
#   7. Save shapefile / GeoJSON

import os
import sys
# Ensure sibling modules (postprocessor, cluster_extractor) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import torch
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box as shapely_box
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import time
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Updated import names to match renamed files
from postprocessor    import CocoPalmPostProcessor
from cluster_extractor import ClusterExtractor

mp.set_start_method('spawn', force=True)


class Timer:
    def __init__(self):
        self.s = self.e = 0
    def start(self):  self.s = time.time()
    def end(self):    self.e = time.time()
    def measure(self): return self.e - self.s
    def reset(self):  self.s = self.e = 0

timer = Timer()


def process_single_aoi(args):
    """Worker function: runs full detection pipeline on one raster file."""
    timer.start()
    aoi_file, detector, output_dir, idx, total = args
    file_id = os.path.splitext(os.path.basename(aoi_file))[0]
    print(f"\n[{idx}/{total}] Processing: {file_id}")

    out_vector          = os.path.join(output_dir, f"{file_id}_pred.{detector.save_format}")
    post_process_folder = os.path.join(output_dir, 'cleaned')
    cluster_path        = os.path.join(output_dir, 'clusters')
    os.makedirs(cluster_path, exist_ok=True)
    os.makedirs(post_process_folder, exist_ok=True)
    cluster_processed   = os.path.join(cluster_path, f"{file_id}_pred.{detector.save_format}")

    # Skip if already processed (safe to resume interrupted runs)
    if os.path.exists(out_vector):
        print(f"  → Already done, skipping.")
        return

    all_detections = []

    with rasterio.open(aoi_file) as src:
        transform    = src.transform
        width, height = src.width, src.height

        # --- Main grid of tiles (interior tiles) ---
        for i in range(0, height - detector.tile_size + 1, detector.step_size):
            for j in range(0, width - detector.tile_size + 1, detector.step_size):
                actual_tile_h = min(detector.tile_size, height - i)
                actual_tile_w = min(detector.tile_size, width - j)
                win = Window(j, i, actual_tile_w, actual_tile_h)
                t   = win_transform(win, transform)

                tile = src.read(window=win)
                if tile.shape[0] > 3:
                    tile = tile[:3]
                elif tile.shape[0] < 3:
                    tile = np.pad(tile, ((0, 3 - tile.shape[0]), (0, 0), (0, 0)), mode='constant')

                if tile.shape[1] < detector.tile_size or tile.shape[2] < detector.tile_size:
                    padded = np.zeros((tile.shape[0], detector.tile_size, detector.tile_size),
                                      dtype=tile.dtype)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded

                tile_tensor  = torch.from_numpy(tile).float() / 255.0
                tile_tensor  = tile_tensor.unsqueeze(0).to(detector.device)
                predictions  = detector._run_detection(tile_tensor)

                for boxes, scores, labels in predictions:
                    for box, score, label in zip(boxes, scores, labels):
                        if score < detector.score_thresh:
                            continue
                        if label == 1:
                            class_name = 'palm'
                        elif label == 2:
                            class_name = 'coconut'
                        else:
                            continue
                        # Skip detections whose centroid is too close to the tile edge
                        # (they will be re-detected cleanly in the overlapping tile)
                        if not detector._is_center_in_valid_region(box, actual_tile_w, actual_tile_h):
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(actual_tile_w, x2), min(actual_tile_h, y2)
                        x1w, y1w = t * (x1, y1)
                        x2w, y2w = t * (x2, y2)
                        # Filter out boxes that are too large or too small in metres
                        if (x2w - x1w) > 12 or (y1w - y2w) > 12 or \
                           (x2w - x1w) < 2 or (y1w - y2w) < 2:
                            continue
                        all_detections.append({
                            'geometry': shapely_box(x1w, y1w, x2w, y2w),
                            'class':    class_name,
                            'score':    float(score),
                        })
                torch.cuda.empty_cache()

        # --- Bottom-edge strip (tiles that cover the bottom of the raster) ---
        if i + detector.tile_size < height:
            for j in range(0, width - detector.tile_size + 1, detector.step_size):
                i_edge        = height - detector.tile_size
                actual_tile_h = detector.tile_size
                actual_tile_w = min(detector.tile_size, width - j)
                win = Window(j, i_edge, actual_tile_w, actual_tile_h)
                t   = win_transform(win, transform)
                tile = src.read(window=win)
                if tile.shape[0] > 3:
                    tile = tile[:3]
                elif tile.shape[0] < 3:
                    tile = np.pad(tile, ((0, 3 - tile.shape[0]), (0, 0), (0, 0)), mode='constant')
                if tile.shape[2] < detector.tile_size:
                    padded = np.zeros((tile.shape[0], detector.tile_size, detector.tile_size),
                                      dtype=tile.dtype)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded
                tile_tensor = torch.from_numpy(tile).float() / 255.0
                tile_tensor = tile_tensor.unsqueeze(0).to(detector.device)
                for boxes, scores, labels in detector._run_detection(tile_tensor):
                    for box, score, label in zip(boxes, scores, labels):
                        if score < detector.score_thresh:
                            continue
                        class_name = 'palm' if label == 1 else ('coconut' if label == 2 else None)
                        if class_name is None:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        if (y1 + y2) / 2 < detector.boundary_buffer:
                            continue
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(actual_tile_w, x2), min(actual_tile_h, y2)
                        x1w, y1w = t * (x1, y1)
                        x2w, y2w = t * (x2, y2)
                        if (x2w - x1w) > 12 or (y1w - y2w) > 12 or \
                           (x2w - x1w) < 2 or (y1w - y2w) < 2:
                            continue
                        all_detections.append({
                            'geometry': shapely_box(x1w, y1w, x2w, y2w),
                            'class':    class_name,
                            'score':    float(score),
                        })

        # --- Right-edge strip (tiles that cover the right side of the raster) ---
        if j + detector.tile_size < width:
            for i in range(0, height - detector.tile_size + 1, detector.step_size):
                j_edge        = width - detector.tile_size
                actual_tile_h = min(detector.tile_size, height - i)
                actual_tile_w = detector.tile_size
                win = Window(j_edge, i, actual_tile_w, actual_tile_h)
                t   = win_transform(win, transform)
                tile = src.read(window=win)
                if tile.shape[0] > 3:
                    tile = tile[:3]
                elif tile.shape[0] < 3:
                    tile = np.pad(tile, ((0, 3 - tile.shape[0]), (0, 0), (0, 0)), mode='constant')
                if tile.shape[1] < detector.tile_size:
                    padded = np.zeros((tile.shape[0], detector.tile_size, detector.tile_size),
                                      dtype=tile.dtype)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded
                tile_tensor = torch.from_numpy(tile).float() / 255.0
                tile_tensor = tile_tensor.unsqueeze(0).to(detector.device)
                for boxes, scores, labels in detector._run_detection(tile_tensor):
                    for box, score, label in zip(boxes, scores, labels):
                        if score < detector.score_thresh:
                            continue
                        class_name = 'palm' if label == 1 else ('coconut' if label == 2 else None)
                        if class_name is None:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        if (x1 + x2) / 2 < detector.boundary_buffer:
                            continue
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(actual_tile_w, x2), min(actual_tile_h, y2)
                        x1w, y1w = t * (x1, y1)
                        x2w, y2w = t * (x2, y2)
                        if (x2w - x1w) > 16 or (y1w - y2w) > 16 or \
                           (x2w - x1w) < 2 or (y1w - y2w) < 2:
                            continue
                        all_detections.append({
                            'geometry': shapely_box(x1w, y1w, x2w, y2w),
                            'class':    class_name,
                            'score':    float(score),
                        })

    # --- Remove cross-tile duplicates with NMS ---
    all_detections = detector._remove_duplicates(all_detections, iou_threshold=0.3)

    if all_detections:
        gdf         = gpd.GeoDataFrame(all_detections, crs=src.crs)
        # Post-process: KD-tree neighbour label correction
        gdf_cleaned = detector.post_processor.quick_fix(gdf=gdf)
        driver      = 'ESRI Shapefile' if detector.save_format == 'shp' else 'GeoJSON'
        gdf_cleaned.to_file(out_vector, driver=driver)

        if detector.extract_clusters:
            gdf_clusters = detector.cluster_extractor.extract_clusters(gdf_cleaned)
            if not gdf_clusters.empty:
                gdf_clusters.to_file(cluster_processed, driver=driver)
                print(f"Saved: {out_vector}")
            else:
                print(f"No dense clusters found for: {file_id}")
    else:
        print(f"No detections for: {file_id}")

    timer.end()
    print(f"Time taken: {timer.measure():.1f}s")
    timer.reset()


class CoconutPalmDetector:
    def __init__(self, ckpt_path, score_thresh=0.40, tile_size=512, num_classes=3,
                 device=None, save_format='geojson', overlap_ratio=0.30,
                 boundary_buffer=70, extract_clusters=True):
        self.ckpt_path       = ckpt_path
        self.score_thresh    = score_thresh
        self.tile_size       = tile_size
        self.num_classes     = num_classes
        self.save_format     = save_format.lower()
        self.overlap_ratio   = overlap_ratio
        self.boundary_buffer = boundary_buffer
        self.step_size       = int(tile_size * (1 - overlap_ratio))
        self.device          = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model           = self._load_model()
        self.post_processor  = CocoPalmPostProcessor()
        self.extract_clusters = extract_clusters
        self.cluster_extractor = ClusterExtractor(hex_size=15, min_trees_per_hex=3)

    def _load_model(self):
        model      = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        checkpoint  = torch.load(self.ckpt_path, map_location=self.device)
        # Handles both Lightning .ckpt format (has 'state_dict' key) and plain state dicts
        state_dict  = {}
        if 'state_dict' in checkpoint:
            for k, v in checkpoint['state_dict'].items():
                state_dict[k.replace('model.', '') if k.startswith('model.') else k] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        print('Model loaded.')
        return model

    def _run_detection(self, batch_tensor):
        with torch.no_grad():
            preds = self.model(batch_tensor)
        return [(p['boxes'], p['scores'], p['labels']) for p in preds]

    def _is_center_in_valid_region(self, box, tile_w, tile_h):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (self.boundary_buffer <= cx <= tile_w - self.boundary_buffer and
                self.boundary_buffer <= cy <= tile_h - self.boundary_buffer)

    def _remove_duplicates(self, detections, iou_threshold=0.5):
        if not detections:
            return detections
        boxes  = torch.tensor(
            [[d['geometry'].bounds[0], d['geometry'].bounds[1],
              d['geometry'].bounds[2], d['geometry'].bounds[3]] for d in detections],
            dtype=torch.float32)
        scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
        keep   = nms(boxes, scores, iou_threshold)
        return [detections[i] for i in keep.tolist()]

    def process(self, aoi_folder, output_dir):
        """Run detection on every .jp2 / .tiff raster in aoi_folder."""
        os.makedirs(output_dir, exist_ok=True)
        aoi_files = (glob.glob(os.path.join(aoi_folder, '*.jp2'))  +
                     glob.glob(os.path.join(aoi_folder, '*.tif')) +
                     glob.glob(os.path.join(aoi_folder, '*.tiff')))
        total = len(aoi_files)
        print(f"\nFound {total} raster(s) to process.\n")
        # Run sequentially — no Pool needed for GPU inference.
        # Pool(processes=N) spawns N processes each loading the model on GPU,
        # filling all VRAM even with N=1 (spawn overhead + extra CUDA contexts).
        for idx, f in enumerate(aoi_files, start=1):
            process_single_aoi((f, self, output_dir, idx, total))
