# =============================================================================
# run_pipeline.py  —  Coconut / Palm Tree Detection  :  Train  +  Predict
# =============================================================================
#
# FOLDER STRUCTURE
# ----------------
#   COCOPALM_TRAINER/
#   ├── run_pipeline.py          ← YOU ARE HERE  (single entry point)
#   │
#   ├── training/
#   │   ├── config.py            ← DetectionParams  (all hyperparameters)
#   │   ├── dataset.py           ← DetectionDataset (reads images + CSV)
#   │   ├── datamodule.py        ← DetectionDataModule (train/val split + loaders)
#   │   ├── model.py             ← DetectionLitModule (Faster R-CNN + Lightning)
#   │   └── trainer.py           ← standalone training entry point
#   │
#   └── prediction/
#       ├── detector.py          ← CoconutPalmDetector (sliding-window inference)
#       ├── postprocessor.py     ← CocoPalmPostProcessor (KD-tree label correction)
#       └── cluster_extractor.py ← ClusterExtractor (hex-grid plantation filter)
#
# PHASE 1 — TRAINING
# ------------------
#   Input  : folder of RGB image tiles (.jpg / .tif)  +  bounding-box CSV
#   Model  : Faster R-CNN ResNet-50 FPN pretrained on COCO
#            head replaced for 3 classes: 0=background, 1=palm, 2=coconut
#   Output : best-checkpoint.ckpt saved to CHECKPOINT_DIR
#
# PHASE 2 — PREDICTION (INFERENCE)
# ---------------------------------
#   Input  : trained .ckpt  +  folder of satellite rasters (.jp2 / .tif / .tiff)
#   Steps  :
#     1. Slide 512-px tiles (30% overlap) across each raster
#     2. Run Faster R-CNN on each tile → (boxes, scores, labels)
#     3. Filter by score threshold; skip detections near tile edges
#     4. Project pixel boxes back to raster CRS coordinates
#     5. NMS (IoU 0.3) to remove cross-tile duplicates
#     6. postprocessor.py  → KD-tree neighbour majority-vote label correction
#     7. cluster_extractor.py → hex-grid plantation density filter
#   Output : shapefile / GeoJSON per raster in OUTPUT_DIR
#            cluster polygons in OUTPUT_DIR/clusters/
#
# HOW TO RUN
# ----------
#   cd COCOPALM_TRAINER/
#
#   Train only:
#     python run_pipeline.py --mode train
#
#   Predict only (checkpoint must exist):
#     python run_pipeline.py --mode predict
#
#   Train then predict (uses the checkpoint just saved):
#     python run_pipeline.py --mode both
#
#   When mode is "both", CKPT_PATH below is ignored — the pipeline
#   automatically picks up CHECKPOINT_DIR/best-checkpoint.ckpt after training.
#
# =============================================================================


# =============================================================================
# ██████  CONFIGURATION  ——  EDIT THIS SECTION BEFORE RUNNING  ████████████████
# =============================================================================

# --- PHASE 1: Training — input paths -----------------------------------------
TRAIN_IMAGE_DIR  = '/path/to/image/tiles/'
#   Folder containing RGB image tiles (.jpg or .tif)
#   Each tile should be ~512 px; name must match 'image_name_1' column in CSV

TRAIN_CSV_FILE   = '/path/to/annotations.csv'
#   Bounding-box annotations CSV. Required columns:
#     image_name_1  — tile filename  (e.g. tile_001.jpg)
#     bbox_x        — left edge of bounding box  (pixels)
#     bbox_y        — top  edge of bounding box  (pixels)
#     bbox_width    — width  of bounding box     (pixels)
#     bbox_height   — height of bounding box     (pixels)
#     label         — class id: 1 = palm,  2 = coconut

CHECKPOINT_DIR   = '/path/to/checkpoints/'
#   Folder where best-checkpoint.ckpt will be saved

# --- PHASE 1: Training — hyperparameters -------------------------------------
NUM_CLASSES   = 3        # 0=background, 1=palm, 2=coconut  (do not change)
BATCH_SIZE    = 8        # reduce to 4 if you get out-of-memory errors
MAX_EPOCHS    = 20
MIN_EPOCHS    = 5
LR            = 1e-3
OPTIMIZER     = 'adam'   # choices: 'adam', 'AdamW', 'SGD', 'RMSprop'
TRAIN_SPLIT   = 0.8      # 80 % train, 20 % validation
EARLY_STOP    = True
PATIENCE      = 5        # stop after this many epochs with no val_loss improvement
DEVICES       = [0]      # GPU index; e.g. [1] for second GPU
NUM_WORKERS   = 4

# --- PHASE 2: Prediction — input paths ---------------------------------------
CKPT_PATH     = '/media/vassarml/HDD/Praneeth/Praneeth-Vassar/coconut-palm-detection/Weights/Coconut_Palm_Detection_Baseline_v1_512_V2.ckpt'
#   Path to the trained checkpoint (output of Phase 1, or existing .ckpt)

AOI_FOLDER    = '/media/vassarml/HDD/Praneeth/IN/Assam/Lakhimpur/Lakhimpur_Whole_Sat_Data/Lakhimpur/'
#   Folder of satellite rasters (.jp2 / .tif / .tiff) to run detection on

OUTPUT_DIR    = '/media/vassarml/HDD/Praneeth/IN/Assam/Lakhimpur/output/coconut-palm-tree-detection/'
#   Detections saved here as shapefile/GeoJSON; clusters in OUTPUT_DIR/clusters/

# --- PHASE 2: Prediction — detection settings --------------------------------
SCORE_THRESH     = 0.40   # discard detections below this confidence score
TILE_SIZE        = 512    # sliding-window size in pixels (match training tile size)
OVERLAP_RATIO    = 0.30   # 30 % overlap between tiles to avoid missing edge trees
BOUNDARY_BUFFER  = 70     # px margin where edge detections are discarded per tile
SAVE_FORMAT      = 'shp'  # 'shp' (shapefile)  or  'geojson'
EXTRACT_CLUSTERS = True   # if True, also save plantation cluster polygons

# --- GPU memory gating -------------------------------------------------------
#   Before each phase the pipeline waits until at least this many MiB of GPU
#   memory are free.  It checks every 20 s and NEVER kills any other process.
TRAIN_GPU_REQUIRED_MIB   = 4000   # MiB needed free before training   (~4 GB)
PREDICT_GPU_REQUIRED_MIB = 2000   # MiB needed free before prediction (~2 GB)

# =============================================================================
# ████████████  END OF CONFIGURATION — do not edit below  ████████████████████
# =============================================================================


import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root + subfolders to sys.path
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'training'))
sys.path.insert(0, os.path.join(BASE_DIR, 'prediction'))


# ===========================================================================
# PHASE 1 — TRAINING
# ===========================================================================
def run_training():
    """
    Trains Faster R-CNN ResNet-50 FPN on the labeled coconut/palm tile dataset.

    training/dataset.py    — reads image tiles + CSV → returns tensor samples
    training/datamodule.py — splits dataset 80/20, wraps in Lightning DataModule
    training/model.py      — Faster R-CNN Lightning module (training + val steps)
    training/config.py     — all hyperparameters in one place
    """

    print('\n' + '='*60)
    print('PHASE 1 — TRAINING')
    print('='*60)
    print(f'  Images     : {TRAIN_IMAGE_DIR}')
    print(f'  CSV        : {TRAIN_CSV_FILE}')
    print(f'  Checkpoints: {CHECKPOINT_DIR}')
    print(f'  Classes    : {NUM_CLASSES}  (0=bg, 1=palm, 2=coconut)')
    print(f'  Epochs     : {MAX_EPOCHS}   Batch: {BATCH_SIZE}   LR: {LR}')
    print(f'  GPU device : {DEVICES}   Required free: {TRAIN_GPU_REQUIRED_MIB:,} MiB')
    print('='*60 + '\n')

    # Imports inside function so skipping training doesn't load Lightning
    from gpu_utils  import print_gpu_stats, wait_for_gpu_memory
    from config     import DetectionParams
    from datamodule import DetectionDataModule
    from model      import DetectionLitModule
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    # 1. Build config from constants above
    cfg                = DetectionParams()
    cfg.image_dir      = TRAIN_IMAGE_DIR
    cfg.csv_file       = TRAIN_CSV_FILE
    cfg.checkpoint_dir = CHECKPOINT_DIR
    cfg.num_classes    = NUM_CLASSES
    cfg.batch_size     = BATCH_SIZE
    cfg.max_epochs     = MAX_EPOCHS
    cfg.min_epochs     = MIN_EPOCHS
    cfg.lr             = LR
    cfg.optimizer      = OPTIMIZER
    cfg.train_split    = TRAIN_SPLIT
    cfg.earlystopping  = EARLY_STOP
    cfg.patience       = PATIENCE
    cfg.devices        = DEVICES
    cfg.num_workers    = NUM_WORKERS

    # 2. Callbacks
    #    ModelCheckpoint — saves the epoch with lowest val_loss as best-checkpoint.ckpt
    #    EarlyStopping   — halts training if val_loss stops improving for PATIENCE epochs
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename='best-checkpoint',
            save_top_k=cfg.top_k_models,
            mode='min',
            monitor='val_loss',
            save_last=True,
            verbose=True,
        )
    ]
    if cfg.earlystopping:
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=cfg.patience))

    # 3. Lightning Trainer
    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
    )

    # 4. DataModule
    #    setup() reads CSV → DetectionDataset → random_split → DataLoaders
    #    Each sample: {'image': Tensor[3,H,W], 'boxes': Tensor[N,4], 'labels': Tensor[N]}
    datamodule = DetectionDataModule(
        root_dir=cfg.image_dir,
        dataframe=cfg.csv_file,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_split=cfg.train_split,
    )

    # 5. Model
    #    Faster R-CNN ResNet-50 FPN pretrained on COCO
    #    Box predictor head replaced for NUM_CLASSES output classes
    #    Internal losses: classification + box regression + objectness + RPN regression
    task = DetectionLitModule(
        model=cfg.model,
        backbone=cfg.backbone,
        weights=cfg.weights,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        lr=cfg.lr,
        patience=cfg.patience,
        optimizer=cfg.optimizer,
    )

    # 6. Wait for GPU memory, then train
    print_gpu_stats(DEVICES[0], label=f'GPU {DEVICES[0]} before training')
    wait_for_gpu_memory(TRAIN_GPU_REQUIRED_MIB, device_idx=DEVICES[0], label='training')
    print('Starting training …')
    trainer.fit(task, datamodule)
    print(f'\nTraining complete. Checkpoint saved to: {CHECKPOINT_DIR}')
    print_gpu_stats(DEVICES[0], label=f'GPU {DEVICES[0]} after training')


# ===========================================================================
# PHASE 2 — PREDICTION
# ===========================================================================
def run_prediction():
    """
    Runs the trained detector over all rasters in AOI_FOLDER.

    prediction/detector.py       — CoconutPalmDetector: loads model, tiles rasters,
                                   runs Faster R-CNN, projects to CRS, NMS
    prediction/postprocessor.py  — CocoPalmPostProcessor: KD-tree label correction
    prediction/cluster_extractor.py — ClusterExtractor: hex-grid plantation filter
    """

    print('\n' + '='*60)
    print('PHASE 2 — PREDICTION')
    print('='*60)
    print(f'  Checkpoint : {CKPT_PATH}')
    print(f'  AOI folder : {AOI_FOLDER}')
    print(f'  Output     : {OUTPUT_DIR}')
    print(f'  Score thr  : {SCORE_THRESH}   Tile: {TILE_SIZE}px   Format: {SAVE_FORMAT}')
    print(f'  GPU device : {DEVICES}   Required free: {PREDICT_GPU_REQUIRED_MIB:,} MiB')
    print('='*60 + '\n')

    # Import kept inside function so skipping prediction doesn't load rasterio etc.
    from gpu_utils import print_gpu_stats, wait_for_gpu_memory
    from detector  import CoconutPalmDetector

    # 1. Build detector
    #    _load_model() — loads checkpoint (Lightning .ckpt or plain state dict),
    #                    replaces Faster R-CNN head, sets eval mode
    detector = CoconutPalmDetector(
        ckpt_path=CKPT_PATH,
        score_thresh=SCORE_THRESH,
        tile_size=TILE_SIZE,
        num_classes=NUM_CLASSES,
        save_format=SAVE_FORMAT,
        overlap_ratio=OVERLAP_RATIO,
        boundary_buffer=BOUNDARY_BUFFER,
        extract_clusters=EXTRACT_CLUSTERS,
    )

    # 2. Wait for GPU memory, then run detection on every raster in AOI_FOLDER
    #    Uses multiprocessing (one process per raster)
    #    Rasters whose output file already exists are skipped (safe to resume)
    print_gpu_stats(DEVICES[0], label=f'GPU {DEVICES[0]} before prediction')
    wait_for_gpu_memory(PREDICT_GPU_REQUIRED_MIB, device_idx=DEVICES[0], label='prediction')
    print('Running prediction …')
    detector.process(AOI_FOLDER, OUTPUT_DIR)

    print(f'\nPrediction complete.')
    print(f'  Detections → {OUTPUT_DIR}')
    if EXTRACT_CLUSTERS:
        print(f'  Clusters   → {os.path.join(OUTPUT_DIR, "clusters/")}')
    print_gpu_stats(DEVICES[0], label=f'GPU {DEVICES[0]} after prediction')


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='CocoPalm detection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'modes:\n'
            '  train    — train the model, save checkpoint\n'
            '  predict  — run inference using an existing checkpoint\n'
            '  both     — train then predict (checkpoint path is auto-derived)\n'
        ),
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'both'],
        required=True,
        help='pipeline mode: train | predict | both',
    )
    args = parser.parse_args()

    RUN_TRAINING   = args.mode in ('train', 'both')
    RUN_PREDICTION = args.mode in ('predict', 'both')

    # When running both phases, auto-derive the checkpoint path so the user
    # does not need to fill in CKPT_PATH separately.
    ckpt_path_to_use = CKPT_PATH
    if args.mode == 'both':
        ckpt_path_to_use = os.path.join(CHECKPOINT_DIR, 'best-checkpoint.ckpt')

    # -----------------------------------------------------------------------
    # PRE-FLIGHT CHECKS — run once before any heavy work starts.
    # Verifies imports, paths, GPU memory, disk space, CSV headers, etc.
    # If anything fails, the pipeline exits cleanly here.
    # -----------------------------------------------------------------------
    from preflight import run_preflight

    _cfg = dict(
        TRAIN_IMAGE_DIR=TRAIN_IMAGE_DIR, TRAIN_CSV_FILE=TRAIN_CSV_FILE,
        CHECKPOINT_DIR=CHECKPOINT_DIR,
        CKPT_PATH=ckpt_path_to_use,
        AOI_FOLDER=AOI_FOLDER, OUTPUT_DIR=OUTPUT_DIR,
        DEVICES=DEVICES,
        TRAIN_GPU_REQUIRED_MIB=TRAIN_GPU_REQUIRED_MIB,
        PREDICT_GPU_REQUIRED_MIB=PREDICT_GPU_REQUIRED_MIB,
    )
    if not run_preflight(args.mode, _cfg):
        sys.exit(1)

    # Validate that placeholder paths have been replaced
    checks = []
    if RUN_TRAINING:
        checks += [
            ('TRAIN_IMAGE_DIR', TRAIN_IMAGE_DIR),
            ('TRAIN_CSV_FILE',  TRAIN_CSV_FILE),
            ('CHECKPOINT_DIR',  CHECKPOINT_DIR),
        ]
    if RUN_PREDICTION:
        checks += [
            ('AOI_FOLDER', AOI_FOLDER),
            ('OUTPUT_DIR', OUTPUT_DIR),
        ]
        if args.mode == 'predict':
            checks.append(('CKPT_PATH', CKPT_PATH))
    for name, val in checks:
        if '/path/to/' in val:
            print(f'ERROR: {name} is still a placeholder.')
            print('       Edit the CONFIGURATION section at the top of run_pipeline.py.')
            sys.exit(1)

    if RUN_TRAINING:
        run_training()

    if RUN_PREDICTION:
        # In 'both' mode, verify the checkpoint was actually produced
        if args.mode == 'both' and not os.path.exists(ckpt_path_to_use):
            print(f'ERROR: Expected checkpoint not found after training: {ckpt_path_to_use}')
            sys.exit(1)
        CKPT_PATH = ckpt_path_to_use
        run_prediction()

    print('\nPipeline finished.')
