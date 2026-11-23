# config.py — all training hyperparameters and paths in one place
# Edit the three path fields before running run_pipeline.py

class DetectionParams:
    def __init__(self):
        # ── Set these before running ──────────────────────────────────────────
        self.image_dir      = None      # folder of image tiles (.jpg / .tif)
        self.csv_file       = None      # path to annotations CSV
        self.checkpoint_dir = None      # where to save .ckpt files
        # ─────────────────────────────────────────────────────────────────────

        self.accelerator    = 'gpu'
        self.devices        = [0]
        self.min_epochs     = 5
        self.max_epochs     = 20
        self.top_k_models   = 1
        self.earlystopping  = True
        self.lr             = 1e-3
        self.patience       = 5
        self.batch_size     = 8
        self.model          = 'faster-rcnn'
        self.backbone       = 'resnet50'
        self.weights        = True      # use COCO pretrained weights
        self.optimizer      = 'adam'    # 'adam', 'AdamW', 'SGD', 'RMSprop'
        self.num_workers    = 4
        self.num_classes    = 3         # 0=background, 1=palm, 2=coconut
        self.train_split    = 0.8
        self.in_channels    = 3
