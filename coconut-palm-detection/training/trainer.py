# trainer.py — standalone entry point for training only (alternative to run_pipeline.py)
#
# To run training directly (without inference):
#   cd COCOPALM_TRAINER/
#   python training/trainer.py

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config     import DetectionParams
from datamodule import DetectionDataModule
from model      import DetectionLitModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def main():
    cfg = DetectionParams()

    # ── Set your paths here ──────────────────────────────────────────────────
    cfg.image_dir       = '/path/to/image/tiles/'
    cfg.csv_file        = '/path/to/annotations.csv'
    cfg.checkpoint_dir  = '/path/to/checkpoints/'
    # ─────────────────────────────────────────────────────────────────────────

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

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
    )

    datamodule = DetectionDataModule(
        root_dir=cfg.image_dir,
        dataframe=cfg.csv_file,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_split=cfg.train_split,
    )

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

    trainer.fit(task, datamodule)


if __name__ == '__main__':
    main()
