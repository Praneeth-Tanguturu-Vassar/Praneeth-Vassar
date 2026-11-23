# datamodule.py — splits dataset into train/val, wraps them in Lightning DataModule
#
# DetectionDataModule:
#   setup()           — reads CSV, creates DetectionDataset, splits 80/20
#   train_dataloader  — shuffled, drop_last=True
#   val_dataloader    — not shuffled

import os
import sys
# Allow direct imports from this folder when running standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import lightning as L
from torch.utils.data import random_split
from dataset import DetectionDataset


def detection_collate_fn(batch):
    # Collates a list of sample dicts into a single batch dict
    # Faster R-CNN expects a list-of-tensors for images, not a stacked tensor
    return {
        'image':    [item['image'] for item in batch],
        'boxes':    [item['target']['boxes'] for item in batch],
        'labels':   [item['target']['labels'] for item in batch],
        'image_id': [item['target']['image_id'] for item in batch],
        'area':     [item['target']['area'] for item in batch],
        'iscrowd':  [item['target']['iscrowd'] for item in batch],
    }


class DetectionDataModule(L.LightningDataModule):
    def __init__(self, root_dir, dataframe, batch_size=8, num_workers=4,
                 transform=None, train_split=0.8):
        super().__init__()
        self.root_dir    = root_dir
        self.dataframe   = dataframe    # path to CSV file
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.transform   = transform
        self.train_split = train_split

    def setup(self, stage=None):
        df      = pd.read_csv(self.dataframe)
        dataset = DetectionDataset(dataframe=df, image_dir=self.root_dir,
                                   transform=self.transform)
        train_size = int(self.train_split * len(dataset))
        val_size   = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            collate_fn=detection_collate_fn, num_workers=self.num_workers,
            shuffle=True, drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            collate_fn=detection_collate_fn, num_workers=self.num_workers,
            shuffle=False, drop_last=False,
        )
