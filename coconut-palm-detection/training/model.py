# model.py — Lightning module wrapping Faster R-CNN via torchgeo ObjectDetectionTask
#
# DetectionLitModule:
#   training_step   — forward pass with targets → returns sum of Faster R-CNN loss components
#                     (cls loss + box regression + objectness + RPN regression)
#   validation_step — forward pass without targets → computes MAP metrics
#   configure_optimizers — builds optimizer from name string in config

import torch
from torchgeo.trainers import ObjectDetectionTask


class DetectionLitModule(ObjectDetectionTask):
    def __init__(self, **kwargs):
        # Pop optimizer name before passing to parent (parent doesn't accept it)
        if 'optimizer' in kwargs:
            self.optimizer_name = kwargs.pop('optimizer')
        else:
            self.optimizer_name = 'adam'
        super().__init__(**kwargs)
        self.lr = kwargs.get('lr', 1e-3)
        self._optimizer_map = {
            'SGD':     torch.optim.SGD,
            'adam':    torch.optim.Adam,
            'AdamW':   torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x          = batch['image']
        batch_size = len(x)
        # Build targets list: Faster R-CNN expects list of dicts {boxes, labels}
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ]
        # In training mode Faster R-CNN returns a dict of losses
        loss_dict  = self(x, y)
        train_loss = sum(loss_dict.values())
        self.log_dict(loss_dict)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x          = batch['image']
        batch_size = len(x)
        y = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(batch_size)
        ]
        # Compute val_loss: Faster R-CNN only returns losses in train mode,
        # so temporarily switch to train mode (same pattern as segmentation modules)
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(x, y)
        val_loss = sum(loss_dict.values())
        self.log('val_loss', val_loss, batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)

        # Compute MAP metrics in eval mode
        self.model.eval()
        y_hat = self(x)
        metrics = self.val_metrics(y_hat, y)
        metrics.pop('val_classes', None)
        self.log_dict(metrics, batch_size=batch_size)

    def configure_optimizers(self):
        # Original code had configure_optimizer (missing 's') — Lightning silently
        # ignored it and used its default. Fixed here.
        optimizer_cls = self._optimizer_map.get(self.optimizer_name, torch.optim.Adam)
        return optimizer_cls(self.model.parameters(), lr=self.lr)
