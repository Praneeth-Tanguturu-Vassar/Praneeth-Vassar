# dataset.py — reads image tiles + CSV annotations, returns tensors for detection
#
# CSV format expected:
#   image_name_1  — tile filename (e.g. tile_001.jpg)
#   bbox_x        — left edge of box (pixels)
#   bbox_y        — top edge of box  (pixels)
#   bbox_width    — width of box     (pixels)
#   bbox_height   — height of box    (pixels)
#   label         — class id: 1=palm, 2=coconut

import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        super().__init__()
        self.df         = dataframe
        self.image_ids  = self.df['image_name_1'].unique()
        self.image_dir  = image_dir
        self.transforms = transform
        self.totensor   = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records  = self.df[self.df['image_name_1'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert xywh → xyxy format expected by Faster R-CNN
        boxes         = records[['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']].values
        boxes[:, 2]   = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3]   = boxes[:, 1] + boxes[:, 3]

        area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area   = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.as_tensor(records['label'].values, dtype=torch.int64)
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {
            'boxes':    torch.tensor(boxes, dtype=torch.float32),
            'labels':   labels,
            'image_id': torch.tensor([index]),
            'area':     area,
            'iscrowd':  iscrowd,
        }

        if self.transforms:
            sample = {'image': image, 'bboxes': target['boxes'], 'labels': labels}
            sample = self.transforms(**sample)
            image  = sample['image']
            target['boxes'] = torch.stack(
                tuple(map(torch.tensor, zip(*sample['bboxes'])))
            ).permute(1, 0)

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        return {'image': image, 'target': target}
