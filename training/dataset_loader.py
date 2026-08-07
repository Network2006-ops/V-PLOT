"""
dataset_loader.py
-------------------
PyTorch Dataset for loading NPTEL keyframes with COCO-format bounding-box
annotations and captions, used by train.py / validation.py / test.py.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VPlotDataset(Dataset):
    """
    Loads keyframe images plus their bounding boxes, class labels, and
    caption strings from a COCO-format JSON (see preprocessing/coco_annotation.py).
    """

    def __init__(self, coco_json: str, image_dir: str, input_size=(512, 512)):
        with open(coco_json, "r") as f:
            self.coco = json.load(f)

        self.image_dir = image_dir
        self.images = {img["id"]: img for img in self.coco["images"]}
        self.image_ids = list(self.images.keys())

        self.annotations_by_image = {}
        for ann in self.coco["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        self.categories = {c["id"]: c["name"] for c in self.coco["categories"]}
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image_tensor = self.transform(image)

        anns = self.annotations_by_image.get(image_id, [])
        boxes, labels, captions = [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"] - 1)  # zero-indexed for CE loss
            captions.append(ann.get("caption", ""))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "captions": captions,
            "image_id": image_id,
            "timestamp": img_info.get("timestamp", 0.0),
            "orig_size": (orig_h, orig_w),
        }
        return image_tensor, target


def collate_fn(batch):
    """Custom collate function since each image may have a variable number
    of ground-truth boxes/captions."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
