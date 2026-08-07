"""
data_split.py
-------------
Splits the curated NPTEL graphical-plot COCO dataset into training,
validation, and testing subsets using a 70:15:15 ratio while preserving
class balance across the five plot categories, as reported in the
manuscript (2,336 train / 501 val / 500 test images from 3,935 keyframes).
"""

import argparse
import json
import os
import random
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PREPROCESS


def _dominant_category(image_id, ann_by_image):
    """Use the most frequent category in an image to stratify the split."""
    cats = [a["category_id"] for a in ann_by_image.get(image_id, [])]
    if not cats:
        return -1
    return max(set(cats), key=cats.count)


def stratified_split(coco_path: str, train_ratio=0.70, val_ratio=0.15,
                      test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    with open(coco_path, "r") as f:
        coco = json.load(f)

    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    images_by_class = defaultdict(list)
    for img in coco["images"]:
        cls = _dominant_category(img["id"], ann_by_image)
        images_by_class[cls].append(img)

    rng = random.Random(seed)
    train_imgs, val_imgs, test_imgs = [], [], []

    for cls, imgs in images_by_class.items():
        imgs = imgs[:]
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_imgs.extend(imgs[:n_train])
        val_imgs.extend(imgs[n_train:n_train + n_val])
        test_imgs.extend(imgs[n_train + n_val:])

    def _subset(images):
        ids = {img["id"] for img in images}
        anns = [a for a in coco["annotations"] if a["image_id"] in ids]
        return {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": images,
            "annotations": anns,
            "categories": coco["categories"],
        }

    return _subset(train_imgs), _subset(val_imgs), _subset(test_imgs)


def save_splits(coco_path, out_dir, train_ratio=None, val_ratio=None, test_ratio=None, seed=None):
    train_ratio = train_ratio or PREPROCESS.train_ratio
    val_ratio = val_ratio or PREPROCESS.val_ratio
    test_ratio = test_ratio or PREPROCESS.test_ratio
    seed = seed if seed is not None else PREPROCESS.seed

    train, val, test = stratified_split(coco_path, train_ratio, val_ratio, test_ratio, seed)

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, subset in zip(["train", "val", "test"], [train, val, test]):
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(subset, f, indent=2)
        paths[name] = path
        print(f"[INFO] {name}: {len(subset['images'])} images, "
              f"{len(subset['annotations'])} annotations -> {path}")

    return paths


def main():
    parser = argparse.ArgumentParser(description="Split the V-PLOT COCO dataset into train/val/test.")
    parser.add_argument("--coco_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=PREPROCESS.seed)
    args = parser.parse_args()

    save_splits(args.coco_json, args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
