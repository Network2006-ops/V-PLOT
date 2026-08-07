"""
coco_annotation.py
-------------------
Converts manually labelled bounding-box annotations for graphical plots
(bar chart, line chart, pie chart, flow chart, tree diagram) into the
standard COCO annotation format used for training DD-RCNN.

Expected raw annotation format (one JSON file per keyframe, or a single CSV):
    image_file, class_name, x_min, y_min, x_max, y_max, caption, timestamp

Output: a single COCO-style JSON file with `images`, `annotations`, and
`categories` fields, as described in the manuscript (COCO format bounding
boxes with precise class descriptions).
"""

import argparse
import csv
import json
import os
from datetime import datetime

from PIL import Image

CLASS_NAMES = ["bar_chart", "line_chart", "pie_chart", "flow_chart", "tree_diagram"]
CATEGORIES = [{"id": i + 1, "name": name, "supercategory": "plot"}
              for i, name in enumerate(CLASS_NAMES)]
CLASS_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}


def build_coco_dict():
    return {
        "info": {
            "description": "NPTEL Graphical Plot Dataset (V-PLOT)",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }


def csv_to_coco(csv_path: str, image_dir: str, output_json: str, captions_dir: str = None):
    """
    Convert a flat CSV of bounding-box labels into a COCO JSON file.

    CSV columns (header required):
        image_file,class_name,x_min,y_min,x_max,y_max,timestamp[,caption]
    """
    coco = build_coco_dict()
    image_id_map = {}
    next_image_id = 1
    next_ann_id = 1

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        image_file = row["image_file"]
        if image_file not in image_id_map:
            img_path = os.path.join(image_dir, image_file)
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except FileNotFoundError:
                width, height = 512, 512  # fall back to configured resize

            image_id_map[image_file] = next_image_id
            coco["images"].append({
                "id": next_image_id,
                "file_name": image_file,
                "width": width,
                "height": height,
                "timestamp": float(row.get("timestamp", 0.0)),
            })
            next_image_id += 1

        image_id = image_id_map[image_file]
        x_min, y_min = float(row["x_min"]), float(row["y_min"])
        x_max, y_max = float(row["x_max"]), float(row["y_max"])
        bbox_w, bbox_h = x_max - x_min, y_max - y_min
        class_name = row["class_name"].strip().lower().replace(" ", "_")
        category_id = CLASS_TO_ID.get(class_name)
        if category_id is None:
            raise ValueError(f"Unknown class name '{class_name}' in row: {row}")

        caption = row.get("caption", "")
        if not caption and captions_dir is not None:
            cap_path = os.path.join(captions_dir, os.path.splitext(image_file)[0] + ".txt")
            if os.path.exists(cap_path):
                with open(cap_path, "r") as cf:
                    caption = cf.read().strip()

        coco["annotations"].append({
            "id": next_ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_min, y_min, bbox_w, bbox_h],
            "area": bbox_w * bbox_h,
            "iscrowd": 0,
            "caption": caption,
        })
        next_ann_id += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[INFO] Wrote COCO annotations for {len(coco['images'])} images, "
          f"{len(coco['annotations'])} boxes -> {output_json}")
    return coco


def main():
    parser = argparse.ArgumentParser(description="Build COCO-format annotations for V-PLOT.")
    parser.add_argument("--csv", type=str, required=True, help="Flat CSV of raw bounding-box labels.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing keyframe images.")
    parser.add_argument("--captions_dir", type=str, default=None, help="Optional directory of per-image caption .txt files.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to write the COCO json file.")
    args = parser.parse_args()

    csv_to_coco(args.csv, args.image_dir, args.output_json, args.captions_dir)


if __name__ == "__main__":
    main()
