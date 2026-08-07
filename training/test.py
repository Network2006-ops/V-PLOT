"""
test.py
-------
Final evaluation on the held-out test split (500 images per the manuscript's
70:15:15 split). Reports precision, accuracy, recall, F1-score, specificity
per class (Table 3-style breakdown) and an overall normalized confusion
matrix (Figure 7).

Usage:
    python test.py --test_json ../dataset/annotations/test.json \
                    --image_dir ../dataset/keyframes \
                    --checkpoint ../checkpoints/vplot_best.pth
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from models.dd_rcnn import DDRCNN
from utils.config import CLASSES, MODEL, PATHS
from utils.metrics import normalized_confusion_matrix
from utils.visualization import plot_confusion_matrix
from training.dataset_loader import VPlotDataset, collate_fn
from training.validation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate V-PLOT DD-RCNN on the test split.")
    parser.add_argument("--test_json", type=str, default=PATHS.coco_test_json)
    parser.add_argument("--image_dir", type=str, default=PATHS.keyframes_dir)
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PATHS.checkpoints_dir, "vplot_best.pth"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--confusion_matrix_out", type=str, default="confusion_matrix.png")
    args = parser.parse_args()

    device = torch.device(args.device)

    test_dataset = VPlotDataset(args.test_json, args.image_dir, input_size=MODEL.input_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    model = DDRCNN(num_classes=CLASSES.num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metrics = evaluate(model, test_loader, device, CLASSES.names)

    print("\n=== Per-class metrics (Table 3 style) ===")
    for row in metrics["per_class"]:
        print(f"{row['class']:>15}: "
              f"precision={row['precision']:.4f} accuracy={row['accuracy']:.4f} "
              f"recall={row['recall']:.4f} f1={row['f1']:.4f} specificity={row['specificity']:.4f}")

    print("\n=== Macro-averaged metrics ===")
    for k, v in metrics["macro"].items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
