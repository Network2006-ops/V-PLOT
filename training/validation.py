"""
validation.py
--------------
Validation loop: runs the DD-RCNN over a validation dataloader and computes
precision, accuracy, recall, F1-score, and specificity (Eqs. 13-17) using
the highest-scoring predicted proposal's class per ground-truth box as the
assigned prediction (greedy IoU matching).
"""

import sys
import os

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import compute_all_metrics, iou as iou_fn


@torch.no_grad()
def evaluate(model, dataloader, device, class_names, iou_thresh=0.5):
    """
    Args:
        model: trained DDRCNN model.
        dataloader: VPlotDataset DataLoader (with collate_fn).
        device: torch device.
        class_names: list of class name strings.
        iou_thresh: IoU threshold for matching a proposal to a GT box.

    Returns:
        dict from utils.metrics.compute_all_metrics: {"per_class": [...], "macro": {...}}
    """
    model.eval()
    y_true, y_pred = [], []

    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)

        proposals = outputs["proposals"]
        cls_logits = outputs["cls_logits"]
        pred_labels_flat = torch.argmax(cls_logits, dim=1).cpu() if cls_logits.numel() > 0 else torch.tensor([])

        offset = 0
        for i, boxes_i in enumerate(proposals):
            n_i = boxes_i.size(0)
            preds_i = pred_labels_flat[offset:offset + n_i]
            offset += n_i

            gt_boxes = targets[i]["boxes"]
            gt_labels = targets[i]["labels"]

            for k in range(gt_boxes.size(0)):
                best_iou, best_j = 0.0, -1
                for j in range(n_i):
                    cur = iou_fn(boxes_i[j].tolist(), gt_boxes[k].tolist())
                    if cur > best_iou:
                        best_iou, best_j = cur, j

                if best_iou >= iou_thresh and best_j >= 0:
                    y_pred.append(int(preds_i[best_j]))
                else:
                    y_pred.append(-1)  # unmatched / missed detection
                y_true.append(int(gt_labels[k]))

    # Treat "unmatched" (-1) predictions as an always-incorrect extra class
    # so they penalize precision/recall without crashing the confusion matrix.
    num_classes = len(class_names)
    y_pred = [p if p != -1 else (num_classes) for p in y_pred]
    y_true_arr = y_true
    y_pred_arr = y_pred

    metrics = compute_all_metrics(y_true_arr, y_pred_arr, num_classes, class_names)
    return metrics
