"""
metrics.py
----------
Classification metrics used to evaluate V-PLOT's plot-type classification
performance (manuscript Section: Performance analysis, Eqs. 13-17):

    Precision (PN)   = TP / (TP + FP)
    Accuracy  (AY)   = (TP + TN) / (TP + TN + FP + FN)
    Recall    (RL)   = TP / (TP + FN)
    F1-score  (F1)   = 2 * PN * RL / (PN + RL)
    Specificity (SY) = TN / (TN + FP)
"""

import numpy as np


def confusion_counts(y_true, y_pred, num_classes):
    """
    Compute per-class TP, FP, FN, TN counts in a one-vs-rest fashion for a
    multi-class classification problem.

    Args:
        y_true, y_pred: 1D arrays of integer class labels.
        num_classes: total number of classes.

    Returns:
        dict of arrays, each of length num_classes: {tp, fp, fn, tn}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tn = np.zeros(num_classes)

    for c in range(num_classes):
        tp[c] = np.sum((y_true == c) & (y_pred == c))
        fp[c] = np.sum((y_true != c) & (y_pred == c))
        fn[c] = np.sum((y_true == c) & (y_pred != c))
        tn[c] = np.sum((y_true != c) & (y_pred != c))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_score(tp, fp, eps=1e-8):
    """Precision (Eq. 13): PN = TP / (TP + FP)."""
    return tp / (tp + fp + eps)


def recall_score(tp, fn, eps=1e-8):
    """Recall (Eq. 15): RL = TP / (TP + FN)."""
    return tp / (tp + fn + eps)


def accuracy_score(tp, tn, fp, fn, eps=1e-8):
    """Accuracy (Eq. 14): AY = (TP + TN) / (TP + TN + FP + FN)."""
    return (tp + tn) / (tp + tn + fp + fn + eps)


def f1_score(precision, recall, eps=1e-8):
    """F1-score (Eq. 16): F1 = 2 * PN * RL / (PN + RL)."""
    return 2 * precision * recall / (precision + recall + eps)


def specificity_score(tn, fp, eps=1e-8):
    """Specificity (Eq. 17): SY = TN / (TN + FP)."""
    return tn / (tn + fp + eps)


def compute_all_metrics(y_true, y_pred, num_classes, class_names=None):
    """
    Compute per-class and macro-averaged precision, accuracy, recall,
    F1-score, and specificity.

    Returns:
        dict with "per_class" (DataFrame-like list of dicts) and "macro"
        (dict of scalar macro-averages).
    """
    counts = confusion_counts(y_true, y_pred, num_classes)
    precision = precision_score(counts["tp"], counts["fp"])
    recall = recall_score(counts["tp"], counts["fn"])
    accuracy = accuracy_score(counts["tp"], counts["tn"], counts["fp"], counts["fn"])
    f1 = f1_score(precision, recall)
    specificity = specificity_score(counts["tn"], counts["fp"])

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    per_class = []
    for i, name in enumerate(class_names):
        per_class.append({
            "class": name,
            "precision": float(precision[i]),
            "accuracy": float(accuracy[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "specificity": float(specificity[i]),
        })

    macro = {
        "precision": float(np.mean(precision)),
        "accuracy": float(np.mean(accuracy)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "specificity": float(np.mean(specificity)),
    }

    return {"per_class": per_class, "macro": macro}


def normalized_confusion_matrix(y_true, y_pred, num_classes):
    """Row-normalized confusion matrix (as visualized in Figure 7)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def iou(box_a, box_b):
    """Axis-aligned IoU between two boxes [x1, y1, x2, y2] (Eq. 9)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def mean_average_precision(pred_boxes, pred_scores, pred_labels,
                            gt_boxes, gt_labels, num_classes, iou_thresh=0.5):
    """
    Simplified single-image / single-batch mean Average Precision (mAP) at
    a fixed IoU threshold, computed per class then averaged (mAP@0.5).
    Intended for quick sanity checks; for benchmark-grade evaluation use
    pycocotools with the COCO-format annotations produced by
    preprocessing/coco_annotation.py.
    """
    aps = []
    for c in range(num_classes):
        c_pred_idx = [i for i, l in enumerate(pred_labels) if l == c]
        c_gt_idx = [i for i, l in enumerate(gt_labels) if l == c]
        if len(c_gt_idx) == 0:
            continue

        matched_gt = set()
        c_pred_idx.sort(key=lambda i: -pred_scores[i])
        tp, fp = 0, 0
        for i in c_pred_idx:
            best_iou, best_j = 0.0, -1
            for j in c_gt_idx:
                if j in matched_gt:
                    continue
                cur_iou = iou(pred_boxes[i], gt_boxes[j])
                if cur_iou > best_iou:
                    best_iou, best_j = cur_iou, j
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (len(c_gt_idx) + 1e-8)
        aps.append(precision * recall)

    return float(np.mean(aps)) if aps else 0.0
