"""
visualization.py
------------------
Plotting utilities for qualitative and quantitative visualization of
V-PLOT results: detection overlays with timestamps/captions (Figure 4),
training curves (Figure 6), and normalized confusion matrices (Figure 7).
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_detections(image: np.ndarray, boxes, labels, class_names,
                     scores=None, captions=None, timestamps=None):
    """
    Draw bounding boxes with class labels, optional confidence scores,
    retrieved captions, and timestamps on a keyframe (as in Figure 4).

    Args:
        image: HxWx3 BGR image (numpy array, uint8).
        boxes: list/array of [x1, y1, x2, y2].
        labels: list of integer class indices, one per box.
        class_names: list of class name strings, indexed by label.
        scores: optional list of confidence scores.
        captions: optional list of retrieved caption strings.
        timestamps: optional list of float timestamps (seconds).

    Returns:
        Annotated copy of `image`.
    """
    vis = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)

        label_text = class_names[labels[i]] if labels[i] < len(class_names) else str(labels[i])
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
        if timestamps is not None:
            label_text += f" @ {timestamps[i]:.1f}s"

        cv2.putText(vis, label_text, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

        if captions is not None and captions[i]:
            cv2.putText(vis, captions[i][:60], (x1, min(vis.shape[0] - 5, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return vis


def plot_training_curves(history: dict, save_path: str = None):
    """
    Plot training/testing accuracy and loss curves side by side (Figure 6).

    Args:
        history: dict with keys 'train_acc', 'val_acc', 'train_loss', 'val_loss',
                 each a list of per-epoch values.
        save_path: optional path to save the figure instead of showing it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_acc"]) + 1)
    axes[0].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[0].plot(epochs, history["val_acc"], label="Test Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("(a) Accuracy curve")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], label="Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("(b) Loss curve")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names, save_path: str = None):
    """Plot a normalized confusion matrix heatmap (Figure 7)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_comparison_bar(metrics_dict: dict, metric_name: str, save_path: str = None):
    """
    Bar chart comparing V-PLOT with baseline methods for a single metric
    (used to reproduce comparison plots such as Figure 8 / Table 6 style
    summaries).

    Args:
        metrics_dict: {"MethodName": value, ...}
        metric_name: label for the y-axis / title.
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color="steelblue")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Comparison of {metric_name}")
    plt.xticks(rotation=30, ha="right")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()
