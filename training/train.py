"""
train.py
--------
Training driver for the V-PLOT DD-RCNN detector with the joint
detection + CLIP caption-alignment objective (manuscript Eqs. 10-11,
Table 2 hyperparameters).

Usage:
    python train.py --train_json ../dataset/annotations/train.json \
                     --val_json ../dataset/annotations/val.json \
                     --image_dir ../dataset/keyframes
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from models.dd_rcnn import DDRCNN
from models.loss import VPlotLoss
from utils.config import CLASSES, MODEL, LOSS, TRAIN, PATHS
from utils.metrics import compute_all_metrics
from training.dataset_loader import VPlotDataset, collate_fn
from training.validation import evaluate


def match_proposals_to_targets(proposals, cls_logits, bbox_deltas, targets, num_classes):
    """
    Simplified proposal-to-ground-truth matching for loss computation:
    for each image, ground-truth boxes are treated as the supervisory
    signal for the top-scoring proposals (nearest match by IoU). This is a
    pedagogical simplification of the sampling strategy used in Faster/DD-RCNN
    training (positive/negative anchor sampling with IoU thresholds).
    """
    from utils.metrics import iou as iou_fn

    all_labels, all_bbox_targets = [], []
    offset = 0
    for i, boxes_i in enumerate(proposals):
        n_i = boxes_i.size(0)
        gt_boxes = targets[i]["boxes"]
        gt_labels = targets[i]["labels"]

        if n_i == 0:
            offset += n_i
            continue

        if gt_boxes.numel() == 0:
            labels_i = torch.zeros(n_i, dtype=torch.long)
            bbox_targets_i = torch.zeros(n_i, 4)
        else:
            labels_i = torch.zeros(n_i, dtype=torch.long)
            bbox_targets_i = torch.zeros(n_i, 4)
            for j in range(n_i):
                best_iou, best_k = 0.0, -1
                for k in range(gt_boxes.size(0)):
                    cur = iou_fn(boxes_i[j].tolist(), gt_boxes[k].tolist())
                    if cur > best_iou:
                        best_iou, best_k = cur, k
                if best_iou > 0.5 and best_k >= 0:
                    labels_i[j] = gt_labels[best_k]
                    bbox_targets_i[j] = gt_boxes[best_k]
                else:
                    bbox_targets_i[j] = boxes_i[j]  # background: regress to itself (no penalty)

        all_labels.append(labels_i)
        all_bbox_targets.append(bbox_targets_i)
        offset += n_i

    if not all_labels:
        return (torch.zeros(0, dtype=torch.long), torch.zeros(0, 4))

    return torch.cat(all_labels), torch.cat(all_bbox_targets)


def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, log_interval=20):
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        cls_logits = outputs["cls_logits"]
        bbox_deltas = outputs["bbox_deltas"][:, :4] if outputs["bbox_deltas"].numel() > 0 else outputs["bbox_deltas"]

        gt_labels, gt_boxes = match_proposals_to_targets(
            outputs["proposals"], cls_logits, outputs["bbox_deltas"], targets, model.num_classes
        )
        gt_labels = gt_labels.to(device)
        gt_boxes = gt_boxes.to(device)

        if cls_logits.size(0) == 0:
            continue

        loss, logs = criterion(cls_logits, gt_labels, bbox_deltas, gt_boxes)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % log_interval == 0:
            print(f"[Epoch {epoch}] step {step}/{len(dataloader)} "
                  f"loss={loss.item():.4f} cls={logs['cls_loss']:.4f} bbox={logs['bbox_loss']:.4f}")

    return running_loss / max(len(dataloader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train the V-PLOT DD-RCNN detector.")
    parser.add_argument("--train_json", type=str, default=PATHS.coco_train_json)
    parser.add_argument("--val_json", type=str, default=PATHS.coco_val_json)
    parser.add_argument("--image_dir", type=str, default=PATHS.keyframes_dir)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch_size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN.learning_rate)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--checkpoint_dir", type=str, default=PATHS.checkpoints_dir)
    args = parser.parse_args()

    device = torch.device(args.device)

    train_dataset = VPlotDataset(args.train_json, args.image_dir, input_size=MODEL.input_size)
    val_dataset = VPlotDataset(args.val_json, args.image_dir, input_size=MODEL.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=TRAIN.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=TRAIN.num_workers, collate_fn=collate_fn)

    model = DDRCNN(num_classes=CLASSES.num_classes).to(device)
    criterion = VPlotLoss(lambda_det=LOSS.lambda_det, lambda_cap=LOSS.lambda_cap,
                           smooth_l1_beta=LOSS.smooth_l1_beta)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=TRAIN.momentum, weight_decay=TRAIN.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=TRAIN.lr_step_size, gamma=TRAIN.lr_gamma)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device,
                                      epoch, log_interval=TRAIN.log_interval)
        val_metrics = evaluate(model, val_loader, device, CLASSES.names)
        scheduler.step()

        val_acc = val_metrics["macro"]["accuracy"]
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_accuracy={val_acc:.4f} val_f1={val_metrics['macro']['f1']:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "vplot_best.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_accuracy": val_acc}, ckpt_path)
            print(f"[INFO] Saved new best checkpoint to {ckpt_path} (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN.early_stopping_patience:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for "
                      f"{TRAIN.early_stopping_patience} epochs).")
                break

    final_ckpt = os.path.join(args.checkpoint_dir, "vplot_last.pth")
    torch.save({"epoch": epoch, "model_state": model.state_dict()}, final_ckpt)
    print(f"[INFO] Training complete. Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
