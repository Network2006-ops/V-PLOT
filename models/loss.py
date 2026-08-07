"""
loss.py
-------
Joint training objective for V-PLOT (manuscript Section: Loss function,
Eqs. 10-11):

    L_det = L_cls + L_bbox                          (Eq. 10)
    L_total = lambda_det * L_det + lambda_cap * L_cap (Eq. 11)

where:
    L_cls  : multi-class cross-entropy between predicted softmax scores and
             ground-truth plot-class labels.
    L_bbox : Smooth L1 loss between predicted and ground-truth box offsets.
    L_cap  : CLIP-based contrastive loss aligning image and caption embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_descriptor import clip_contrastive_loss


def classification_loss(pred_logits: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
    """Multi-class cross-entropy classification loss L_cls."""
    return F.cross_entropy(pred_logits, gt_labels)


def bbox_regression_loss(pred_deltas: torch.Tensor, gt_deltas: torch.Tensor,
                          beta: float = 1.0) -> torch.Tensor:
    """Smooth L1 (Huber) bounding-box regression loss L_bbox."""
    return F.smooth_l1_loss(pred_deltas, gt_deltas, beta=beta)


def detection_loss(pred_logits, gt_labels, pred_deltas, gt_deltas, beta=1.0):
    """Two-stage detection loss (Eq. 10): L_det = L_cls + L_bbox."""
    l_cls = classification_loss(pred_logits, gt_labels)
    l_bbox = bbox_regression_loss(pred_deltas, gt_deltas, beta=beta)
    return l_cls + l_bbox, {"cls_loss": l_cls.item(), "bbox_loss": l_bbox.item()}


class VPlotLoss(nn.Module):
    """
    Full multi-task V-PLOT training objective combining detection loss
    (classification + bbox regression) and CLIP caption-alignment loss
    (Eq. 11): L_total = lambda_det * L_det + lambda_cap * L_cap.
    """

    def __init__(self, lambda_det: float = 1.0, lambda_cap: float = 1.0,
                 smooth_l1_beta: float = 1.0, clip_temperature: float = 0.07):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_cap = lambda_cap
        self.smooth_l1_beta = smooth_l1_beta
        self.clip_temperature = clip_temperature

    def forward(self, pred_logits, gt_labels, pred_deltas, gt_deltas,
                image_embeds=None, text_embeds=None):
        l_det, det_logs = detection_loss(
            pred_logits, gt_labels, pred_deltas, gt_deltas, beta=self.smooth_l1_beta
        )

        if image_embeds is not None and text_embeds is not None:
            l_cap = clip_contrastive_loss(image_embeds, text_embeds, temperature=self.clip_temperature)
        else:
            l_cap = torch.tensor(0.0, device=pred_logits.device)

        total = self.lambda_det * l_det + self.lambda_cap * l_cap

        logs = {**det_logs, "cap_loss": float(l_cap), "total_loss": float(total)}
        return total, logs


if __name__ == "__main__":
    criterion = VPlotLoss(lambda_det=1.0, lambda_cap=1.0)

    pred_logits = torch.randn(8, 5)          # 5 plot classes
    gt_labels = torch.randint(0, 5, (8,))
    pred_deltas = torch.randn(8, 4)
    gt_deltas = torch.randn(8, 4)
    image_embeds = torch.randn(8, 512)
    text_embeds = torch.randn(8, 512)

    total_loss, logs = criterion(pred_logits, gt_labels, pred_deltas, gt_deltas,
                                  image_embeds, text_embeds)
    print("Total loss:", total_loss.item())
    print(logs)
