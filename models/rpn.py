"""
rpn.py
------
Region Proposal Network (RPN) that slides a small network over the enriched
DDConv feature map to output an objectness score and bounding-box regression
offsets for each anchor (manuscript Section: Region Proposal Network, Eq. 8):

    p_i = sigmoid(objectness logit for anchor i)
    t_i = (tx, ty, tw, th) predicted bounding-box adjustments
    b_i = decode(anchor_i, t_i)   -> refined proposal coordinates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


def generate_anchors(scales, ratios, base_size=16):
    """Generate a set of anchor boxes (w, h) for a single spatial location,
    centred at the origin, for the given scales and aspect ratios."""
    anchors = []
    for scale in scales:
        area = (scale) ** 2
        for ratio in ratios:
            w = (area / ratio) ** 0.5
            h = w * ratio
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def shift_anchors(anchors, feat_h, feat_w, stride):
    """Tile the base anchors over every spatial location of the feature map."""
    shift_x = (torch.arange(0, feat_w, dtype=torch.float32) + 0.5) * stride
    shift_y = (torch.arange(0, feat_h, dtype=torch.float32) + 0.5) * stride
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                           shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)

    A = anchors.size(0)
    K = shifts.size(0)
    all_anchors = anchors.view(1, A, 4) + shifts.view(K, 1, 4)
    return all_anchors.view(K * A, 4)


def decode_boxes(anchors, deltas):
    """
    Apply predicted deltas (tx, ty, tw, th) to anchor boxes to obtain refined
    proposals (Eq. 8): b_i = anchor_center/size adjusted by t_i.
    """
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack([x1, y1, x2, y2], dim=1)


class RegionProposalNetwork(nn.Module):
    """Sliding-window RPN head producing objectness scores and bbox deltas
    per anchor, followed by NMS-based proposal selection (Eq. 8)."""

    def __init__(self, in_channels=256, mid_channels=256,
                 scales=(32, 64, 128, 256), ratios=(0.5, 1.0, 2.0),
                 nms_thresh=0.5, pre_nms_top_n=2000, post_nms_top_n=300):
        super().__init__()
        self.anchors_base = generate_anchors(scales, ratios)
        self.num_anchors = self.anchors_base.size(0)
        self.nms_thresh = nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n

        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, self.num_anchors, kernel_size=1)     # objectness p_i
        self.bbox_pred = nn.Conv2d(mid_channels, self.num_anchors * 4, kernel_size=1)  # deltas t_i

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feature_map, image_size, stride=16):
        """
        Args:
            feature_map: enriched DDConv feature map, (B, C, H, W)
            image_size: (img_h, img_w) of the original image, for clipping.
            stride: effective stride of feature_map relative to the input image.

        Returns:
            proposals: list length B, each a tensor (N, 4) of kept boxes.
            scores: list length B, each a tensor (N,) of objectness scores.
            raw_cls: objectness logits (for loss computation)
            raw_bbox: bbox deltas (for loss computation)
        """
        b, _, h, w = feature_map.shape
        t = F.relu(self.conv(feature_map))
        raw_cls = self.cls_logits(t)     # (B, num_anchors, H, W)
        raw_bbox = self.bbox_pred(t)     # (B, num_anchors*4, H, W)

        anchors = shift_anchors(self.anchors_base, h, w, stride).to(feature_map.device)

        cls = raw_cls.permute(0, 2, 3, 1).reshape(b, -1)                 # (B, H*W*A)
        bbox = raw_bbox.permute(0, 2, 3, 1).reshape(b, -1, 4)            # (B, H*W*A, 4)
        obj_scores = torch.sigmoid(cls)

        proposals_batch, scores_batch = [], []
        img_h, img_w = image_size
        for i in range(b):
            deltas = bbox[i]
            scores_i = obj_scores[i]

            top_n = min(self.pre_nms_top_n, scores_i.numel())
            top_scores, top_idx = scores_i.topk(top_n)
            top_anchors = anchors[top_idx]
            top_deltas = deltas[top_idx]

            boxes = decode_boxes(top_anchors, top_deltas)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)

            keep = nms(boxes, top_scores, self.nms_thresh)[:self.post_nms_top_n]
            proposals_batch.append(boxes[keep])
            scores_batch.append(top_scores[keep])

        return proposals_batch, scores_batch, raw_cls, raw_bbox, anchors


if __name__ == "__main__":
    rpn = RegionProposalNetwork(in_channels=256)
    dummy = torch.randn(1, 256, 32, 32)
    proposals, scores, raw_cls, raw_bbox, anchors = rpn(dummy, image_size=(512, 512), stride=16)
    print("proposals:", proposals[0].shape, "scores:", scores[0].shape)
