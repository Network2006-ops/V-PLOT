"""
dd_rcnn.py
----------
Deformable-Dilated Region Convolutional Neural Network (DD-RCNN): the core
detection module of V-PLOT (manuscript Section: DD-RCNN for Plot Detection).

Pipeline:
    Keyframe -> STN (geometric normalization)
             -> ResNeSt backbone (split-attention multi-scale features)
             -> DDConv module (dilated + deformable convolution)
             -> RPN (region proposals)
             -> ROI Align (fixed-size region features)
             -> classification head (plot-type logits)
             -> bbox regression head (box refinement)
"""

import torch
import torch.nn as nn

from stn import SpatialTransformerNetwork
from resnest_backbone import resnest50
from ddconv import DDConvModule
from rpn import RegionProposalNetwork
from roi_align import extract_roi_features


class DDRCNN(nn.Module):
    """End-to-end DD-RCNN detector used for graphical plot localization and
    classification in V-PLOT."""

    def __init__(self, num_classes=5, backbone_widths=(64, 128, 256, 512),
                 stn_in_channels=64, ddconv_out_channels=256,
                 dilation_rate=2, deform_kernel_size=3,
                 anchor_scales=(32, 64, 128, 256), anchor_ratios=(0.5, 1.0, 2.0),
                 roi_output_size=(7, 7), roi_sampling_ratio=2,
                 feature_stride=16):
        super().__init__()
        self.num_classes = num_classes
        self.feature_stride = feature_stride
        self.roi_output_size = roi_output_size

        self.backbone = resnest50(widths=backbone_widths)
        c4_channels = self.backbone.out_channels["C4"]

        # STN operates on an intermediate backbone feature map (C4) prior to DDConv.
        self.stn = SpatialTransformerNetwork(in_channels=c4_channels, hidden_dim=128)

        self.ddconv = DDConvModule(
            in_channels=c4_channels, out_channels=ddconv_out_channels,
            dilation_rate=dilation_rate, deform_kernel_size=deform_kernel_size
        )

        self.rpn = RegionProposalNetwork(
            in_channels=ddconv_out_channels, mid_channels=ddconv_out_channels,
            scales=anchor_scales, ratios=anchor_ratios
        )

        pooled_dim = ddconv_out_channels * roi_output_size[0] * roi_output_size[1]
        self.roi_sampling_ratio = roi_sampling_ratio

        self.classifier_head = nn.Sequential(
            nn.Linear(pooled_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(pooled_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes * 4),
        )

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, 3, H, W) batch of keyframes.

        Returns:
            dict with:
                proposals: list[Tensor(N_i, 4)] final region proposals per image
                scores: list[Tensor(N_i,)] objectness scores per image
                cls_logits: (sum N_i, num_classes) classification logits
                bbox_deltas: (sum N_i, num_classes*4) bbox regression outputs
                rpn_raw_cls, rpn_raw_bbox, anchors: RPN raw outputs for loss computation
        """
        img_h, img_w = images.shape[-2:]

        feats = self.backbone(images)
        c4 = feats["C4"]

        c4_aligned = self.stn(c4)                       # geometric normalization
        enriched = self.ddconv(c4_aligned)               # dilated-deformable features

        # Effective stride from input image to `enriched` feature map.
        stride = img_h / enriched.shape[-2]

        proposals, scores, raw_cls, raw_bbox, anchors = self.rpn(
            enriched, image_size=(img_h, img_w), stride=stride
        )

        spatial_scale = enriched.shape[-1] / img_w
        pooled = extract_roi_features(
            enriched, proposals, output_size=self.roi_output_size,
            spatial_scale=spatial_scale, sampling_ratio=self.roi_sampling_ratio
        )

        pooled_flat = pooled.flatten(1)
        if pooled_flat.size(0) == 0:
            cls_logits = pooled_flat.new_zeros((0, self.num_classes))
            bbox_deltas = pooled_flat.new_zeros((0, self.num_classes * 4))
        else:
            cls_logits = self.classifier_head(pooled_flat)
            bbox_deltas = self.bbox_head(pooled_flat)

        return {
            "proposals": proposals,
            "scores": scores,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "rpn_raw_cls": raw_cls,
            "rpn_raw_bbox": raw_bbox,
            "anchors": anchors,
            "roi_features": pooled_flat,
            "enriched_features": enriched,
        }


if __name__ == "__main__":
    model = DDRCNN(num_classes=5)
    dummy = torch.randn(2, 3, 512, 512)  # batch > 1 required by BatchNorm during training-mode smoke test
    out = model(dummy)
    print("proposals:", out["proposals"][0].shape)
    print("cls_logits:", out["cls_logits"].shape)
    print("bbox_deltas:", out["bbox_deltas"].shape)
