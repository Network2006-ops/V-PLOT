"""
roi_align.py
------------
ROI Align stage that converts variable-sized region proposals into
fixed-size feature maps via bilinear interpolation (avoiding the
quantization error of ROI Pooling), plus the IoU-based temporal
aggregation mechanism used to merge duplicate detections of the same
graphical object across neighbouring frames (manuscript Section: Region
Proposal Network / temporal aggregation, Eq. 9).
"""

import torch
from torchvision.ops import roi_align


def extract_roi_features(feature_map, proposals, output_size=(7, 7),
                          spatial_scale=1.0, sampling_ratio=2):
    """
    Apply ROI Align to obtain fixed-size feature maps for a batch of
    proposals.

    Args:
        feature_map: (B, C, H, W) enriched feature map.
        proposals: list length B of (N_i, 4) boxes in image coordinates
                   (x1, y1, x2, y2).
        output_size: fixed spatial output size, e.g. (7, 7) per Table 2.
        spatial_scale: ratio of feature_map size to original image size
                       (1 / stride).
        sampling_ratio: number of sampling points per bin (bilinear
                        interpolation grid density).

    Returns:
        Tensor of shape (sum(N_i), C, output_h, output_w).
    """
    boxes_with_index = []
    for i, boxes in enumerate(proposals):
        if boxes.numel() == 0:
            continue
        idx_col = torch.full((boxes.size(0), 1), i, dtype=boxes.dtype, device=boxes.device)
        boxes_with_index.append(torch.cat([idx_col, boxes], dim=1))

    if len(boxes_with_index) == 0:
        c = feature_map.size(1)
        return feature_map.new_zeros((0, c, output_size[0], output_size[1]))

    rois = torch.cat(boxes_with_index, dim=0)
    pooled = roi_align(feature_map, rois, output_size=output_size,
                        spatial_scale=spatial_scale,
                        sampling_ratio=sampling_ratio, aligned=True)
    return pooled


def compute_iou(box_a, box_b):
    """
    Intersection-over-Union between two axis-aligned boxes (Eq. 9):
        IoU(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        box_a, box_b: (4,) tensors/arrays [x1, y1, x2, y2].
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class TemporalDetection:
    """Container for a single detected graphical region with its bounding
    box, class, ROI feature vector, and originating frame timestamp."""

    def __init__(self, box, category_id, roi_feature, timestamp, frame_idx):
        self.box = box                      # [x1, y1, x2, y2]
        self.category_id = category_id
        self.roi_feature = roi_feature       # pooled ROI feature tensor
        self.timestamp = timestamp
        self.frame_idx = frame_idx


def temporal_aggregate(detections_by_frame, window=2, iou_thresh=0.5):
    """
    Merge duplicate detections of the same graphical object across a
    temporal window of neighbouring frames using IoU-based matching
    (manuscript Eq. 9 and surrounding temporal-aggregation description).

    Args:
        detections_by_frame: dict {frame_idx: List[TemporalDetection]},
            ordered by frame_idx.
        window: +/- number of neighbouring frames to consider for merging.
        iou_thresh: IoU threshold above which two detections are merged.

    Returns:
        List[TemporalDetection]: a de-duplicated set of representative
        detections, each with an averaged bounding box, averaged ROI
        feature, and a representative timestamp taken from the central
        frame of the merged window.
    """
    frame_indices = sorted(detections_by_frame.keys())
    visited = set()
    merged_detections = []

    for f in frame_indices:
        for det in detections_by_frame[f]:
            key = (f, tuple(det.box), det.category_id)
            if key in visited:
                continue

            cluster = [det]
            visited.add(key)

            for delta in range(-window, window + 1):
                neighbor_f = f + delta
                if neighbor_f == f or neighbor_f not in detections_by_frame:
                    continue
                for cand in detections_by_frame[neighbor_f]:
                    cand_key = (neighbor_f, tuple(cand.box), cand.category_id)
                    if cand_key in visited or cand.category_id != det.category_id:
                        continue
                    if compute_iou(det.box, cand.box) > iou_thresh:
                        cluster.append(cand)
                        visited.add(cand_key)

            avg_box = [sum(d.box[i] for d in cluster) / len(cluster) for i in range(4)]
            avg_feature = sum(d.roi_feature for d in cluster) / len(cluster)
            central = cluster[len(cluster) // 2]

            merged_detections.append(TemporalDetection(
                box=avg_box,
                category_id=det.category_id,
                roi_feature=avg_feature,
                timestamp=central.timestamp,
                frame_idx=central.frame_idx,
            ))

    return merged_detections


if __name__ == "__main__":
    feat = torch.randn(1, 256, 32, 32)
    boxes = [torch.tensor([[10., 10., 100., 100.], [50., 50., 200., 200.]])]
    pooled = extract_roi_features(feat, boxes, output_size=(7, 7), spatial_scale=1 / 16)
    print("ROI pooled shape:", pooled.shape)
