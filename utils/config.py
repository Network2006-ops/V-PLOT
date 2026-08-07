"""
config.py
---------
Central configuration for the V-PLOT pipeline: dataset paths, preprocessing
thresholds, model hyperparameters, and training settings. Values follow the
settings reported in the manuscript (Table 2 and Section: Results and
Discussion).
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PathConfig:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    videos_dir: str = os.path.join(root_dir, "dataset", "videos")
    keyframes_dir: str = os.path.join(root_dir, "dataset", "keyframes")
    annotations_dir: str = os.path.join(root_dir, "dataset", "annotations")
    captions_dir: str = os.path.join(root_dir, "dataset", "captions")
    checkpoints_dir: str = os.path.join(root_dir, "checkpoints")
    coco_train_json: str = os.path.join(annotations_dir, "train.json")
    coco_val_json: str = os.path.join(annotations_dir, "val.json")
    coco_test_json: str = os.path.join(annotations_dir, "test.json")


@dataclass
class PreprocessConfig:
    frame_sample_stride: int = 5          # sample every Nth frame before diffing
    mad_threshold: float = 0.12           # tau, empirically selected (Section: Keyframe extraction)
    resize_hw: Tuple[int, int] = (512, 512)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class PlotClasses:
    names: List[str] = field(default_factory=lambda: [
        "bar_chart", "line_chart", "pie_chart", "flow_chart", "tree_diagram"
    ])

    @property
    def num_classes(self) -> int:
        return len(self.names)


@dataclass
class ModelConfig:
    input_size: Tuple[int, int] = (512, 512)
    backbone: str = "resnest50"
    resnest_radix: int = 2                 # split-attention cardinality per group
    resnest_cardinality: int = 1
    resnest_widths: Tuple[int, ...] = (64, 128, 256, 512)

    # Dilated-Deformable convolution module
    dilation_rate: int = 2
    deform_kernel_size: int = 3
    ddconv_channels: int = 256

    # Spatial Transformer Network
    stn_in_channels: int = 64
    stn_loc_hidden: int = 128

    # Region Proposal Network
    anchor_scales: Tuple[int, ...] = (32, 64, 128, 256)
    anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    rpn_nms_thresh: float = 0.5
    rpn_pre_nms_top_n: int = 2000
    rpn_post_nms_top_n: int = 300

    # ROI Align
    roi_output_size: Tuple[int, int] = (7, 7)
    roi_sampling_ratio: int = 2

    # Temporal aggregation
    temporal_window: int = 2               # +/- frames
    iou_merge_thresh: float = 0.5

    # CLIP descriptor
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 512


@dataclass
class LossConfig:
    lambda_det: float = 1.0     # weight for detection loss (cls + bbox)
    lambda_cap: float = 1.0     # weight for CLIP contrastive caption loss
    cls_loss: str = "cross_entropy"
    bbox_loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    clip_temperature: float = 0.07


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    early_stopping_patience: int = 10
    num_workers: int = 4
    device: str = "cuda"
    log_interval: int = 20


PATHS = PathConfig()
PREPROCESS = PreprocessConfig()
CLASSES = PlotClasses()
MODEL = ModelConfig()
LOSS = LossConfig()
TRAIN = TrainConfig()
