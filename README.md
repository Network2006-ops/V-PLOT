# V-PLOT: Video-based Plot Type Classification and Captioning via Deformable-Dilated Region Convolution Neural Network

This repository contains the reference PyTorch implementation accompanying the manuscript
**"V-PLOT: Video-based Plot Type Classification and Captioning via Deformable-Dilated Region
Convolution Neural Network"**, provided to support reviewer requests for source code.

## Overview

V-PLOT localizes, classifies, and captions graphical plots (bar charts, line charts, pie charts,
flow charts, tree diagrams) found in NPTEL educational lecture videos. The pipeline consists of:

1. **Keyframe extraction** — Mean Absolute Difference (MAD) frame filtering to discard
   near-duplicate slide frames (`preprocessing/`).
2. **Spatial Transformer Network (STN)** — geometric normalization of feature maps prior to
   detection (`models/stn.py`).
3. **DD-RCNN** — a Deformable-Dilated Region CNN with:
   - a **ResNeSt** split-attention backbone (`models/resnest_backbone.py`)
   - a **DDConv** dilated + deformable convolution module (`models/ddconv.py`)
   - a **Region Proposal Network (RPN)** (`models/rpn.py`)
   - **ROI Align** with IoU-based temporal aggregation across frames (`models/roi_align.py`)
4. **CLIP-based descriptor** — retrieval-based captioning that ranks OCR-extracted text
   candidates by cosine similarity to the detected region's image embedding
   (`models/clip_descriptor.py`).
5. **Joint loss** — detection loss (classification + Smooth L1 bbox regression) combined with a
   CLIP contrastive caption-alignment loss (`models/loss.py`).

## Repository structure

```
V-PLOT/
├── dataset/            # raw videos, extracted keyframes, COCO annotations, captions
├── preprocessing/       # keyframe extraction, MAD frame differencing, COCO conversion, splits
├── models/              # STN, ResNeSt, DDConv, DD-RCNN, RPN, ROIAlign, CLIP descriptor, losses
├── training/            # train / validate / test / inference scripts + dataset loader
├── utils/               # metrics, visualization, Grad-CAM, caption metrics, config
├── checkpoints/         # saved model weights (created during training)
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

GPU training requires a CUDA-enabled PyTorch build matching your driver version; see
https://pytorch.org for the appropriate install command.

## Data preparation

1. Place raw `.mp4`/`.avi` lecture videos in `dataset/videos/`.
2. Extract keyframes:
   ```bash
   python preprocessing/keyframe_extraction.py --video_dir dataset/videos --output_dir dataset/keyframes --tau 0.12
   ```
3. Annotate bounding boxes (class, coordinates, caption) into a flat CSV, then convert to COCO format:
   ```bash
   python preprocessing/coco_annotation.py --csv raw_labels.csv --image_dir dataset/keyframes \
       --captions_dir dataset/captions --output_json dataset/annotations/full.json
   ```
4. Split into train/val/test (70:15:15, class-stratified):
   ```bash
   python preprocessing/data_split.py --coco_json dataset/annotations/full.json --out_dir dataset/annotations
   ```

## Training

```bash
python training/train.py \
    --train_json dataset/annotations/train.json \
    --val_json dataset/annotations/val.json \
    --image_dir dataset/keyframes \
    --epochs 100 --batch_size 32 --lr 0.001
```

Hyperparameters (Table 2 of the manuscript) default to: 512×512 input resolution, 100 epochs,
learning rate 0.001, momentum 0.9, weight decay 0.0001, 7×7 ROI Align output, dilation rate 2,
3×3 deformable offset kernels, and RPN NMS threshold 0.5. These are set in `utils/config.py`.

## Evaluation

```bash
python training/test.py --test_json dataset/annotations/test.json \
    --image_dir dataset/keyframes --checkpoint checkpoints/vplot_best.pth
```

Reports per-class precision, accuracy, recall, F1-score, and specificity (Eqs. 13–17), and can
render a normalized confusion matrix (Figure 7 style) via `utils/visualization.py`.

Caption quality (BLEU-4, METEOR, ROUGE-L, CIDEr — Eqs. 18–21) can be computed with
`utils/caption_metrics.py::evaluate_captions`.

## Inference on a new video

```bash
python training/inference.py --video path/to/lecture.mp4 --checkpoint checkpoints/vplot_best.pth
```

Produces a JSON list of detected plots with plot type, bounding box, and timestamp, after
IoU-based temporal aggregation (Eq. 9) removes duplicate detections across neighbouring frames.

## Interpretability

`utils/gradcam.py` implements Grad-CAM for visualizing which spatial regions of a keyframe drive
the classification decision (Figure 5 in the manuscript).

## Notes for reviewers

- This codebase implements the architecture and equations exactly as described in the manuscript
  (STN affine transform in Eqs. 2–4, split-attention aggregation in Eq. 5, dilated/deformable
  convolution in Eqs. 6–7, RPN proposal decoding in Eq. 8, IoU-based temporal aggregation in
  Eq. 9, the joint loss in Eqs. 10–11, CLIP cosine-similarity retrieval in Eq. 12, and the
  evaluation metrics in Eqs. 13–21).
- The NPTEL video/annotation dataset itself is not redistributed here (see the manuscript's
  "Availability of data and material" statement); `dataset/` contains the expected folder layout
  and format for reproducing the pipeline on a locally gathered dataset.
- Some engineering choices (e.g. the simplified proposal-to-ground-truth matching in
  `training/train.py`, or the pure-Python IoU/temporal-aggregation loop in
  `models/roi_align.py`) prioritize readability and faithfulness to the manuscript's equations
  over production-level runtime efficiency; they can be replaced with vectorized/Detectron2-style
  samplers for large-scale training without changing the model definitions.
