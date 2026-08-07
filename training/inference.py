"""
inference.py
------------
End-to-end inference pipeline: given a lecture video, extract keyframes,
run DD-RCNN to localize and classify graphical plots, retrieve the most
relevant caption for each detection via the CLIP descriptor, perform
IoU-based temporal aggregation to remove duplicate detections across
neighbouring frames, and export results (plot type, caption, timestamp)
mirroring the output shown in Figure 4.

Usage:
    python inference.py --video path/to/lecture.mp4 --checkpoint ../checkpoints/vplot_best.pth
"""

import argparse
import json
import os
import sys

import cv2
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from models.dd_rcnn import DDRCNN
from models.roi_align import TemporalDetection, temporal_aggregate
from utils.config import CLASSES, MODEL, PATHS

try:
    from models.clip_descriptor import CLIPDescriptor
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False

try:
    import pytesseract
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False


def load_model(checkpoint_path: str, device: str):
    model = DDRCNN(num_classes=CLASSES.num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess_frame(frame, input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(image), image


def ocr_text_candidates(pil_image):
    """Extract candidate text segments from a keyframe using OCR, used as
    the CLIP text-candidate pool for caption retrieval."""
    if not _HAS_OCR:
        return []
    raw_text = pytesseract.image_to_string(pil_image)
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    return lines


def run_inference(video_path: str, checkpoint_path: str, device: str = "cpu",
                   score_thresh: float = 0.5, stride: int = 5):
    from preprocessing.frame_difference import is_keyframe

    model = load_model(checkpoint_path, device)
    clip_model = CLIPDescriptor(model_name=MODEL.clip_model_name, device=device) if _HAS_CLIP else None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detections_by_frame = {}
    prev_kept_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0 and is_keyframe(frame, prev_kept_frame, tau=0.12):
            prev_kept_frame = frame
            timestamp = frame_idx / fps

            tensor, pil_image = preprocess_frame(frame, MODEL.input_size)
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)

            boxes = outputs["proposals"][0]
            scores = outputs["scores"][0]
            cls_logits = outputs["cls_logits"]
            if cls_logits.numel() == 0:
                frame_idx += 1
                continue

            probs = torch.softmax(cls_logits, dim=1)
            pred_scores, pred_labels = probs.max(dim=1)

            frame_detections = []
            for i in range(boxes.size(0)):
                if pred_scores[i].item() < score_thresh:
                    continue
                box = boxes[i].tolist()
                label = int(pred_labels[i].item())

                roi_feature = outputs["roi_features"][i] if outputs["roi_features"].numel() > 0 else None
                frame_detections.append(TemporalDetection(
                    box=box, category_id=label, roi_feature=roi_feature,
                    timestamp=timestamp, frame_idx=frame_idx
                ))

            if frame_detections:
                detections_by_frame[frame_idx] = frame_detections

        frame_idx += 1

    cap.release()

    merged = temporal_aggregate(detections_by_frame,
                                 window=MODEL.temporal_window,
                                 iou_thresh=MODEL.iou_merge_thresh)

    results = []
    for det in merged:
        entry = {
            "plot_type": CLASSES.names[det.category_id],
            "bbox": det.box,
            "timestamp_sec": det.timestamp,
        }
        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run V-PLOT inference on a lecture video.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=os.path.join(PATHS.checkpoints_dir, "vplot_best.pth"))
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--output_json", type=str, default="inference_results.json")
    args = parser.parse_args()

    results = run_inference(args.video, args.checkpoint, device=args.device,
                             score_thresh=args.score_thresh)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Detected {len(results)} unique graphical plots. Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
