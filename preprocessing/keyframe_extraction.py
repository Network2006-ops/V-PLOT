"""
keyframe_extraction.py
-----------------------
Extracts keyframes from NPTEL lecture videos. Frames are sampled at a fixed
stride, converted to grayscale, and compared to the previously retained
keyframe using the Mean Absolute Difference (MAD) metric (see
frame_difference.py, Eq. 1). Frames whose MAD exceeds the threshold `tau`
are saved as keyframes, discarding near-duplicate slide frames caused by
transitions or static narration.
"""

import argparse
import os

import cv2
from tqdm import tqdm

from frame_difference import is_keyframe

try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.config import PATHS, PREPROCESS
except Exception:
    PATHS = None
    PREPROCESS = None


def extract_keyframes(video_path: str, output_dir: str,
                       stride: int = 5, tau: float = 0.12,
                       resize_hw=(512, 512)) -> list:
    """
    Extract keyframes from a single video file.

    Args:
        video_path: path to the input video.
        output_dir: directory to save extracted keyframe images.
        stride: sample every `stride`-th frame before applying MAD filtering.
        tau: MAD threshold above which a frame is treated as a keyframe.
        resize_hw: (height, width) to resize saved keyframes.

    Returns:
        List of saved keyframe file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    saved_paths = []
    prev_kept_frame = None
    frame_idx = 0
    keyframe_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    with tqdm(total=total_frames, desc=f"Scanning {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                if is_keyframe(frame, prev_kept_frame, tau=tau):
                    resized = cv2.resize(frame, (resize_hw[1], resize_hw[0]))
                    timestamp_sec = frame_idx / fps
                    fname = f"{video_name}_kf{keyframe_idx:05d}_t{timestamp_sec:.2f}.jpg"
                    out_path = os.path.join(output_dir, fname)
                    cv2.imwrite(out_path, resized)
                    saved_paths.append(out_path)
                    prev_kept_frame = frame
                    keyframe_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return saved_paths


def extract_from_directory(video_dir: str, output_dir: str,
                            stride: int = 5, tau: float = 0.12):
    """Run keyframe extraction over every video file in `video_dir`."""
    video_extensions = (".mp4", ".avi", ".mkv", ".mov")
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]

    all_saved = {}
    for video_file in videos:
        video_path = os.path.join(video_dir, video_file)
        video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
        saved = extract_keyframes(video_path, video_output_dir, stride=stride, tau=tau)
        all_saved[video_file] = saved
        print(f"[INFO] {video_file}: {len(saved)} keyframes extracted")

    return all_saved


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from NPTEL lecture videos.")
    parser.add_argument("--video_dir", type=str, default=(PATHS.videos_dir if PATHS else "dataset/videos"))
    parser.add_argument("--output_dir", type=str, default=(PATHS.keyframes_dir if PATHS else "dataset/keyframes"))
    parser.add_argument("--stride", type=int, default=(PREPROCESS.frame_sample_stride if PREPROCESS else 5))
    parser.add_argument("--tau", type=float, default=(PREPROCESS.mad_threshold if PREPROCESS else 0.12))
    args = parser.parse_args()

    extract_from_directory(args.video_dir, args.output_dir, stride=args.stride, tau=args.tau)


if __name__ == "__main__":
    main()
