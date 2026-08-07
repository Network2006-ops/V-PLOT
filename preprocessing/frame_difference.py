"""
frame_difference.py
--------------------
Implements the Mean Absolute Difference (MAD) metric used to decide whether
a sampled video frame is sufficiently different from its predecessor to be
retained as a keyframe (manuscript Eq. 1):

    MAD(t) = (1 / (H * W)) * sum_{x,y} | I_t(x, y) - I_{t-1}(x, y) |

If MAD(t) exceeds the threshold tau, frame t is kept.
"""

import cv2
import numpy as np


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to single-channel grayscale float32."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return gray.astype(np.float32) / 255.0


def mean_absolute_difference(curr_frame: np.ndarray, prev_frame: np.ndarray) -> float:
    """
    Compute the Mean Absolute Difference between two frames (Eq. 1).

    Args:
        curr_frame: current frame, HxW or HxWx3 (uint8 or float)
        prev_frame: previous frame, same shape as curr_frame

    Returns:
        Scalar MAD value in [0, 1].
    """
    curr_gray = to_grayscale(curr_frame)
    prev_gray = to_grayscale(prev_frame)

    if curr_gray.shape != prev_gray.shape:
        prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))

    h, w = curr_gray.shape
    mad = np.sum(np.abs(curr_gray - prev_gray)) / (h * w)
    return float(mad)


def is_keyframe(curr_frame: np.ndarray, prev_frame: np.ndarray, tau: float = 0.12) -> bool:
    """Return True if MAD(curr, prev) exceeds threshold tau."""
    if prev_frame is None:
        return True
    return mean_absolute_difference(curr_frame, prev_frame) > tau


def batch_mad_scores(frames):
    """
    Compute MAD scores for a sequence of frames relative to their immediate
    predecessor. The first frame always has an undefined score (set to inf so
    that it is always retained).

    Args:
        frames: list/array of frames, ordered temporally.

    Returns:
        List[float] of length len(frames) with MAD scores.
    """
    scores = [float("inf")]
    for i in range(1, len(frames)):
        scores.append(mean_absolute_difference(frames[i], frames[i - 1]))
    return scores
