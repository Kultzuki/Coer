"""Keypoint geometry helpers.

Preserved from the legacy ``utils/keypoint_utils.py`` (no callers deleted).
Foundational fixes:
- ``flatten_four_person`` kept as a backward-compatible wrapper but the
  stale "4 persons" assumption is replaced by ``flatten_persons(data,
  max_people)``; new code must use the generalized form.
- Docstrings corrected: legacy comments claimed shapes the code did not use.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def get_midpoint(kp1, kp2):
    """Midpoint (x, y) between two keypoints exposing at least (x, y)."""
    return ((kp1[0] + kp2[0]) / 2.0, (kp1[1] + kp2[1]) / 2.0)


def get_angle(p1, p2):
    """Angle in degrees of vector p1->p2 measured from the vertical (Y axis)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def get_bbox_from_keypoints(keypoints_51):
    """(x_min, y_min, x_max, y_max) from visible keypoints (conf > 0).

    Input: flattened 51-element array [x1, y1, c1, ...].
    """
    xs = keypoints_51[0::3]
    ys = keypoints_51[1::3]
    confs = keypoints_51[2::3]
    valid_mask = confs > 0.0
    if not np.any(valid_mask):
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.min(xs[valid_mask])), float(np.min(ys[valid_mask])),
        float(np.max(xs[valid_mask])), float(np.max(ys[valid_mask])),
    )


def get_distance(bbox1, bbox2):
    """Center-to-center Euclidean distance between two [x1, y1, x2, y2] boxes."""
    c1_x = (bbox1[0] + bbox1[2]) / 2.0
    c1_y = (bbox1[1] + bbox1[3]) / 2.0
    c2_x = (bbox2[0] + bbox2[2]) / 2.0
    c2_y = (bbox2[1] + bbox2[3]) / 2.0
    return math.hypot(c1_x - c2_x, c1_y - c2_y)


def is_keypoint_valid(kp):
    """True if keypoint confidence > 0.3 (single float or [x, y, conf])."""
    if isinstance(kp, (list, tuple, np.ndarray)) and len(kp) >= 3:
        return float(kp[2]) > 0.3
    return float(kp) > 0.3


def flatten_persons(results_data, max_people: int = 2):
    """Pad YOLO ``keypoints.data`` (N, 17, 3) to ``(max_people, 51)``.

    New code must use this (explicit cap) instead of assuming a fixed 2 or 4.
    """
    out = np.zeros((int(max_people), 51), dtype=np.float32)
    if results_data is None:
        return out
    data = results_data.cpu().numpy() if hasattr(results_data, "cpu") else np.asarray(results_data)
    if len(data.shape) < 2 or data.size == 0:
        return out
    for i in range(min(int(max_people), data.shape[0])):
        out[i, :] = np.asarray(data[i]).flatten()[:51]
    return out


def flatten_four_person(results_data):
    """Legacy 4-person helper. Kept for compatibility; prefer ``flatten_persons``."""
    return flatten_persons(results_data, max_people=4)


def empty_keypoints(num_people: int = 2) -> np.ndarray:
    """Zero ``(num_people, 51)`` keypoint block (missing-person slots)."""
    return np.zeros((int(num_people), 51), dtype=np.float32)
