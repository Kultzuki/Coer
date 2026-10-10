"""Canonical 2D skeleton format. The single joint definition for VIGLANCE.

YOLO pose already emits COCO-17 order, so the YOLO -> canonical index map is
the identity; it is written out explicitly (YOLO_TO_CANONICAL) so every
downstream module shares ONE definition instead of re-deriving joint order.

COCO indices: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip, 13 left_knee,
14 right_knee, 15 left_ankle, 16 right_ankle.

numpy-only. Normalization never yields NaN/Inf (clamped, guarded fallbacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

COCO_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
NUM_JOINTS = len(COCO_KEYPOINT_NAMES)

YOLO_TO_CANONICAL: Tuple[int, ...] = tuple(range(NUM_JOINTS))

# Single bone topology, matching PYSKL/MMAction2 `layout='coco'` inward edges
# (undirected here for drawing + bone features). Verified against
# pyskl/utils/graph.py and mmaction/models/utils/graph.py: shoulders link to
# the nose (5-0, 6-0), hips link via shoulders (11-5, 12-6); there are NO
# shoulder-shoulder or hip-hip links. Using any other topology would
# silently mismatch the graph adjacency the checkpoints were trained with.
COCO_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
NUM_BONES = len(COCO_EDGES)

# JointToBone pairs, exactly as PYSKL `JointToBone` / MMAction2 pose
# transforms define them for COCO: one bone per joint (V entries); the root
# bone (0,0) is the zero vector. Score channel = mean of the two scores.
PYSKL_BONE_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0),
    (7, 5), (8, 6), (9, 7), (10, 8), (11, 0), (12, 0),
    (13, 11), (14, 12), (15, 13), (16, 14),
)

_MIN_SCALE = 1e-3
_MAX_COORD = 1e4


@dataclass
class CanonicalSkeleton:
    xy: np.ndarray            # (17, 2) float32, root-centered, scale-normalized
    confidence: np.ndarray    # (17,) float32 raw detector confidences
    presence: np.ndarray      # (17,) bool, confidence >= threshold
    bbox: List[float]         # [x1, y1, x2, y2] pixels
    root: np.ndarray          # (2,) float32 pixel-space root used for centering
    scale: float              # body scale in pixels used for normalization
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32).reshape(NUM_JOINTS, 2)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(NUM_JOINTS)
        self.presence = np.asarray(self.presence, dtype=bool).reshape(NUM_JOINTS)
        self.root = np.asarray(self.root, dtype=np.float32).reshape(2)

    def center(self) -> np.ndarray:
        s = self.scale if self.scale > 0 else 1.0
        return (np.asarray(self.bbox[:2]) + np.asarray(self.bbox[2:])) / 2.0 / s


def _finite(v: float, default: float = 0.0) -> float:
    if v is None or not np.isfinite(v):
        return default
    return float(v)


def normalize_person(person, root_conf: float = 0.3,
                     joint_conf: float = 0.3) -> CanonicalSkeleton:
    kpts = np.asarray(person.keypoints, dtype=np.float32).reshape(NUM_JOINTS, 3)
    kpts = np.nan_to_num(kpts, nan=0.0, posinf=0.0, neginf=0.0)
    xy_raw = kpts[:, :2]
    conf = np.clip(kpts[:, 2], 0.0, 1.0)
    presence = conf >= joint_conf
    bbox = [float(v) for v in person.bbox]
    if len(bbox) != 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    bbox = [_finite(v) for v in bbox]

    ls, rs, lh, rh = xy_raw[5], xy_raw[6], xy_raw[11], xy_raw[12]
    hips_ok = bool(conf[11] >= root_conf and conf[12] >= root_conf)
    sh_ok = bool(conf[5] >= root_conf and conf[6] >= root_conf)
    if hips_ok:
        root = (lh + rh) / 2.0
    elif sh_ok:
        root = (ls + rs) / 2.0
    else:
        root = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                        dtype=np.float32)
    root = np.nan_to_num(root, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    scale = 0.0
    if hips_ok and sh_ok:
        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    if not np.isfinite(scale) or scale <= 0.0:
        spans = []
        if sh_ok:
            spans.append(float(np.linalg.norm(ls - rs)))
        if hips_ok:
            spans.append(float(np.linalg.norm(lh - rh)))
        scale = max(spans) if spans else 0.0
    bbox_h = bbox[3] - bbox[1]
    if (not np.isfinite(scale) or scale <= 0.0) and np.isfinite(bbox_h) and bbox_h > 0.0:
        scale = float(bbox_h)
    if not np.isfinite(scale) or scale < _MIN_SCALE:
        scale = 1.0

    norm = (xy_raw - root) / scale
    norm = np.nan_to_num(norm, nan=0.0, posinf=_MAX_COORD, neginf=-_MAX_COORD)
    norm = np.clip(norm, -_MAX_COORD, _MAX_COORD).astype(np.float32)
    norm[~presence] = 0.0
    return CanonicalSkeleton(xy=norm, confidence=conf, presence=presence,
                             bbox=bbox, root=root, scale=float(scale),
                             timestamp=float(getattr(person, "timestamp", 0.0)))


def bones_from_joints(xy: np.ndarray, presence: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32).reshape(NUM_JOINTS, 2)
    presence = np.asarray(presence, dtype=bool).reshape(NUM_JOINTS)
    out = np.zeros((NUM_BONES, 2), dtype=np.float32)
    for i, (a, b) in enumerate(COCO_EDGES):
        if presence[a] and presence[b]:
            out[i] = xy[b] - xy[a]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def bones_pyskl(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Bone stream exactly as PYSKL/MMAction2 `JointToBone` for COCO.

    Returns (17, 3): bone[v] = xy[v] - xy[center[v]], score = mean of the
    two joint scores; root bone (0,0) is zero. This is the input layout the
    released bone-stream checkpoints expect (in_channels=3, V=17).
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(NUM_JOINTS, 2)
    conf = np.clip(np.asarray(conf, dtype=np.float32).reshape(NUM_JOINTS), 0.0, 1.0)
    out = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for v, (joint, center) in enumerate(PYSKL_BONE_PAIRS):
        out[v, :2] = xy[joint] - xy[center]
        out[v, 2] = (conf[joint] + conf[center]) / 2.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def frame_normalize(xy_px: np.ndarray, conf: np.ndarray,
                    img_hw: Tuple[float, float],
                    threshold: float = 0.01) -> np.ndarray:
    """Model-input normalization exactly as PYSKL `PreNormalize2D` ('fix').

    x = (x - w/2)/(w/2), y = (y - h/2)/(h/2) with img_hw=(h, w); joints with
    score <= threshold get x,y = 0; score channel passes through untouched.
    Released 2D checkpoints were trained on this distribution, NOT on
    body-scale-normalized coordinates, so the adapter must use this for
    model input. Returns (17, 3) = (x_norm, y_norm, score).
    """
    h, w = float(img_hw[0]), float(img_hw[1])
    w = w if np.isfinite(w) and w > 0 else 1920.0
    h = h if np.isfinite(h) and h > 0 else 1080.0
    xy = np.nan_to_num(np.asarray(xy_px, dtype=np.float32).reshape(NUM_JOINTS, 2),
                       nan=0.0, posinf=0.0, neginf=0.0)
    sc = np.clip(np.asarray(conf, dtype=np.float32).reshape(NUM_JOINTS), 0.0, 1.0)
    out = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    out[:, 0] = (xy[:, 0] - w / 2.0) / (w / 2.0)
    out[:, 1] = (xy[:, 1] - h / 2.0) / (h / 2.0)
    out[:, 2] = sc
    out[sc <= threshold, :2] = 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def motion_from_pair(prev_xy: np.ndarray, curr_xy: np.ndarray,
                     prev_presence: np.ndarray, curr_presence: np.ndarray,
                     dt_sec: float) -> np.ndarray:
    dt = float(dt_sec)
    if not np.isfinite(dt) or dt <= 1e-6:
        return np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    prev_xy = np.asarray(prev_xy, dtype=np.float32).reshape(NUM_JOINTS, 2)
    curr_xy = np.asarray(curr_xy, dtype=np.float32).reshape(NUM_JOINTS, 2)
    valid = (np.asarray(prev_presence, dtype=bool).reshape(NUM_JOINTS)
             & np.asarray(curr_presence, dtype=bool).reshape(NUM_JOINTS))
    vel = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    vel[valid] = (curr_xy[valid] - prev_xy[valid]) / dt
    return np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)


def motion_sequence(seq: np.ndarray) -> np.ndarray:
    """PYSKL `ToMotion` convention over a (T, V, C) stream.

    motion[t] = seq[t+1] - seq[t] on x/y, score = mean of the two adjacent
    scores; last frame is all zeros. (Internal diagnostic velocity in
    pairs.py intentionally differs: it divides by dt for irregular clocks.)
    """
    arr = np.nan_to_num(np.asarray(seq, dtype=np.float32), nan=0.0,
                        posinf=0.0, neginf=0.0)
    t = arr.shape[0]
    out = np.zeros_like(arr)
    if t >= 2:
        out[:-1, :, :2] = arr[1:, :, :2] - arr[:-1, :, :2]
        if arr.shape[2] > 2:
            out[:-1, :, 2] = (arr[1:, :, 2] + arr[:-1, :, 2]) / 2.0
    return out


def modality_pack(skel: CanonicalSkeleton, prev: CanonicalSkeleton | None,
                  dt_sec: float) -> dict:
    joint = skel.xy
    bone = bones_from_joints(skel.xy, skel.presence)
    if prev is None:
        jm = np.zeros_like(joint)
        bm = np.zeros_like(bone)
    else:
        jm = motion_from_pair(prev.xy, skel.xy, prev.presence, skel.presence, dt_sec)
        prev_bone = bones_from_joints(prev.xy, prev.presence)
        both = np.ones(NUM_BONES, dtype=bool)
        for i, (a, b) in enumerate(COCO_EDGES):
            both[i] = bool(prev.presence[a] and prev.presence[b]
                           and skel.presence[a] and skel.presence[b])
        bm = np.zeros_like(bone)
        if dt_sec and np.isfinite(dt_sec) and dt_sec > 1e-6:
            bm[both] = (bone[both] - prev_bone[both]) / dt_sec
        bm = np.nan_to_num(bm, nan=0.0, posinf=0.0, neginf=0.0)
    return {"joint": joint, "bone": bone, "joint_motion": jm, "bone_motion": bm}
