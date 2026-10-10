"""Dynamic candidate person-pair generation, identity, and temporal buffers.

Replaces the legacy first-two-people selection: every ACTIVE person is
eligible, pairs are filtered by deterministic spatial rules, identity is the
canonical (track_id, generation) key, and each pair owns a timestamped
history that dies with the pair (grace period for transient LOST).

numpy-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logging_utils import get_logger
from core.skeleton import CanonicalSkeleton
from core.tracked_person import TrackedPerson

log = get_logger("pairs")


def pair_key_for(a: TrackedPerson, b: TrackedPerson) -> Tuple[int, int, int, int]:
    ka = (int(a.track_id), int(a.generation))
    kb = (int(b.track_id), int(b.generation))
    lo, hi = (ka, kb) if ka <= kb else (kb, ka)
    return (lo[0], lo[1], hi[0], hi[1])


def pair_key_name(key: Tuple[int, int, int, int]) -> str:
    return f"P{key[0]}g{key[1]}-P{key[2]}g{key[3]}"


def bbox_iou(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def _center(bbox: List[float]) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                    dtype=np.float64)


def candidate_score_inputs(a: TrackedPerson, b: TrackedPerson,
                           scale_a: float, scale_b: float) -> dict:
    ca, cb = _center(a.bbox), _center(b.bbox)
    mean_scale = max((scale_a + scale_b) / 2.0, 1e-6)
    dist_px = float(np.linalg.norm(ca - cb))
    return {
        "dist_px": dist_px,
        "norm_dist": dist_px / mean_scale,
        "iou": bbox_iou(a.bbox, b.bbox),
        "vis": min(float(a.visibility_ratio), float(b.visibility_ratio)),
    }


def generate_candidate_pairs(active_people: List[TrackedPerson],
                             skeletons: Dict[int, CanonicalSkeleton],
                             pair_max_distance: float,
                             min_person_visibility: float,
                             max_candidate_pairs: int) -> List[dict]:
    people = sorted(active_people, key=lambda p: (p.track_id, p.generation))
    scored = []
    n = len(people)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = people[i], people[j]
            sa = skeletons.get(a.track_id)
            sb = skeletons.get(b.track_id)
            if sa is None or sb is None:
                continue
            info = candidate_score_inputs(a, b, sa.scale, sb.scale)
            if info["vis"] < min_person_visibility:
                continue
            if info["norm_dist"] > pair_max_distance:
                continue
            scored.append((info["norm_dist"], a, b, info))
    scored.sort(key=lambda t: (t[0], t[1].track_id, t[2].track_id))
    out = []
    for _, a, b, info in scored[:max(0, int(max_candidate_pairs))]:
        out.append({"key": pair_key_for(a, b), "a": a, "b": b, "filter": info})
    return out


def compute_pair_features(skel_a: CanonicalSkeleton, skel_b: CanonicalSkeleton,
                          prev: Optional[dict], dt_sec: float) -> dict:
    ca = _center(skel_a.bbox)
    cb = _center(skel_b.bbox)
    mean_scale = max((skel_a.scale + skel_b.scale) / 2.0, 1e-6)
    offset = (cb - ca) / mean_scale
    norm_dist = float(np.linalg.norm(offset))
    iou = bbox_iou(skel_a.bbox, skel_b.bbox)
    dmin = float("inf")
    for i in range(skel_a.xy.shape[0]):
        if skel_a.presence[i] and skel_b.presence[i]:
            d = float(np.linalg.norm(skel_a.xy[i] - skel_b.xy[i]))
            if d < dmin:
                dmin = d
    if not np.isfinite(dmin):
        dmin = -1.0
    scale_ratio = 0.0
    if skel_a.scale > 0 and skel_b.scale > 0:
        scale_ratio = float(max(skel_a.scale, skel_b.scale)
                            / max(min(skel_a.scale, skel_b.scale), 1e-6))
    rel_vel, closing = 0.0, 0.0
    if prev is not None and dt_sec and dt_sec > 1e-6:
        va = (ca - prev["ca"]) / dt_sec / mean_scale
        vb = (cb - prev["cb"]) / dt_sec / mean_scale
        rel_vel = float(np.linalg.norm(va - vb))
        closing = float((prev["dist"] - norm_dist) / dt_sec)
    feats = {
        "center_offset_x": float(offset[0]),
        "center_offset_y": float(offset[1]),
        "normalized_center_distance": norm_dist,
        "bbox_overlap": iou,
        "minimum_joint_distance": float(dmin),
        "relative_velocity": float(rel_vel),
        "closing_speed": float(closing),
        "scale_ratio": float(scale_ratio),
    }
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    return feats


@dataclass
class PairTrack:
    key: Tuple[int, int, int, int]
    id_a: int
    id_b: int
    created_at: float = 0.0
    last_seen: float = 0.0
    active: bool = True
    frame_count: int = 0
    buffer: List[dict] = field(default_factory=list)
    latest_features: dict = field(default_factory=dict)
    _prev_centers: Optional[dict] = None
    max_history_seconds: float = 4.0

    def observe(self, skel_a: CanonicalSkeleton, skel_b: CanonicalSkeleton,
                timestamp: float) -> dict:
        ts = float(timestamp)
        if self.frame_count == 0:
            self.created_at = ts
        dt = ts - self.last_seen if self.frame_count > 0 else 0.0
        feats = compute_pair_features(skel_a, skel_b, self._prev_centers,
                                      dt if dt > 0 else 0.0)
        self.buffer.append({"t": ts, "a": skel_a, "b": skel_b,
                            "features": feats})
        if self.max_history_seconds > 0.0 and len(self.buffer) > 1:
            # Retain one extra frame beyond the window: trimming at exactly
            # max_history_seconds caps span just UNDER any equal-length
            # readiness window (off by one frame interval), which would make
            # full-coverage readiness unreachable no matter how long the
            # pair is observed. Slack = largest observed gap, so a full
            # window stays achievable at any frame rate.
            times = [float(entry["t"]) for entry in self.buffer]
            slack = max(b - a for a, b in zip(times[:-1], times[1:]))
            slack = slack if slack > 1e-6 else 1e-3
            cutoff = ts - self.max_history_seconds - slack
            self.buffer = [entry for entry in self.buffer
                           if float(entry["t"]) >= cutoff]
        self.latest_features = feats
        self._prev_centers = {"ca": _center(skel_a.bbox), "cb": _center(skel_b.bbox),
                              "dist": feats["normalized_center_distance"]}
        self.last_seen = ts
        self.frame_count += 1
        self.active = True
        return feats

    def close(self) -> None:
        self.active = False
        self.buffer.clear()
        self._prev_centers = None
        self.latest_features = {}


class PairManager:
    def __init__(self, pair_grace_seconds: float = 0.5,
                 history_seconds: float = 4.0):
        self.pair_grace_seconds = float(pair_grace_seconds)
        self.history_seconds = float(history_seconds)
        if self.history_seconds <= 0.0:
            raise ValueError("history_seconds must be > 0")
        self.tracks: Dict[Tuple[int, int, int, int], PairTrack] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, candidates: List[dict],
               skeletons: Dict[int, CanonicalSkeleton],
               timestamp: float) -> List[PairTrack]:
        ts = float(timestamp)
        seen = set()
        live: List[PairTrack] = []
        for cand in candidates:
            key = cand["key"]
            seen.add(key)
            track = self.tracks.get(key)
            if track is None or not track.active:
                a, b = cand["a"], cand["b"]
                track = PairTrack(
                    key=key, id_a=a.track_id, id_b=b.track_id,
                    max_history_seconds=self.history_seconds,
                )
                self.tracks[key] = track
            track.observe(skeletons[track.id_a], skeletons[track.id_b], ts)
            live.append(track)
        for key, track in self.tracks.items():
            if key in seen or not track.active:
                continue
            if ts - track.last_seen > self.pair_grace_seconds:
                track.close()
                log.info("event=pair_closed pair=%s ts=%.3f", pair_key_name(key), ts)
        return live

    def open_tracks(self) -> List[PairTrack]:
        return [t for t in self.tracks.values() if t.active]

    def get(self, key: Tuple[int, int, int, int]) -> Optional[PairTrack]:
        t = self.tracks.get(key)
        return t if t is not None and t.active else None
