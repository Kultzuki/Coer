"""Canonical tracked-person object. The single currency of the pipeline.

Every detector output and every scenario input uses :class:`TrackedPerson`.
Timestamps are first-class (seconds, float); frame counts are never the
source of truth for event timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence

import numpy as np

NUM_KPTS = 17
FLAT_DIM = NUM_KPTS * 3  # 51: (x, y, conf) * 17 COCO keypoints

DEFAULT_MAX_PEOPLE = 32


class Presence(Enum):
    """Explicit presence/validity of a track (req. 7, 10)."""
    ACTIVE = "active"  # seen in the latest update
    LOST = "lost"      # not seen recently, kept in the recently-lost window
    REMOVED = "removed"  # expired; no longer queryable (terminal state)


@dataclass
class TrackedPerson:
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    keypoints: np.ndarray  # shape (17, 3), float32, rows = (x, y, conf)
    detection_conf: float = 0.0
    timestamp: float = 0.0  # seconds; authoritative event clock (last update)
    present: bool = True
    # --- lifecycle (maintained by TrackRegistry; defaults keep direct
    # --- construction backward compatible) ---
    first_seen: float = 0.0   # timestamp of first sighting (track birth)
    last_seen: float = 0.0    # timestamp of most recent detection
    age_frames: int = 1       # number of frames this incarnation was observed
    generation: int = 0       # incarnation counter: bumped on every
                              # reappearance after LOST so a returning ID never
                              # silently inherits its old event history
    state: Presence = Presence.ACTIVE

    def __post_init__(self) -> None:
        arr = np.asarray(self.keypoints, dtype=np.float32)
        if arr.shape == (FLAT_DIM,):
            arr = arr.reshape(NUM_KPTS, 3)
        if arr.shape != (NUM_KPTS, 3):
            raise ValueError(f"keypoints must be (17,3) or (51,), got {arr.shape}")
        object.__setattr__(self, "keypoints", arr)
        self.bbox = [float(v) for v in self.bbox]
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {self.bbox}")

    # ---- derived views -------------------------------------------------
    @property
    def keypoint_conf(self) -> np.ndarray:
        return self.keypoints[:, 2]

    @property
    def keypoints_flat(self) -> np.ndarray:
        return self.keypoints.reshape(-1).astype(np.float32)

    @property
    def visibility_ratio(self) -> float:
        """Fraction of keypoints with conf > 0.3."""
        return float(np.mean(self.keypoint_conf > 0.3))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "bbox": list(self.bbox),
            "keypoints": self.keypoints_flat.tolist(),
            "detection_conf": float(self.detection_conf),
            "timestamp": float(self.timestamp),
            "present": bool(self.present),
            "first_seen": float(self.first_seen),
            "last_seen": float(self.last_seen),
            "age_frames": int(self.age_frames),
            "generation": int(self.generation),
            "state": self.state.value,
        }

    @property
    def track_age_sec(self) -> float:
        """Wall-independent track age: last_seen - first_seen (seconds)."""
        return max(0.0, float(self.last_seen) - float(self.first_seen))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedPerson":
        state = data.get("state", Presence.ACTIVE.value)
        return cls(
            track_id=int(data["track_id"]),
            bbox=list(data["bbox"]),
            keypoints=np.asarray(data["keypoints"], dtype=np.float32),
            detection_conf=float(data.get("detection_conf", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
            present=bool(data.get("present", True)),
            first_seen=float(data.get("first_seen", data.get("timestamp", 0.0))),
            last_seen=float(data.get("last_seen", data.get("timestamp", 0.0))),
            age_frames=int(data.get("age_frames", 1)),
            generation=int(data.get("generation", 0)),
            state=Presence(state) if isinstance(state, str) else state,
        )

    # ---- legacy compat (old dicts used key "keypoints" flat + no timestamp)
    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "bbox": list(self.bbox),
            "keypoints": self.keypoints_flat,
        }

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any], timestamp: float = 0.0) -> "TrackedPerson":
        return cls(
            track_id=int(data["track_id"]),
            bbox=list(data["bbox"]),
            keypoints=np.asarray(data["keypoints"], dtype=np.float32),
            detection_conf=float(data.get("detection_conf", 0.0)),
            timestamp=float(timestamp),
            present=True,
            first_seen=float(timestamp),
            last_seen=float(timestamp),
        )


def pair_detections(
    boxes: Sequence[Sequence[float]],
    keypoints: Sequence[np.ndarray],
    track_ids: Sequence[int],
    det_confs: Sequence[float] | None = None,
    timestamp: float = 0.0,
    max_people: int = DEFAULT_MAX_PEOPLE,
) -> List[TrackedPerson]:
    """Index-aligned pairing fix.

    The legacy pipeline built ``bboxes`` and ``kp_list`` in two separate loops
    truncated by different lengths, so a disappearing track ID could shift one
    list relative to the other. Here a SINGLE loop over
    ``n = min(len(boxes), len(keypoints), len(track_ids))`` pairs
    ``(box[i], kpts[i], id[i])`` atomically, then truncates to ``max_people``.
    Detection order is preserved (callers needing stability sort by track_id).
    """
    n = min(len(boxes), len(keypoints), len(track_ids))
    n = min(n, max(0, int(max_people)))
    persons: List[TrackedPerson] = []
    for i in range(n):
        persons.append(TrackedPerson(
            track_id=int(track_ids[i]),
            bbox=list(boxes[i]),
            keypoints=np.asarray(keypoints[i], dtype=np.float32),
            detection_conf=float(det_confs[i]) if det_confs is not None and i < len(det_confs) else 0.0,
            timestamp=float(timestamp),
            present=True,
        ))
    return persons


def sort_by_track_id(persons: List[TrackedPerson]) -> List[TrackedPerson]:
    return sorted(persons, key=lambda p: p.track_id)


def legacy_dicts_to_persons(
    items: Sequence[Dict[str, Any]], timestamp: float = 0.0
) -> List[TrackedPerson]:
    """Adapt old ``{"track_id","bbox","keypoints"}`` dicts (e.g. in tests)."""
    return [TrackedPerson.from_legacy_dict(d, timestamp=timestamp) for d in items]
