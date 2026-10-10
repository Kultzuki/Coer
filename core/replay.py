"""Deterministic replay: same video in -> comparable outputs out.

Rules enforced here:
- ``pipeline.reset()`` before every run (no leaked buffers/counters).
- Timestamps come from ``frame_index / fps`` (video FPS when readable,
  else ``config.fps``), never from wall-clock.
- ``random``/``numpy`` are re-seeded from ``config`` seed so stochastic
  callers behave identically across runs.
- Tracked persons are serialized in ``track_id`` order; alerts in sorted
  key order; floats rounded -- so two runs diff cleanly.
- Evidence is a JSONL file (one record per frame) under ``evidence_dir``.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from config import VigilanceConfig
from core.logging_utils import get_logger

log = get_logger("replay")


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    try:
        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass


def frame_record(frame_index: int, timestamp: float, persons, alerts) -> Dict[str, Any]:
    persons_sorted = sorted(list(persons), key=lambda p: p.track_id)
    return {
        "frame": int(frame_index),
        "timestamp": round(float(timestamp), 6),
        "persons": [p.to_dict() for p in persons_sorted],
        "alerts": {k: alerts[k] for k in sorted(alerts)},
    }


def compare_runs(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare two replay record lists; report first divergence (if any)."""
    result: Dict[str, Any] = {
        "frames_a": len(a), "frames_b": len(b),
        "identical": False, "first_diff_frame": None, "detail": "",
    }
    if len(a) != len(b):
        result["detail"] = f"frame count differs: {len(a)} vs {len(b)}"
        return result
    for ra, rb in zip(a, b):
        if ra["timestamp"] != rb["timestamp"]:
            result.update(first_diff_frame=ra["frame"], detail="timestamp differs")
            return result
        pa, pb = ra["persons"], rb["persons"]
        if len(pa) != len(pb):
            result.update(first_diff_frame=ra["frame"], detail="person count differs")
            return result
        for xa, xb in zip(pa, pb):
            if xa["track_id"] != xb["track_id"] or xa["bbox"] != xb["bbox"]:
                result.update(first_diff_frame=ra["frame"], detail="person differs")
                return result
        if set(ra["alerts"]) != set(rb["alerts"]):
            result.update(first_diff_frame=ra["frame"], detail="alert set differs")
            return result
    result["identical"] = True
    result["detail"] = "all frames comparable"
    return result


@dataclass
class ReplayRunner:
    config: VigilanceConfig = field(default_factory=VigilanceConfig)
    seed: int = 0

    def run_frames(self, frames: Iterable[np.ndarray], pipeline=None) -> List[Dict[str, Any]]:
        from core.pipeline import Pipeline
        pipe = pipeline or Pipeline(config=self.config)
        pipe.reset()
        seed_everything(self.seed)
        records: List[Dict[str, Any]] = []
        for idx, frame in enumerate(frames):
            ts = idx / float(self.config.fps)
            result = pipe.process(np.asarray(frame), timestamp=ts)
            records.append(frame_record(result.frame_index, result.timestamp, result.persons, result.alerts))
        log.info("event=replay_done frames=%d fps=%.2f", len(records), self.config.fps)
        return records

    def run_frames_and_save(
        self, frames: Iterable[np.ndarray], run_name: str = "replay", pipeline=None
    ) -> str:
        records = self.run_frames(frames, pipeline=pipeline)
        os.makedirs(self.config.evidence_dir, exist_ok=True)
        path = os.path.join(self.config.evidence_dir, f"{run_name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=float) + "\n")
        log.info("event=evidence_written path=%s frames=%d", path, len(records))
        return path

    def run_video(self, source: str, pipeline=None, max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required to read video files.") from exc
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.config.fps = float(fps)
        from core.pipeline import Pipeline
        pipe = pipeline or Pipeline(config=self.config)
        pipe.reset()
        seed_everything(self.seed)
        records: List[Dict[str, Any]] = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = idx / float(self.config.fps)
                result = pipe.process(frame, timestamp=ts)
                records.append(frame_record(result.frame_index, result.timestamp, result.persons, result.alerts))
                idx += 1
                if max_frames is not None and idx >= max_frames:
                    break
        finally:
            cap.release()
        log.info("event=replay_video_done source=%s frames=%d", source, len(records))
        return records
