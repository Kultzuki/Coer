"""YOLO pose + object detector with tracking.

Foundational fixes vs the legacy ``core/detector.py``:
1. Broken default fixed: ``pose_weights`` defaults to ``yolo26m-pose.pt``
   (the file the project actually ships), not ``yolo26x-pose.pt``.
2. All paths / tracker / imgsz / conf / device / max_people are
   configurable (see ``config.VigilanceConfig``); ``resolve_model_path``
   searches explicit path -> env vars -> repo root -> models/ -> cwd.
3. The stale "4 persons" comment vs ``(2, 51)`` code contradiction is gone:
   the person cap is the explicit ``max_people`` parameter.
4. Track/keypoint pairing is index-atomic via
   ``tracked_person.pair_detections`` (single loop, no split lists).
5. ``ultralytics`` is imported lazily inside the constructor so importing
   this module (tests, config inspection, Kaggle CPU checks) never
   requires torch/ultralytics. Construction raises an informative error.
6. Multi-person: the scene is NEVER truncated to 2 people. ``max_people``
   is a high safety cap only (default 32, configurable); the registry
   downstream keeps every reported identity.
7. Tracker backends: BoT-SORT is primary, ByteTrack is the configured
   fallback. If the primary tracker raises at runtime, the detector fails
   over to the fallback once per process and keeps serving (logged loudly).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

from core.logging_utils import get_logger
from core.tracked_person import DEFAULT_MAX_PEOPLE, TrackedPerson, pair_detections

log = get_logger("detector")

REPO_ROOT_CANDIDATES = ("VIGILANCE_POSE_WEIGHTS", "VIGILANCE_DET_WEIGHTS")

PRIMARY_TRACKER = "botsort.yaml"
FALLBACK_TRACKER = "bytetrack.yaml"
SUPPORTED_TRACKERS = (PRIMARY_TRACKER, FALLBACK_TRACKER)


def check_tracker_supported(name: str) -> str:
    if name not in SUPPORTED_TRACKERS and not name.lower().endswith((".yaml", ".yml")):
        log.warning("event=tracker_unsupported name=%s supported=%s", name, SUPPORTED_TRACKERS)
    return name


def resolve_model_path(explicit: str, env_var: str) -> str:
    """Resolve a weights path; raise FileNotFoundError naming every place tried."""
    tried = []
    candidates = [explicit]
    env_val = os.environ.get(env_var)
    if env_val:
        candidates.append(env_val)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.basename(explicit)
    candidates += [
        os.path.join(here, base),
        os.path.join(here, explicit),
        os.path.join(here, "models", base),
        os.path.join(os.getcwd(), base),
        os.path.join(os.getcwd(), explicit),
    ]
    for cand in candidates:
        tried.append(cand)
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Model weights '{explicit}' not found. Tried: {tried}. "
        f"Set {env_var} or pass an explicit path (see config.py)."
    )


@dataclass
class DetectorOutput:
    persons: List[TrackedPerson] = field(default_factory=list)
    pose_results: Any = None
    det_results: Any = None


class Detector:
    def __init__(
        self,
        pose_model_path: str = "yolo26m-pose.pt",
        det_model_path: str = "yolo26m.pt",
        tracker: str = PRIMARY_TRACKER,
        fallback_tracker: str = FALLBACK_TRACKER,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "auto",
        max_people: int = DEFAULT_MAX_PEOPLE,
        load_object_model: bool = True,
    ):
        try:
            from ultralytics import YOLO  # lazy: import-time must stay light
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for Detector but is not installed. "
                "Install the ML requirements (pip install -r requirements.txt)."
            ) from exc

        self.tracker = check_tracker_supported(tracker)
        self.fallback_tracker = check_tracker_supported(fallback_tracker)
        self.tracker_in_use = self.tracker
        self._fallback_engaged = False
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = None if device == "auto" else device
        self.max_people = int(max_people)

        pose_path = resolve_model_path(pose_model_path, "VIGILANCE_POSE_WEIGHTS")
        det_path = (resolve_model_path(det_model_path, "VIGILANCE_DET_WEIGHTS")
                    if load_object_model else None)
        log.info("event=detector_loading pose=%s det=%s tracker=%s fallback=%s",
                 pose_path, det_path, self.tracker, self.fallback_tracker)
        self.pose_model = YOLO(pose_path)
        self.det_model = YOLO(det_path) if det_path is not None else None
        self.pose_weights_path = pose_path
        self.det_weights_path = det_path
        self._last_frame_shape: tuple | None = None
        log.info("event=detector_ready max_people=%d", self.max_people)

    def reset(self) -> None:
        """Clear ultralytics tracker state (track IDs + GMC prev frame).

        Required because we call with persist=True: without this, switching
        video sources leaves a stale previous frame inside BoT-SORT's GMC,
        whose optical-flow pyramids then mismatch the new resolution and
        spam 'GMC failed, falling back to identity' (killing camera-motion
        compensation), while old track IDs leak into the new video.
        Pipeline.reset() already calls this via hasattr.
        """
        self.tracker_in_use = self.tracker
        self._fallback_engaged = False
        self._last_frame_shape = None
        for model in (getattr(self, "pose_model", None),
                      getattr(self, "det_model", None)):
            trackers = getattr(getattr(model, "predictor", None), "trackers", None)
            if not trackers:
                continue
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        pass
        log.info("event=detector_reset")

    def _track_with_fallback(self, run_fn: Callable[[str], Any]) -> Any:
        try:
            return run_fn(self.tracker_in_use)
        except Exception as exc:
            if self._fallback_engaged or self.tracker_in_use == self.fallback_tracker:
                raise
            self._fallback_engaged = True
            self.tracker_in_use = self.fallback_tracker
            log.warning(
                "event=tracker_fallback primary=%s fallback=%s error=%r",
                self.tracker, self.fallback_tracker, exc,
            )
            return run_fn(self.tracker_in_use)

    # ---- overridable seams (used by tests via FakeDetector) ----------------
    def _run_pose(self, frame: np.ndarray):
        def _run(tracker_name: str):
            kwargs = dict(conf=self.conf, tracker=tracker_name, persist=True, verbose=False)
            if self.device:
                kwargs["device"] = self.device
            if self.imgsz:
                kwargs["imgsz"] = self.imgsz
            return self.pose_model.track(frame, **kwargs)

        return self._track_with_fallback(_run)

    def _run_det(self, frame: np.ndarray):
        if self.det_model is None:
            return []
        kwargs = dict(conf=self.conf, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        return self.det_model(frame, **kwargs)

    # ---- main entry ---------------------------------------------------------
    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectorOutput:
        shape = tuple(int(v) for v in frame.shape[:2]) if frame is not None \
            and getattr(frame, "ndim", 0) >= 2 else None
        if (shape is not None and self._last_frame_shape is not None
                and shape != self._last_frame_shape):
            # Mid-stream resolution change (or a source switch that bypassed
            # reset): stale GMC/track state would corrupt IDs and flood
            # 'GMC failed' warnings. Reset first.
            log.warning("event=detector_resolution_change old=%s new=%s resetting_tracker",
                        self._last_frame_shape, shape)
            self.reset()
        self._last_frame_shape = shape
        pose_results = self._run_pose(frame)
        det_results = self._run_det(frame)

        boxes: List[List[float]] = []
        kpts: List[np.ndarray] = []
        ids: List[int] = []
        confs: List[float] = []

        if len(pose_results) > 0:
            result = pose_results[0]
            if result.keypoints is not None and result.keypoints.data is not None:
                raw = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                for row in raw:
                    kpts.append(np.asarray(row, dtype=np.float32).reshape(17, 3))
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy().tolist()
                try:
                    confs = result.boxes.conf.cpu().numpy().tolist()
                except Exception:
                    confs = [0.0] * len(boxes)
                if getattr(result.boxes, "id", None) is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                else:
                    ids = [-(index + 1) for index in range(len(boxes))]
                    log.warning(
                        "event=tracker_ids_missing count=%d using_frame_fallback_ids",
                        len(ids),
                    )
            else:
                ids = [-(index + 1) for index in range(len(kpts))]

        persons = pair_detections(
            boxes, kpts, ids, det_confs=confs or None,
            timestamp=timestamp, max_people=self.max_people,
        )
        log.debug("event=detect persons=%d ts=%.3f", len(persons), timestamp)
        return DetectorOutput(persons=persons, pose_results=pose_results, det_results=det_results)
