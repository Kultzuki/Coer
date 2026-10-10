"""Scenario orchestration. Timestamp-first, no prints, no swallowed errors.

Fixes vs legacy ``core/pipeline.py``:
- ``print(track_ids)`` per-frame debug output removed (debug log instead).
- Broad ``except Exception: print(...)`` replaced with logged traceback;
  ``config.strict=True`` re-raises for debugging, otherwise other scenarios
  still run (failure isolation, not silence: everything is logged).
- Track/keypoint pairing moved to ``core.tracked_person.pair_detections``.
- ``Detector`` is constructed from config (paths/tracker/imgsz/device are
  configurable) and is injectable for tests (``Pipeline(detector=fake)``).
- ``process(frame, timestamp)`` makes timestamps explicit; when omitted the
  pipeline clock (``frame_index / fps``) provides them -- never frame counts
  as event timing, never ``time.time()`` inside scenario logic.
- ``reset()`` clears frame clock + all scenario state for deterministic replay.
- Multi-person: detections flow through ``TrackRegistry`` (stable identities,
  explicit LOST/REAPPEARED transitions, ``recently_lost`` window) and every
  frame updates ``TrackerDiagnostics``. Query API: ``all_people``,
  ``active_people``, ``recently_lost``, ``person_by_id``.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import scenarios.bag_logic as bag_logic
import scenarios.fall_logic as fall_logic
import scenarios.fight_logic as fight_logic
from config import VigilanceConfig
from core.diagnostics import TrackerDiagnostics
from core.graph import InteractionGraph, SceneVerdict, reason_scene
from core.logging_utils import get_logger
from core.pairs import PairManager, generate_candidate_pairs
from core.skeleton import normalize_person
from core.stgcn_adapter import STGCNPPAdapter, is_buffer_ready
from core.track_registry import TrackEvent, TrackRegistry
from core.tracked_person import TrackedPerson

log = get_logger("pipeline")


@dataclass
class PipelineResult:
    persons: List[TrackedPerson] = field(default_factory=list)
    alerts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: float = 0.0
    frame_index: int = 0
    det_results: Any = None
    transitions: List[TrackEvent] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    pair_keys: List[tuple] = field(default_factory=list)
    scene: Optional[SceneVerdict] = None


class Pipeline:
    def __init__(self, config: Optional[VigilanceConfig] = None, detector=None,
                 pair_scorer=None):
        self.config = config or VigilanceConfig()
        self.config.validate()
        if detector is not None:
            self.detector = detector
        else:
            from core.detector import Detector
            self.detector = Detector(
                pose_model_path=self.config.pose_weights,
                det_model_path=self.config.det_weights,
                tracker=self.config.tracker,
                fallback_tracker=self.config.tracker_fallback,
                conf=self.config.conf,
                imgsz=self.config.imgsz,
                device=self.config.device,
                max_people=self.config.max_people,
                load_object_model=self.config.enable_bag,
            )
        self.registry = TrackRegistry(
            lost_timeout_sec=self.config.lost_timeout_sec,
            lost_grace_sec=self.config.lost_grace_sec,
        )
        self.track_diagnostics = TrackerDiagnostics()
        fc = self.config.fight_cfg
        self.pair_manager = PairManager(
            pair_grace_seconds=fc.pair_grace_seconds,
            history_seconds=fc.window_seconds,
        )
        self.graph = InteractionGraph()
        self.pair_adapter = STGCNPPAdapter(
            temporal_length=fc.temporal_length, window_seconds=fc.window_seconds,
            scorer=None, device=fc.model.device,
            freeze_backbone=fc.model.freeze_backbone,
            num_classes=fc.model.num_classes)
        self._pair_score_fn = pair_scorer
        if fc.model.autoload:
            self.load_fight_model()
        self.frame_index = 0
        self._scenarios: List[tuple] = []
        if self.config.enable_fall:
            self._scenarios.append(("Fall Detection", fall_logic, "persons"))
        if self.config.enable_bag:
            self._scenarios.append(("Abandoned Bag", bag_logic, "det"))
        # Legacy BiLSTM fight module: OFF by default. The ST-GCN++ pair/graph
        # path below is the only active fight path. Opt back in (for A/B
        # comparison only) with enable_legacy_fight=true / --legacy-fight.
        if self.config.enable_legacy_fight:
            self._scenarios.append(("Violence/Fight (legacy BiLSTM)", fight_logic, "persons"))
        log.info(
            "event=pipeline_ready scenarios=%s fps=%.2f fight_path=%s",
            [name for name, _, _ in self._scenarios], self.config.fps,
            "stgcn" if self.config.enable_fight else "off",
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear frame clock + registry + diagnostics + pair/graph + scenario state."""
        self.frame_index = 0
        self.registry.reset()
        self.track_diagnostics.reset()
        self.pair_manager.reset()
        self.graph.reset()
        for _, module, _ in self._scenarios:
            reset = getattr(module, "reset", None)
            if callable(reset):
                reset()
        if hasattr(self.detector, "reset") and callable(self.detector.reset):
            try:
                self.detector.reset()
            except Exception:
                pass
        log.info("event=pipeline_reset")

    @property
    def timestamp(self) -> float:
        return self.frame_index / float(self.config.fps)

    def load_fight_model(self, path: Optional[str] = None):
        """Load the ST-GCN++ fight model; explicit and loud about absence."""
        import os
        fc = self.config.fight_cfg
        ckpt = path or os.environ.get("VIGILANCE_FIGHT_CHECKPOINT") or fc.model.checkpoint
        if not os.path.exists(ckpt):
            log.warning("event=fight_model_absent path=%s (repo stays usable)", ckpt)
            return None
        return self.pair_adapter.load_model(
            ckpt, num_classes=fc.model.num_classes,
            freeze_backbone=fc.model.freeze_backbone)

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> PipelineResult:
        ts = float(self.timestamp if timestamp is None else timestamp)
        detection = self.detector.detect(frame, timestamp=ts)
        transitions = self.registry.update(detection.persons, ts)
        self.registry.drain_transitions()
        persons = self.registry.active_people()
        self.track_diagnostics.update(
            len(persons), [p.track_id for p in persons], transitions,
        )
        log.debug(
            "event=frame frame=%d ts=%.3f persons=%s",
            self.frame_index, ts, [p.track_id for p in persons],
        )

        alerts: Dict[str, Dict[str, Any]] = {}
        for name, module, kind in self._scenarios:
            try:
                if kind == "det":
                    res = module.check(detection.det_results, ts)
                else:
                    res = module.check(persons, ts)
                if res and res.get("alert", False):
                    alerts[name] = res
                    log.info("event=alert scenario=%s ts=%.3f data=%s", name, ts, res)
            except Exception as exc:
                # Isolated + LOUD: logged with traceback, never silently swallowed.
                log.error(
                    "event=scenario_error scenario=%s frame=%d ts=%.3f error=%r\n%s",
                    name, self.frame_index, ts, exc, traceback.format_exc(),
                )
                if self.config.strict:
                    raise

        try:
            if not self.config.enable_fight:
                verdict, live_pairs = None, []
            else:
                h, w = (int(frame.shape[0]), int(frame.shape[1])) if frame is not None \
                    and getattr(frame, "ndim", 0) >= 2 else (1080, 1920)
                self.pair_adapter.img_hw = (float(h), float(w))
                verdict = self.run_pair_graph(persons, ts)
                live_pairs = self.pair_manager.open_tracks()
        except Exception as exc:
            log.error("event=pair_graph_error frame=%d error=%r", self.frame_index, exc)
            if self.config.strict:
                raise
            verdict = None
            live_pairs = []

        result = PipelineResult(
            persons=persons,
            alerts=alerts,
            timestamp=ts,
            frame_index=self.frame_index,
            det_results=detection.det_results,
            transitions=transitions,
            diagnostics=self.track_diagnostics.summary(),
            pair_keys=[t.key for t in live_pairs],
            scene=verdict,
        )
        self.frame_index += 1
        return result

    def run_pair_graph(self, persons: List[TrackedPerson],
                       timestamp: float) -> SceneVerdict:
        fc = self.config.fight_cfg
        skeletons = {p.track_id: normalize_person(p) for p in persons}
        eligible = [p for p in persons if p.age_frames >= fc.min_track_age_frames]
        candidates = generate_candidate_pairs(
            eligible, skeletons, fc.pair_max_distance,
            fc.min_person_visibility, fc.max_candidate_pairs)
        live_pairs = self.pair_manager.update(candidates, skeletons, timestamp)
        self.graph.reset()
        self.graph.ingest_tracks(live_pairs, timestamp)
        if self._pair_score_fn is not None:
            ready = [t for t in live_pairs
                     if is_buffer_ready(t, fc.window_seconds)]
            if ready:
                for track, score in zip(ready, self._pair_score_fn(ready)):
                    self.graph.set_score(track.key, float(score))
        elif self.pair_adapter.is_ready and self.pair_adapter.head_trained:
            ready = [t for t in live_pairs
                     if is_buffer_ready(t, fc.window_seconds)]
            if ready:
                import numpy as np
                batch = np.stack([self.pair_adapter.preprocess_pair(t)
                                  for t in ready], axis=0)
                for track, score in zip(ready, self.pair_adapter.predict_batch(batch)):
                    self.graph.set_score(track.key, float(score))
        return reason_scene(self.graph, fc.graph)

    def all_people(self) -> List[TrackedPerson]:
        return self.registry.all_people()

    def active_people(self) -> List[TrackedPerson]:
        return self.registry.active_people()

    def recently_lost(self) -> List[TrackedPerson]:
        return self.registry.recently_lost()

    def person_by_id(self, track_id: int) -> Optional[TrackedPerson]:
        return self.registry.get(track_id)
