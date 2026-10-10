"""Tracker diagnostics: churn, dropout, density. No heavy dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.logging_utils import get_logger
from core.track_registry import TrackEvent, Transition

log = get_logger("diagnostics")


@dataclass
class TrackerDiagnostics:
    frames: int = 0
    people_total: int = 0
    max_people: int = 0
    empty_frames: int = 0
    new_ids: int = 0
    lost_ids: int = 0
    reappeared_ids: int = 0
    removed_ids: int = 0
    unique_ids: set = field(default_factory=set)

    def reset(self) -> None:
        self.frames = 0
        self.people_total = 0
        self.max_people = 0
        self.empty_frames = 0
        self.new_ids = 0
        self.lost_ids = 0
        self.reappeared_ids = 0
        self.removed_ids = 0
        self.unique_ids.clear()

    def update(self, num_active: int, active_ids: List[int], events: List[TrackEvent]) -> None:
        self.frames += 1
        self.people_total += int(num_active)
        self.max_people = max(self.max_people, int(num_active))
        if num_active == 0:
            self.empty_frames += 1
        for tid in active_ids:
            self.unique_ids.add(int(tid))
        for ev in events:
            if ev.transition == Transition.NEW:
                self.new_ids += 1
            elif ev.transition == Transition.LOST:
                self.lost_ids += 1
            elif ev.transition == Transition.REAPPEARED:
                self.reappeared_ids += 1
            elif ev.transition == Transition.REMOVED:
                self.removed_ids += 1

    def summary(self) -> Dict[str, Any]:
        avg = (self.people_total / self.frames) if self.frames else 0.0
        return {
            "frames": self.frames,
            "avg_people_per_frame": round(avg, 3),
            "max_people_per_frame": self.max_people,
            "empty_frames": self.empty_frames,
            "detection_dropout_rate": round(self.empty_frames / self.frames, 4) if self.frames else 0.0,
            "unique_ids": len(self.unique_ids),
            "id_churn": {
                "new": self.new_ids,
                "lost": self.lost_ids,
                "reappeared": self.reappeared_ids,
                "removed": self.removed_ids,
            },
        }
