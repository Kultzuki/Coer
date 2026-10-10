"""Registry of all currently active people. Stable identities for downstream reasoning.

The detector reports per-frame sightings; the registry owns identity over
time. Scenario modules (fight/fall/bag, when implemented) query the registry
instead of raw array positions, so event history is always keyed by
(track_id, generation) and a new track can never silently inherit another
person's history.

State machine per track_id: ACTIVE -> LOST -> (REAPPEARED -> ACTIVE | REMOVED).
Every transition is explicit and drainable via ``drain_transitions()``.

No torch / cv2 / ultralytics dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from core.logging_utils import get_logger
from core.tracked_person import Presence, TrackedPerson

log = get_logger("registry")


class Transition(Enum):
    NEW = "new"
    UPDATED = "updated"
    LOST = "lost"
    REAPPEARED = "reappeared"
    REMOVED = "removed"


@dataclass
class TrackEvent:
    track_id: int
    transition: Transition
    generation: int
    timestamp: float


@dataclass
class _Record:
    person: TrackedPerson
    missing_frames: int = 0


class TrackRegistry:
    def __init__(self, lost_timeout_sec: float = 1.0, lost_grace_sec: float = 5.0):
        if lost_timeout_sec <= 0 or lost_grace_sec <= 0:
            raise ValueError("lost timeouts must be > 0")
        self.lost_timeout_sec = float(lost_timeout_sec)
        self.lost_grace_sec = float(lost_grace_sec)
        self._tracks: Dict[int, _Record] = {}
        self._pending: List[TrackEvent] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._pending.clear()
        log.info("event=registry_reset")

    def update(self, detections: List[TrackedPerson], timestamp: float) -> List[TrackEvent]:
        ts = float(timestamp)
        seen = set()
        events: List[TrackEvent] = []
        for det in detections or []:
            tid = int(det.track_id)
            seen.add(tid)
            rec = self._tracks.get(tid)
            if rec is None or rec.person.state == Presence.REMOVED:
                person = _fresh(det, ts, generation=(rec.person.generation + 1 if rec else 0))
                self._tracks[tid] = _Record(person=person)
                events.append(TrackEvent(tid, Transition.NEW, person.generation, ts))
            elif rec.person.state == Presence.LOST:
                person = _fresh(det, ts, generation=rec.person.generation + 1)
                person.age_frames = 1
                self._tracks[tid] = _Record(person=person)
                events.append(TrackEvent(tid, Transition.REAPPEARED, person.generation, ts))
                log.info("event=track_reappeared id=%d gen=%d ts=%.3f", tid, person.generation, ts)
            else:
                person = rec.person
                person.bbox = list(det.bbox)
                person.keypoints = det.keypoints
                person.detection_conf = float(det.detection_conf)
                person.timestamp = ts
                person.last_seen = ts
                person.present = True
                person.state = Presence.ACTIVE
                person.age_frames += 1
                rec.missing_frames = 0
                events.append(TrackEvent(tid, Transition.UPDATED, person.generation, ts))

        for tid, rec in self._tracks.items():
            if tid in seen or rec.person.state != Presence.ACTIVE:
                continue
            gap = ts - rec.person.last_seen
            if gap >= self.lost_timeout_sec:
                rec.person.state = Presence.LOST
                rec.person.present = False
                rec.missing_frames = 1
                events.append(TrackEvent(tid, Transition.LOST, rec.person.generation, ts))
                log.info("event=track_lost id=%d gen=%d gap=%.3f", tid, rec.person.generation, gap)

        for tid, rec in self._tracks.items():
            if rec.person.state == Presence.LOST and ts - rec.person.last_seen >= self.lost_grace_sec:
                rec.person.state = Presence.REMOVED
                events.append(TrackEvent(tid, Transition.REMOVED, rec.person.generation, ts))

        self._pending.extend(events)
        return events

    def all_people(self) -> List[TrackedPerson]:
        return [r.person for r in self._tracks.values() if r.person.state != Presence.REMOVED]

    def active_people(self) -> List[TrackedPerson]:
        return [r.person for r in self._tracks.values() if r.person.state == Presence.ACTIVE]

    def recently_lost(self) -> List[TrackedPerson]:
        return [r.person for r in self._tracks.values() if r.person.state == Presence.LOST]

    def get(self, track_id: int) -> Optional[TrackedPerson]:
        rec = self._tracks.get(int(track_id))
        if rec is None or rec.person.state == Presence.REMOVED:
            return None
        return rec.person

    def drain_transitions(self) -> List[TrackEvent]:
        drained, self._pending = self._pending, []
        return drained

    def __len__(self) -> int:
        return len(self.active_people())

    def __contains__(self, track_id: object) -> bool:
        return self.get(int(track_id)) is not None  # type: ignore[arg-type]


def _fresh(det: TrackedPerson, timestamp: float, generation: int) -> TrackedPerson:
    return TrackedPerson(
        track_id=int(det.track_id),
        bbox=list(det.bbox),
        keypoints=det.keypoints,
        detection_conf=float(det.detection_conf),
        timestamp=float(timestamp),
        present=True,
        first_seen=float(timestamp),
        last_seen=float(timestamp),
        age_frames=1,
        generation=int(generation),
        state=Presence.ACTIVE,
    )


class PerTrackStore:
    """Per-identity scenario state keyed by (track_id, generation).

    The helper downstream reasoning must use: a returning ID maps to a new
    key, so old counters/histories are unreachable instead of inherited.
    """

    def __init__(self):
        self._state: Dict[tuple, dict] = {}

    def for_person(self, person: TrackedPerson) -> dict:
        key = (int(person.track_id), int(person.generation))
        return self._state.setdefault(key, {})

    def drop_person(self, person: TrackedPerson) -> None:
        self._state.pop((int(person.track_id), int(person.generation)), None)

    def prune_removed(self, registry: TrackRegistry) -> None:
        live = {(p.track_id, p.generation) for p in registry.all_people()}
        for key in [k for k in self._state if k not in live]:
            del self._state[key]

    def __len__(self) -> int:
        return len(self._state)
