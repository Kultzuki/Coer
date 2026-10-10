"""Multi-person perception tests. No torch / cv2 / ultralytics required.

Verifies the phase requirements on synthetic replay streams:
- 0 / 1 / 2 / 3-4 / 5-8 / crowded scenes are all retained (never truncated)
- 3+ people coexist with stable identities across frames
- scenario-style state keyed by (track_id, generation) follows identity,
  not array position, and never leaks across disappear/return
- disappear -> LOST and return -> REAPPEARED are explicit with generation bump
- registry query API: all / active / recently_lost / by ID
- diagnostics: avg/max people, dropout, ID churn
- tracker fallback: ByteTrack engages iff BoT-SORT raises (pure-logic test)
- crowded replay is deterministic
"""

import pytest

from config import VigilanceConfig
from core.diagnostics import TrackerDiagnostics
from core.pipeline import Pipeline
from core.replay import ReplayRunner, compare_runs
from core.track_registry import PerTrackStore, TrackRegistry, Transition
from core.tracked_person import Presence
from tests.conftest import FakeDetector, black_frames


def _crowd(n, start_id=1, x_step=90.0):
    return [(start_id + i, 40.0 + i * x_step) for i in range(n)]


def _pipe(script, **cfg_kwargs):
    cfg = VigilanceConfig(enable_bag=False, **cfg_kwargs)
    cfg.validate()
    return Pipeline(config=cfg, detector=FakeDetector(script))


# density ladder ------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 6, 8, 12])
def test_density_ladder_all_retained(n):
    pipe = _pipe([_crowd(n)])
    res = pipe.process(black_frames(1)[0])
    assert len(res.persons) == n
    assert sorted(p.track_id for p in res.persons) == list(range(1, n + 1))


def test_three_plus_simultaneous_with_stable_identities():
    script = [_crowd(5), _crowd(5), _crowd(5)]
    pipe = _pipe(script)
    seen_gens = {}
    for frame in black_frames(3):
        res = pipe.process(frame)
        assert len(res.persons) == 5
        for p in res.persons:
            assert p.state == Presence.ACTIVE
            assert p.last_seen == pytest.approx(res.timestamp)
            assert p.track_age_sec >= 0.0
            seen_gens.setdefault(p.track_id, p.generation)
            assert p.generation == seen_gens[p.track_id] == 0
    assert pipe.track_diagnostics.summary()["max_people_per_frame"] == 5


# identity-keyed state, not array position ------------------------------------
def test_per_track_state_follows_identity_not_position():
    store = PerTrackStore()
    reg = TrackRegistry()
    reg.update(_persons([(1, 0.0), (2, 100.0), (3, 200.0)]), 0.0)
    for p in reg.active_people():
        store.for_person(p)["count"] = p.track_id * 10
    # same identities, reversed detection order
    reg.update(_persons([(3, 200.0), (2, 100.0), (1, 0.0)]), 0.1)
    by_id = {p.track_id: store.for_person(p)["count"] for p in reg.active_people()}
    assert by_id == {1: 10, 2: 20, 3: 30}


def test_returning_id_does_not_inherit_history():
    store = PerTrackStore()
    reg = TrackRegistry(lost_timeout_sec=0.2, lost_grace_sec=5.0)
    reg.update(_persons([(1, 0.0)]), 0.0)
    first = reg.get(1)
    store.for_person(first)["events"] = ["punch", "kick"]
    reg.update([], 0.5)  # gap 0.5s > timeout -> LOST
    assert reg.get(1).state == Presence.LOST
    reg.update(_persons([(1, 0.0)]), 0.6)  # same numeric ID returns
    second = reg.get(1)
    assert second.generation == first.generation + 1
    assert store.for_person(second) == {}  # fresh key, no inherited events
    assert store.for_person(first) == {"events": ["punch", "kick"]}  # old key intact


# explicit transitions ---------------------------------------------------------
def test_disappear_return_transitions_explicit():
    reg = TrackRegistry(lost_timeout_sec=0.3, lost_grace_sec=5.0)
    ev = reg.update(_persons([(1, 0.0), (2, 100.0)]), 0.0)
    assert {e.transition for e in ev} == {Transition.NEW}
    ev = reg.update([], 0.5)
    assert {e.transition for e in ev} == {Transition.LOST}
    assert {p.track_id for p in reg.recently_lost()} == {1, 2}
    assert reg.active_people() == []
    ev = reg.update(_persons([(2, 100.0)]), 0.6)
    kinds = {e.transition for e in ev}
    assert Transition.REAPPEARED in kinds
    assert reg.get(2).state == Presence.ACTIVE
    assert reg.get(1).state == Presence.LOST  # still explicitly lost, not dropped


def test_removed_id_returns_as_new_generation():
    reg = TrackRegistry(lost_timeout_sec=0.2, lost_grace_sec=0.5)
    reg.update(_persons([(1, 0.0)]), 0.0)
    reg.update([], 0.5)  # LOST (gap 0.5 >= 0.2)
    reg.update([], 1.2)  # REMOVED (gap 1.2 >= 0.5)
    assert reg.get(1) is None
    ev = reg.update(_persons([(1, 0.0)]), 1.3)
    assert ev[0].transition == Transition.NEW
    assert reg.get(1).generation == 1  # generation keeps counting, no reuse


# registry query API -------------------------------------------------------------
def test_registry_query_api():
    pipe = _pipe([_crowd(3), [(1, 40.0)]])
    pipe.process(black_frames(1)[0])
    assert len(pipe.all_people()) == 3
    assert len(pipe.active_people()) == 3
    assert pipe.person_by_id(2).track_id == 2
    assert 99 not in pipe.registry
    assert 2 in pipe.registry


# diagnostics ----------------------------------------------------------------------
def test_diagnostics_avg_max_dropout_churn():
    script = [[], _crowd(2), _crowd(6), [], _crowd(4)]
    pipe = _pipe(script, lost_timeout_sec=0.01, lost_grace_sec=1.0)
    for frame in black_frames(len(script)):
        pipe.process(frame)
    d = pipe.track_diagnostics.summary()
    assert d["frames"] == 5
    assert d["avg_people_per_frame"] == pytest.approx((0 + 2 + 6 + 0 + 4) / 5)
    assert d["max_people_per_frame"] == 6
    assert d["empty_frames"] == 2
    assert d["detection_dropout_rate"] == pytest.approx(0.4)
    assert d["unique_ids"] == 6
    assert d["id_churn"]["new"] == 6


def test_diagnostics_unit_churn_counts():
    diag = TrackerDiagnostics()
    assert diag.summary()["frames"] == 0
    assert diag.summary()["avg_people_per_frame"] == 0.0


# tracker fallback -------------------------------------------------------------------
def test_tracker_fallback_engages_once_on_primary_failure():
    from core.detector import Detector
    det = Detector.__new__(Detector)
    det.tracker = "botsort.yaml"
    det.fallback_tracker = "bytetrack.yaml"
    det.tracker_in_use = det.tracker
    det._fallback_engaged = False
    calls = []

    def flaky(name):
        calls.append(name)
        if name == "botsort.yaml":
            raise RuntimeError("botsort exploded")
        return f"ok-{name}"

    assert det._track_with_fallback(flaky) == f"ok-{det.fallback_tracker}"
    assert det.tracker_in_use == det.fallback_tracker
    assert det._track_with_fallback(flaky) == f"ok-{det.fallback_tracker}"  # sticks to fallback
    assert calls == [det.tracker, det.fallback_tracker, det.fallback_tracker]


def test_tracker_defaults_botsort_primary_bytetrack_fallback():
    cfg = VigilanceConfig()
    assert cfg.tracker == "botsort.yaml"
    assert cfg.tracker_fallback == "bytetrack.yaml"
    assert cfg.max_people >= 8  # safety cap, never a 2-person truncation


# crowded replay determinism ------------------------------------------------------------
def test_crowded_replay_deterministic():
    script = [_crowd(8), _crowd(10), _crowd(6), _crowd(10)]
    cfg = VigilanceConfig(enable_bag=False, fps=10.0,
                           lost_timeout_sec=0.05, lost_grace_sec=5.0)
    runner = ReplayRunner(config=cfg, seed=0)
    mk = lambda: Pipeline(config=cfg, detector=FakeDetector(script))  # noqa: E731
    run_a = runner.run_frames(black_frames(4), pipeline=mk())
    run_b = runner.run_frames(black_frames(4), pipeline=mk())
    cmp = compare_runs(run_a, run_b)
    assert cmp["identical"], cmp
    assert [len(r["persons"]) for r in run_a] == [8, 10, 6, 10]


# helpers ------------------------------------------------------------------------------
def _persons(pairs, timestamp=0.0):
    from tests.conftest import make_person
    return [make_person(tid, x0=x0, timestamp=timestamp) for tid, x0 in pairs]
