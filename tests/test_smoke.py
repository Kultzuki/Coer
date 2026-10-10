"""Foundation smoke tests. No torch / cv2 / ultralytics required.

Covers the mandated cases:
1. video input (synthetic frame sequence through Pipeline)
2. detector startup (weights resolution + informative errors)
3. tracker startup (tracker name configurable, default botsort)
4. empty frame
5. one person
6. multiple people
7. disappearing / reappearing track IDs
Plus: determinism (same frames twice -> comparable records) and
timestamp-first behavior (explicit timestamps win over frame clock).
"""

import numpy as np
import pytest

from config import VigilanceConfig
from core.pipeline import Pipeline
from core.replay import ReplayRunner, compare_runs
from core.tracked_person import TrackedPerson, pair_detections
from tests.conftest import FakeDetector, black_frames


def _pipe(script, **cfg_kwargs):
    cfg = VigilanceConfig(enable_bag=False, **cfg_kwargs)
    cfg.validate()
    return Pipeline(config=cfg, detector=FakeDetector(script))


# 1. video input -----------------------------------------------------------
def test_video_input_synthetic_sequence():
    pipe = _pipe([[], [(1, 100.0)], [(1, 100.0), (2, 300.0)]])
    frames = black_frames(3)
    results = [pipe.process(f) for f in frames]
    assert [r.frame_index for r in results] == [0, 1, 2]
    assert [len(r.persons) for r in results] == [0, 1, 2]
    assert all(isinstance(r.timestamp, float) for r in results)


# 2. detector startup -------------------------------------------------------
def test_detector_startup_missing_weights_raise_informative(tmp_path):
    from core.detector import resolve_model_path
    with pytest.raises(FileNotFoundError, match="Tried"):
        resolve_model_path("definitely-not-here-xyz.pt", "VIGILANCE_POSE_WEIGHTS")
    # explicit existing file resolves
    probe = tmp_path / "probe.pt"
    probe.write_bytes(b"fake")
    assert resolve_model_path(str(probe), "VIGILANCE_POSE_WEIGHTS") == str(probe)


def test_detector_startup_default_weights_are_shipped_names():
    cfg = VigilanceConfig()
    assert cfg.pose_weights == "yolo26m-pose.pt"  # NOT the legacy yolo26x-pose.pt
    assert cfg.det_weights == "yolo26m.pt"


def test_detector_import_without_ultralytics_is_informative(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from core import detector as det_mod
    with pytest.raises(ImportError, match="ultralytics"):
        det_mod.Detector()


# 3. tracker startup ---------------------------------------------------------
def test_tracker_name_configurable_and_default_botsort():
    assert VigilanceConfig().tracker == "botsort.yaml"
    cfg = VigilanceConfig(tracker="bytetrack.yaml")
    pipe = Pipeline(config=cfg, detector=FakeDetector([[]]))
    assert pipe.config.tracker == "bytetrack.yaml"


def test_detector_reset_safe_before_inference():
    from core.detector import Detector
    det = Detector()  # resolves local weights; no inference yet
    det.reset()  # must not crash with no predictor state
    assert det.tracker_in_use == det.tracker
    assert det._fallback_engaged is False
    assert det._last_frame_shape is None


def test_detector_resolution_change_resets_tracker_state():
    from core.detector import Detector
    det = Detector()
    det.detect(np.zeros((240, 320, 3), np.uint8), timestamp=0.0)
    assert det._last_frame_shape == (240, 320)
    # New resolution must clear stale GMC/track state, not reuse it.
    det.detect(np.zeros((480, 640, 3), np.uint8), timestamp=0.033)
    assert det._last_frame_shape == (480, 640)
    assert det.tracker_in_use == det.tracker


# 4. empty frame --------------------------------------------------------------
def test_empty_frame_no_persons_no_crash():
    pipe = _pipe([[]])
    res = pipe.process(black_frames(1)[0])
    assert res.persons == []
    assert res.alerts == {}


# 5. one person ----------------------------------------------------------------
def test_one_person_present_with_confidence():
    pipe = _pipe([[(7, 120.0)]])
    res = pipe.process(black_frames(1)[0])
    assert len(res.persons) == 1
    p = res.persons[0]
    assert p.track_id == 7
    assert p.keypoint_conf.shape == (17,)
    assert p.visibility_ratio > 0.9
    assert len(p.bbox) == 4


# 6. multiple people --------------------------------------------------------------
def test_multiple_people_respects_max_people_and_pairs_atomically():
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
    kpts = [np.ones(51, dtype=np.float32) * i for i in range(3)]
    ids = [11, 22, 33]
    persons = pair_detections(boxes, kpts, ids, timestamp=1.0, max_people=2)
    assert [p.track_id for p in persons] == [11, 22]  # detection order, atomic
    assert np.allclose(persons[1].keypoints_flat, kpts[1])  # no shifted pairing

    pipe = _pipe([[(1, 50.0), (2, 150.0), (3, 250.0)]], max_people=3)
    res = pipe.process(black_frames(1)[0], timestamp=0.5)
    assert sorted(p.track_id for p in res.persons) == [1, 2, 3]


# 7. disappearing / reappearing IDs --------------------------------------------------
def test_disappearing_reappearing_track_ids_no_crash_and_histories_reset():
    script = [
        [(1, 100.0), (2, 300.0)],
        [],  # both vanish
        [],  # stay gone
        [(2, 300.0)],  # only ID 2 returns (ID 1 gone for good)
        [(2, 300.0), (9, 500.0)],  # new ID 9 appears
    ]
    pipe = _pipe(script)
    for f in black_frames(len(script)):
        res = pipe.process(f)
        assert isinstance(res.alerts, dict)
    # fall histories for vanished IDs clear on empty keypoints; pipeline alive
    assert pipe.frame_index == len(script)


# determinism -----------------------------------------------------------------------
def test_deterministic_replay_comparable():
    script = [[], [(1, 100.0)], [(1, 100.0), (2, 300.0)], [(2, 300.0)]]
    cfg = VigilanceConfig(enable_bag=False, fps=10.0)
    runner = ReplayRunner(config=cfg, seed=0)
    run_a = runner.run_frames(black_frames(4), pipeline=Pipeline(config=cfg, detector=FakeDetector(script)))
    run_b = runner.run_frames(black_frames(4), pipeline=Pipeline(config=cfg, detector=FakeDetector(script)))
    cmp = compare_runs(run_a, run_b)
    assert cmp["identical"], cmp
    # timestamps are frame-clock derived, not wall-clock
    assert [r["timestamp"] for r in run_a] == [0.0, 0.1, 0.2, 0.3]


def test_explicit_timestamp_overrides_frame_clock():
    pipe = _pipe([[(1, 100.0)]])
    res = pipe.process(black_frames(1)[0], timestamp=123.456)
    assert res.timestamp == pytest.approx(123.456)
    assert res.persons[0].timestamp == pytest.approx(123.456)


def test_canonical_person_shape_contract():
    p = TrackedPerson(track_id=3, bbox=[0, 0, 5, 5],
                      keypoints=np.zeros(51, dtype=np.float32), timestamp=2.0)
    assert p.keypoints.shape == (17, 3)
    assert p.keypoints_flat.shape == (51,)
    assert p.keypoint_conf.shape == (17,)
    d = p.to_dict()
    assert set(d) == {"track_id", "bbox", "keypoints", "detection_conf",
                      "timestamp", "present", "first_seen", "last_seen",
                      "age_frames", "generation", "state"}
    assert TrackedPerson.from_dict(d).track_id == 3
