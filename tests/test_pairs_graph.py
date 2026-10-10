"""Pair / skeleton / graph / adapter tests. numpy + pytest only.

Covers the 20 required cases (Part 22): pair counts 0/1/2/3, spatial
filtering, ordering, generation freshness, LOST grace, closure,
normalization invariance/robustness, resampling, graph verdicts
(1-edge / hub-assault / split components / clique / quiet crowd),
inference gating, plus adapter shapes, config parity, and end-to-end
pipeline scenarios with injected demo scores (no model = no invented
scores anywhere).
"""

import numpy as np
import pytest

from config import FightConfig, VigilanceConfig
from core.graph import (GraphConfig, InteractionGraph, SceneCandidate,
                        overlay_lines, reason_scene)
from core.pairs import (PairManager, bbox_iou, compute_pair_features,
                        generate_candidate_pairs, pair_key_for)
from core.pipeline import Pipeline
from core.skeleton import (COCO_EDGES, COCO_KEYPOINT_NAMES, NUM_JOINTS,
                           YOLO_TO_CANONICAL, bones_from_joints,
                           modality_pack, motion_from_pair, normalize_person)
from core.stgcn_adapter import (ModelNotLoadedError, STGCNPPAdapter,
                                is_buffer_ready, preprocess_pair,
                                resample_buffer)
from core.tracked_person import TrackedPerson
from tests.conftest import FakeDetector, black_frames, make_person


def _skels(persons):
    from core.skeleton import normalize_person as _n
    return {p.track_id: _n(p) for p in persons}


def _gen(persons, **kw):
    fc = FightConfig(**kw)
    return generate_candidate_pairs(persons, _skels(persons), fc.pair_max_distance,
                                    fc.min_person_visibility, fc.max_candidate_pairs)


def _graph_with(scores, n=4):
    from core.graph import EdgeInfo
    g = InteractionGraph()
    for i in range(1, n + 1):
        g.nodes.add(i)
    for (a, b), s in scores.items():
        key = (a, 0, b, 0)
        g.edges[key] = EdgeInfo(key=key, person_ids=(a, b), score=s)
    return g


# 1-4: pair counts ---------------------------------------------------------
def test_zero_people_zero_pairs():
    assert _gen([]) == []


def test_one_person_zero_pairs():
    assert _gen([make_person(1)]) == []


def test_two_people_one_pair():
    cands = _gen([make_person(1, x0=40.0), make_person(2, x0=220.0)])
    assert len(cands) == 1
    assert cands[0]["key"] == (1, 0, 2, 0)


def test_three_people_three_pairs():
    persons = [make_person(1, x0=40.0), make_person(2, x0=220.0), make_person(3, x0=400.0)]
    cands = _gen(persons, pair_max_distance=30.0)
    assert len(cands) == 3


# 5-6: spatial filter -------------------------------------------------------
def test_distant_people_filtered():
    persons = [make_person(1, x0=40.0), make_person(2, x0=2000.0)]
    assert _gen(persons) == []


def test_nearby_people_candidate():
    cands = _gen([make_person(1, x0=40.0), make_person(2, x0=220.0)])
    assert len(cands) == 1
    assert cands[0]["filter"]["norm_dist"] <= 3.0


# 7: ordering -----------------------------------------------------------------
def test_pair_ordering_canonical():
    a, b = make_person(2), make_person(7)
    assert pair_key_for(a, b) == pair_key_for(b, a) == (2, 0, 7, 0)


# 8: generation freshness -------------------------------------------------------
def test_generation_change_fresh_pair_state():
    pm = PairManager()
    p1 = make_person(1, x0=40.0)
    p2 = make_person(2, x0=220.0)
    live = pm.update(_gen([p1, p2]), _skels([p1, p2]), 0.0)
    assert live[0].frame_count == 1
    p2b = make_person(2, x0=220.0)
    p2b.generation = 5
    live = pm.update(_gen([p1, p2b]), _skels([p1, p2b]), 0.1)
    assert live[0].key == (1, 0, 2, 5)
    assert live[0].frame_count == 1
    assert len(live[0].buffer) == 1


# 9-10: grace and closure ----------------------------------------------------------
def test_lost_grace_then_close():
    pm = PairManager(pair_grace_seconds=0.5)
    p1, p2 = make_person(1, x0=40.0), make_person(2, x0=220.0)
    pm.update(_gen([p1, p2]), _skels([p1, p2]), 0.0)
    live = pm.update([], {}, 0.3)
    assert live == []
    assert len(pm.open_tracks()) == 1  # grace holds
    assert pm.open_tracks()[0].frame_count == 1
    pm.update([], {}, 0.6)
    assert pm.open_tracks() == []  # timed out
    assert pm.tracks[(1, 0, 2, 0)].buffer == []  # history cleared


# 11: normalization invariance ---------------------------------------------------------
def test_normalization_translation_scale_invariant():
    base = make_person(1, x0=100.0)
    moved = make_person(1, x0=400.0)
    nb, nm = normalize_person(base), normalize_person(moved)
    assert np.allclose(nb.xy, nm.xy, atol=1e-5)
    scaled = make_person(1, x0=100.0)
    arr = np.asarray(scaled.keypoints)
    arr[:, :2] = (arr[:, :2] - 100.0) * 2.0 + 100.0
    scaled.keypoints = arr
    scaled.bbox = [100.0, 80.0, 220.0, 440.0]
    ns = normalize_person(scaled)
    assert np.allclose(nb.xy, ns.xy, atol=1e-4)


# 12: missing keypoints ------------------------------------------------------------------
@pytest.mark.parametrize("case", ["missing_hips", "missing_shoulders", "tiny_bbox",
                                  "zero_bbox", "low_conf", "partial"])
def test_missing_keypoints_finite(case):
    p = make_person(1, x0=100.0)
    k = np.asarray(p.keypoints)
    if case == "missing_hips":
        k[11, 2] = k[12, 2] = 0.0
    elif case == "missing_shoulders":
        k[5, 2] = k[6, 2] = 0.0
    elif case == "tiny_bbox":
        p.bbox = [100.0, 100.0, 101.0, 102.0]
    elif case == "zero_bbox":
        p.bbox = [50.0, 50.0, 50.0, 50.0]
    elif case == "low_conf":
        k[:, 2] = 0.05
    elif case == "partial":
        k[::2, 2] = 0.0
        k[5, 2] = k[6, 2] = k[11, 2] = k[12, 2] = 0.0
    p.keypoints = k
    s = normalize_person(p)
    assert np.all(np.isfinite(s.xy)) and np.all(np.isfinite(s.root))
    assert np.isfinite(s.scale) and s.scale > 0
    pack = modality_pack(s, None, 0.0)
    for v in pack.values():
        assert np.all(np.isfinite(v))


# 13: resampling ----------------------------------------------------------------------------
def test_resampling_deterministic_fixed_length():
    pm = PairManager()
    p1, p2 = make_person(1, x0=40.0), make_person(2, x0=220.0)
    for i in range(5):
        pm.update(_gen([p1, p2]), _skels([p1, p2]), i * 0.1)
    track = pm.open_tracks()[0]
    a = resample_buffer(track.buffer, 0.5, 10)
    b = resample_buffer(track.buffer, 0.5, 10)
    assert len(a) == 10
    assert [e["t"] for e in a] == [e["t"] for e in b]
    t = preprocess_pair(track.buffer, 0.5, 10)
    assert t.shape == (2, 10, 17, 3)


# 14-18: graph verdicts --------------------------------------------------------------------------
def test_one_strong_pair_single_edge():
    g = _graph_with({(1, 2): 0.92, (1, 3): 0.2, (2, 3): 0.1}, n=3)
    v = reason_scene(g, GraphConfig())
    assert v.scene == SceneCandidate.FIGHT
    assert len(v.incidents) == 1 and v.incidents[0].candidate == SceneCandidate.FIGHT
    assert v.features["number_of_high_score_edges"] == 1


def test_two_edges_sharing_person_group_assault():
    g = _graph_with({(1, 2): 0.94, (1, 3): 0.91, (2, 3): 0.15}, n=3)
    v = reason_scene(g, GraphConfig())
    assert v.scene == SceneCandidate.GROUP_ASSAULT
    assert v.incidents[0].hub_person == 1


def test_three_clique_multi_person_fight():
    g = _graph_with({(1, 2): 0.94, (2, 3): 0.89, (1, 3): 0.91}, n=3)
    v = reason_scene(g, GraphConfig())
    assert v.scene == SceneCandidate.MULTI_PERSON_FIGHT


def test_two_independent_pairs_two_fights():
    g = _graph_with({(1, 2): 0.9, (3, 4): 0.88, (1, 3): 0.1}, n=4)
    v = reason_scene(g, GraphConfig())
    assert v.scene == SceneCandidate.FIGHT
    assert len(v.incidents) == 2
    assert {tuple(sorted(i.members)) for i in v.incidents} == {(1, 2), (3, 4)}


def test_four_clique_multi_person_fight():
    scores = {(1, 2): .9, (1, 3): .9, (1, 4): .9, (2, 3): .9, (2, 4): .9, (3, 4): .9}
    v = reason_scene(_graph_with(scores, n=4), GraphConfig())
    assert v.scene == SceneCandidate.MULTI_PERSON_FIGHT


def test_quiet_crowd_no_fight():
    scores = {(1, 2): .1, (1, 3): .2, (2, 3): .05, (3, 4): .12}
    v = reason_scene(_graph_with(scores, n=4), GraphConfig())
    assert v.scene == SceneCandidate.NO_FIGHT
    assert v.features["number_of_active_people"] == 4
    assert v.features["violent_degree_per_person"] == {"1": 0, "2": 0, "3": 0, "4": 0}


# 19-20: inference gating -------------------------------------------------------------------------------
def test_unrelated_people_no_unnecessary_inference():
    calls = []
    cfg = VigilanceConfig(enable_bag=False, enable_fall=False, fps=10.0)
    cfg.fight_cfg.window_seconds = 0.4
    cfg.fight_cfg.temporal_length = 8
    script = [[(1, 40.0), (2, 220.0), (9, 2500.0)]] * 8
    pipe = Pipeline(config=cfg, detector=FakeDetector(script),
                    pair_scorer=lambda tracks: calls.append(len(tracks)) or [0.1] * len(tracks))
    for f in black_frames(8):
        res = pipe.process(f)
    assert [list(k[:4:2]) for k in res.pair_keys] == [[1, 2]]
    assert calls and all(c == 1 for c in calls)


def test_insufficient_history_no_inference():
    seen = []
    cfg = VigilanceConfig(enable_bag=False, enable_fall=False)
    cfg.fight_cfg.min_track_age_frames = 1  # pair exists immediately; buffer still short
    pipe = Pipeline(config=cfg, detector=FakeDetector([[(1, 40.0), (2, 220.0)]]),
                    pair_scorer=lambda tracks: seen.extend(tracks) or [0.9] * len(tracks))
    res = pipe.process(black_frames(1)[0])
    assert seen == []
    assert pipe.graph.edges[res.pair_keys[0]].score is None


# adapter --------------------------------------------------------------------------------------
def test_adapter_tensor_and_batch_shapes():
    ad = STGCNPPAdapter(temporal_length=10, window_seconds=0.5)
    assert not ad.is_ready
    pm = PairManager()
    p1, p2 = make_person(1, x0=40.0), make_person(2, x0=220.0)
    for i in range(4):
        pm.update(_gen([p1, p2]), _skels([p1, p2]), i * 0.1)
    track = pm.open_tracks()[0]
    assert ad.preprocess_pair(track).shape == (2, 10, 17, 3)
    with pytest.raises(ModelNotLoadedError):
        ad.predict_batch(np.zeros((1, 2, 10, 17, 3), dtype=np.float32))
    with pytest.raises((ImportError, FileNotFoundError)):
        # no torch here -> ImportError; with torch -> FileNotFoundError
        ad.load_model("nope.pt")


def test_adapter_modalities_finite():
    pm = PairManager()
    p1, p2 = make_person(1, x0=40.0), make_person(2, x0=220.0)
    for i in range(3):
        pm.update(_gen([p1, p2]), _skels([p1, p2]), i * 0.1)
    buf = pm.open_tracks()[0].buffer
    assert preprocess_pair(buf, 0.4, 6, "bone").shape == (2, 6, 17, 3)
    jm = preprocess_pair(buf, 0.4, 6, "joint_motion")
    bm = preprocess_pair(buf, 0.4, 6, "bone_motion")
    assert jm.shape == (2, 6, 17, 3) and bm.shape == (2, 6, 17, 3)
    assert np.all(np.isfinite(jm)) and np.all(np.isfinite(bm))
    assert np.all(jm[:, -1] == 0.0) and np.all(bm[:, -1] == 0.0)


def test_pyskl_bone_topology_and_scores():
    from core.skeleton import COCO_EDGES, PYSKL_BONE_PAIRS, bones_pyskl
    pyskl_inward = {(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                    (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                    (1, 0), (3, 1), (2, 0), (4, 2)}
    assert {tuple(sorted(e)) for e in COCO_EDGES} == {tuple(sorted(e)) for e in pyskl_inward}
    assert len(PYSKL_BONE_PAIRS) == 17 and PYSKL_BONE_PAIRS[0] == (0, 0)
    xy = np.arange(34, dtype=np.float32).reshape(17, 2)
    conf = np.full(17, 0.8, dtype=np.float32)
    b = bones_pyskl(xy, conf)
    assert b.shape == (17, 3)
    assert np.all(b[0, :2] == 0.0) and b[0, 2] == pytest.approx(0.8)
    assert np.allclose(b[7, :2], xy[7] - xy[5]) and b[7, 2] == pytest.approx(0.8)
    assert np.all(np.isfinite(b))


def test_frame_normalize_matches_pyskl_formula():
    from core.skeleton import frame_normalize
    xy = np.array([[320.0, 120.0]] * 17, dtype=np.float32)
    conf = np.full(17, 0.9, dtype=np.float32)
    out = frame_normalize(xy, conf, (240.0, 320.0))
    assert out.shape == (17, 3)
    assert np.allclose(out[0, :2], [(320 - 160) / 160, (120 - 120) / 120])
    assert np.all(out[:, 2] == 0.9)
    conf[3] = 0.005
    out = frame_normalize(xy, conf, (240.0, 320.0))
    assert np.all(out[3, :2] == 0.0) and out[3, 2] == pytest.approx(0.005)


def test_motion_sequence_last_zero_score_mean():
    from core.skeleton import motion_sequence
    seq = np.zeros((4, 17, 3), dtype=np.float32)
    seq[:, :, 0] = np.arange(4).reshape(4, 1)
    seq[:, :, 2] = 0.6
    m = motion_sequence(seq)
    assert np.all(m[-1] == 0.0)
    assert np.all(m[:-1, :, 0] == 1.0)
    assert np.all(m[:-1, :, 2] == pytest.approx(0.6))


def test_normalize_state_dict_bare_wrapped_rename():
    from core.stgcn_adapter import normalize_state_dict
    bare = {"backbone.data_bn.weight": [1], "cls_head.fc_cls.weight": [2]}
    assert normalize_state_dict(bare)["cls_head.fc_cls.weight"] == [2]
    wrapped = {"meta": {}, "state_dict": dict(bare)}
    assert normalize_state_dict(wrapped)["cls_head.fc_cls.weight"] == [2]
    mm = {"backbone.data_bn.weight": [1], "cls_head.fc.weight": [3], "cls_head.fc.bias": [4]}
    out = normalize_state_dict(mm, target="pyskl")
    assert out["cls_head.fc_cls.weight"] == [3] and "cls_head.fc.weight" not in out
    out2 = normalize_state_dict(bare, target="mmaction2")
    assert out2["cls_head.fc.weight"] == [2]


def test_buffer_ready_gate():
    pm = PairManager()
    p1, p2 = make_person(1, x0=40.0), make_person(2, x0=220.0)
    pm.update(_gen([p1, p2]), _skels([p1, p2]), 0.0)
    assert not is_buffer_ready(pm.open_tracks()[0], 2.0)
    for i in range(1, 22):
        pm.update(_gen([p1, p2]), _skels([p1, p2]), i * 0.1)
    assert is_buffer_ready(pm.open_tracks()[0], 2.0)


def test_trimmed_buffer_still_reaches_full_window():
    # Regression: trimming at exactly max_history_seconds capped span just
    # under an equal-length readiness window, so inference could NEVER fire
    # no matter how long a pair was observed.
    pm = PairManager(history_seconds=2.0)
    p1, p2 = make_person(1, x0=40.0), make_person(2, x0=220.0)
    for i in range(60):
        pm.update(_gen([p1, p2]), _skels([p1, p2]), i * 0.1)
    track = pm.open_tracks()[0]
    assert len(track.buffer) > 0
    span = track.buffer[-1]["t"] - track.buffer[0]["t"]
    assert span >= 2.0
    assert is_buffer_ready(track, 2.0)


# config -----------------------------------------------------------------------------------------
def test_fight_config_defaults_and_parity():
    fc = FightConfig()
    assert (fc.window_seconds, fc.target_fps, fc.temporal_length) == (4.0, 25, 100)
    assert set(vars(fc.graph)) == set(vars(GraphConfig()))
    cfg = VigilanceConfig.from_file("config/default.json")
    cfg.validate()
    assert cfg.fight_cfg.graph.violence_edge_threshold == 0.3


# end-to-end pipeline scenarios -------------------------------------------------------------------------
def _pipe_scores(scores, **fc_kw):
    fc = FightConfig(window_seconds=0.4, temporal_length=8,
                     min_track_age_frames=1, **fc_kw)
    cfg = VigilanceConfig(enable_bag=False, enable_fall=False, fps=10.0)
    cfg.fight_cfg = fc
    table = {}
    for k, v in scores.items():
        table[tuple(sorted(k))] = v
    def scorer(tracks):
        return [table.get(tuple(sorted((t.id_a, t.id_b))), 0.0) for t in tracks]
    return cfg, scorer


def test_pipeline_group_assault_end_to_end():
    script = [[(1, 40.0), (2, 160.0), (3, 280.0)]] * 8
    cfg, scorer = _pipe_scores({(1, 2): 0.94, (1, 3): 0.91})
    pipe = Pipeline(config=cfg, detector=FakeDetector(script), pair_scorer=scorer)
    for f in black_frames(8):
        res = pipe.process(f)
    assert res.scene.scene == SceneCandidate.GROUP_ASSAULT
    assert res.scene.incidents[0].hub_person == 1
    assert len(res.pair_keys) == 3  # all combinations eligible, none truncated


def test_pipeline_two_separate_fights():
    script = [[(1, 40.0), (2, 220.0), (3, 900.0), (4, 1080.0)]] * 8
    cfg, scorer = _pipe_scores({(1, 2): 0.9, (3, 4): 0.88})
    pipe = Pipeline(config=cfg, detector=FakeDetector(script), pair_scorer=scorer)
    for f in black_frames(8):
        res = pipe.process(f)
    assert res.scene.scene == SceneCandidate.FIGHT
    assert len(res.scene.incidents) == 2


def test_irrelevant_people_preserve_pair_history():
    script = [[(1, 40.0), (2, 220.0)]] * 3 + [[(1, 40.0), (2, 220.0), (7, 1500.0), (8, 1700.0)]] * 3
    cfg, scorer = _pipe_scores({})
    pipe = Pipeline(config=cfg, detector=FakeDetector(script), pair_scorer=scorer)
    keys = []
    for f in black_frames(6):
        keys.append(pipe.process(f).pair_keys)
    assert all((1, 0, 2, 0) in k for k in keys[1:])
    track = pipe.pair_manager.get((1, 0, 2, 0))
    assert track.frame_count == 6


def test_overlay_lines_explain_scene():
    g = _graph_with({(1, 2): 0.94, (1, 3): 0.91, (2, 3): 0.15}, n=3)
    v = reason_scene(g, GraphConfig())
    lines = overlay_lines(v, g)
    assert any("GROUP_ASSAULT" in l for l in lines)
    assert any("P1<->P2" in l and "0.94" in l for l in lines)


# fight-path selection: ST-GCN++ ON, legacy BiLSTM OFF by default ---------------
def test_legacy_fight_off_by_default_stgcn_path_runs():
    cfg = VigilanceConfig(enable_bag=False, enable_fall=False)
    cfg.fight_cfg.min_track_age_frames = 1
    pipe = Pipeline(config=cfg, detector=FakeDetector([[(1, 40.0), (2, 220.0)]]))
    assert all("legacy" not in name.lower() for name, _, _ in pipe._scenarios)
    assert not any("fight" in name.lower() and "legacy" not in name.lower()
                   for name, _, _ in pipe._scenarios)
    res = pipe.process(black_frames(1)[0])
    assert res.pair_keys != []  # ST-GCN++ pair path is the active fight path
    assert res.scene is not None


def test_legacy_fight_opt_in_registers_bilstm():
    cfg = VigilanceConfig(enable_bag=False, enable_fall=False, enable_legacy_fight=True)
    pipe = Pipeline(config=cfg, detector=FakeDetector([[(1, 40.0), (2, 220.0)]]))
    assert any("legacy" in name.lower() for name, _, _ in pipe._scenarios)
    res = pipe.process(black_frames(1)[0])
    assert isinstance(res.alerts, dict)


def test_fight_disabled_skips_pair_graph():
    cfg = VigilanceConfig(enable_bag=False, enable_fall=False, enable_fight=False)
    cfg.fight_cfg.min_track_age_frames = 1
    pipe = Pipeline(config=cfg, detector=FakeDetector([[(1, 40.0), (2, 220.0)]]))
    res = pipe.process(black_frames(1)[0])
    assert res.scene is None
    assert res.pair_keys == []
    assert len(res.persons) == 2  # tracking/detection unaffected
