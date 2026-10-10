"""ST-GCN++ real-model tests. torch-gated, checkpoint-gated.

Offline/unit tests run everywhere (numpy-only paths, error paths).
Checkpoint-dependent tests run only when models/stgcnpp/j.pth exists;
otherwise they report SKIPPED — checkpoint unavailable (never fake-passed).
CUDA tests skip when torch.cuda is unavailable.
"""

import numpy as np
import pytest

try:
    import torch  # noqa: F401
    needs_torch = pytest.mark.skipif(False, reason="")
    _has_torch = True
except ImportError:
    needs_torch = pytest.mark.skipif(True, reason="SKIPPED - torch unavailable")
    _has_torch = False
    torch = None  # type: ignore

from core.stgcn_adapter import (ModelNotLoadedError, STGCNPPAdapter,
                                normalize_state_dict)

CKPT = "models/stgcnpp/j.pth"
ckpt_available = pytest.mark.skipif(
    __import__("os").path.exists(CKPT) is False,
    reason="SKIPPED - checkpoint unavailable (models/stgcnpp/j.pth not present)")

TRAINED_CKPT = "models/stgcnpp/best_fight_stgcnpp_v1.pth"
trained_available = pytest.mark.skipif(
    __import__("os").path.exists(TRAINED_CKPT) is False,
    reason="SKIPPED - trained checkpoint unavailable "
           "(models/stgcnpp/best_fight_stgcnpp_v1.pth not present)")
cuda_available = pytest.mark.skipif(
    not (_has_torch and torch.cuda.is_available()),
    reason="SKIPPED - CUDA unavailable")


def _load_adapter(**kw):
    ad = STGCNPPAdapter(temporal_length=100, window_seconds=2.0, **kw)
    return ad.load_model(CKPT), ad


# 1. model construction -------------------------------------------------------
@needs_torch
def test_model_construction_channels_stages():
    from core.stgcn_model import build_model
    m = build_model()
    assert len(m.backbone.gcn) == 10
    assert m.backbone.out_channels == 256
    assert m.fight_head.out_features == 2
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.fight_head.parameters())


# 2. PYSKL graph topology ------------------------------------------------------
@needs_torch
def test_graph_topology_matches_pyskl():
    from core.stgcn_model import build_model, coco_spatial_adjacency
    A = coco_spatial_adjacency()
    assert A.shape == (3, 17, 17)
    assert np.allclose(A[0], np.eye(17))
    m = build_model()
    assert tuple(m.backbone.gcn[0].gcn.A.shape) == (3, 17, 17)
    init = m.backbone.gcn[0].gcn.A.detach().cpu().numpy()
    assert np.allclose(init, A)


# 3/4. input + batch shape validation -------------------------------------------
def test_input_shape_validation():
    ad = STGCNPPAdapter()
    bad = [
        np.zeros((1, 2, 50, 16, 3), np.float32),   # wrong V
        np.zeros((1, 3, 50, 17, 3), np.float32),   # wrong M
        np.zeros((1, 2, 50, 17, 2), np.float32),   # wrong C
        np.zeros((1, 2, 50, 17), np.float32),      # wrong ndim
        np.zeros((1, 2, 99, 17, 3), np.float32),   # wrong T
    ]
    for b in bad:
        with pytest.raises(ValueError):
            ad._validate_batch(b)


# 5/6. state_dict normalization + head removal ----------------------------------
def test_state_dict_normalization():
    bare = {"backbone.data_bn.weight": 1, "cls_head.fc_cls.weight": 2,
            "module.backbone.gcn.0.gcn.A": 3}
    out = normalize_state_dict(bare)
    assert out["backbone.data_bn.weight"] == 1
    assert out["backbone.gcn.0.gcn.A"] == 3
    mm = {"meta": {}, "state_dict": {"cls_head.fc.weight": 4}}
    assert normalize_state_dict(mm, target="mmaction2")["cls_head.fc.weight"] == 4
    assert normalize_state_dict(
        {"cls_head.fc.weight": 4}, target="pyskl")["cls_head.fc_cls.weight"] == 4


# 7/8/9. backbone loading, missing/unexpected keys -------------------------------
@ckpt_available
@needs_torch
def test_backbone_checkpoint_loading():
    import torch as _t
    from core.stgcn_model import build_model
    obj = _t.load(CKPT, map_location="cpu", weights_only=True)
    sd = normalize_state_dict(obj)
    bd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    head = sorted(k for k in sd if k.startswith("cls_head."))
    assert head == ["cls_head.fc_cls.bias", "cls_head.fc_cls.weight"]
    assert obj["cls_head.fc_cls.weight"].shape == (60, 256)  # ignored, not loaded
    m = build_model()
    res = m.backbone.load_state_dict(bd, strict=True)
    assert res.missing_keys == [] and res.unexpected_keys == []


@ckpt_available
@needs_torch
def test_adapter_load_reports_keys():
    info, ad = _load_adapter()
    assert info.loaded and info.missing_keys == [] and info.unexpected_keys == []
    assert info.ignored_head_keys == ["cls_head.fc_cls.bias", "cls_head.fc_cls.weight"]
    assert ad.is_ready and not ad.head_trained


@needs_torch
def test_incompatible_checkpoint_rejected(tmp_path):
    ad = STGCNPPAdapter()
    bad = tmp_path / "bad.pth"
    torch.save({"backbone.foo": torch.zeros(3)}, str(bad))
    with pytest.raises(ValueError, match="unexplained missing keys"):
        ad.load_model(str(bad))
    empty = tmp_path / "empty.pth"
    torch.save({"cls_head.fc.weight": torch.zeros(2, 256)}, str(empty))
    with pytest.raises(ValueError, match="no backbone"):
        ad.load_model(str(empty))
    with pytest.raises(FileNotFoundError, match="not found"):
        ad.load_model("models/stgcnpp/does-not-exist.pth")


# 10/12/19. CPU forward + batch + 2-class shape ----------------------------------
@ckpt_available
@needs_torch
def test_cpu_forward_single():
    _, ad = _load_adapter(device="cpu")
    x = np.random.rand(1, 2, 100, 17, 3).astype(np.float32)
    preds = ad.predict_detailed([(1, 0, 2, 0)], x)
    assert len(preds) == 1
    p = preds[0]
    assert len(p.logits) == 2 and len(p.probabilities) == 2
    assert abs(sum(p.probabilities) - 1.0) < 1e-5
    assert p.head_status == "UNTRAINED_HEAD"
    assert p.device == "cpu"


@ckpt_available
@pytest.mark.parametrize("n", [1, 2, 4, 8])
@needs_torch
def test_batch_inference_shapes(n):
    _, ad = _load_adapter()
    x = np.random.rand(n, 2, 100, 17, 3).astype(np.float32)
    scores = ad.predict_batch(x)
    assert len(scores) == n and all(isinstance(s, float) for s in scores)


# 11. CUDA forward -----------------------------------------------------------------
@ckpt_available
@cuda_available
@needs_torch
def test_cuda_forward():
    _, ad = _load_adapter()
    ad.load_model(CKPT, device="cuda")
    x = np.random.rand(1, 2, 100, 17, 3).astype(np.float32)
    preds = ad.predict_detailed([(1, 0, 2, 0)], x)
    assert preds[0].device == "cuda"


# 13. pair-key mapping -----------------------------------------------------------------
@ckpt_available
@needs_torch
def test_batch_index_to_pair_key_mapping():
    _, ad = _load_adapter()
    keys = [(1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 3, 0), (4, 0, 5, 0)]
    x = np.random.rand(4, 2, 100, 17, 3).astype(np.float32)
    preds = ad.predict_detailed(keys, x)
    assert [p.pair_key for p in preds] == keys


# 14/17. ModelNotLoadedError, no fakes ----------------------------------------------------
def test_model_not_loaded_error():
    ad = STGCNPPAdapter()
    assert not ad.is_ready
    with pytest.raises(ModelNotLoadedError):
        ad.predict_batch(np.zeros((1, 2, 50, 17, 3), dtype=np.float32))


# 15. insufficient history (adapter-level gate) --------------------------------------------
@ckpt_available
@needs_torch
def test_insufficient_history_no_inference_pipeline():
    from tests.conftest import FakeDetector, black_frames
    from config import VigilanceConfig
    from core.pipeline import Pipeline
    cfg = VigilanceConfig()
    cfg.fight_cfg.min_track_age_frames = 1
    pipe = Pipeline(config=cfg, detector=FakeDetector([[(1, 40.0), (2, 220.0)]]))
    pipe.load_fight_model(CKPT)
    res = pipe.process(black_frames(1)[0])
    assert pipe.graph.edges[res.pair_keys[0]].score is None


# 16. real graph score propagation ------------------------------------------------------------
@trained_available
@needs_torch
def test_real_graph_score_propagation():
    from tests.conftest import FakeDetector, black_frames
    from config import VigilanceConfig
    from core.pipeline import Pipeline
    cfg = VigilanceConfig(fps=10.0)
    cfg.fight_cfg.min_track_age_frames = 1
    script = [[(1, 40.0), (2, 220.0)]] * 41
    pipe = Pipeline(config=cfg, detector=FakeDetector(script))
    pipe.load_fight_model(TRAINED_CKPT)
    assert pipe.pair_adapter.head_trained
    for f in black_frames(41):
        res = pipe.process(f)
    edge = pipe.graph.edges[res.pair_keys[0]]
    assert edge.score is not None and 0.0 <= edge.score <= 1.0
    assert res.scene is not None


@ckpt_available
@needs_torch
def test_untrained_checkpoint_does_not_reach_graph():
    from tests.conftest import FakeDetector, black_frames
    from config import VigilanceConfig
    from core.pipeline import Pipeline
    cfg = VigilanceConfig(fps=10.0)
    cfg.fight_cfg.window_seconds = 0.4
    cfg.fight_cfg.temporal_length = 8
    cfg.fight_cfg.min_track_age_frames = 1
    script = [[(1, 40.0), (2, 220.0)]] * 8
    pipe = Pipeline(config=cfg, detector=FakeDetector(script))
    pipe.load_fight_model(CKPT)
    assert not pipe.pair_adapter.head_trained
    for f in black_frames(8):
        res = pipe.process(f)
    edge = pipe.graph.edges[res.pair_keys[0]]
    assert edge.score is None


# 18. frozen backbone ---------------------------------------------------------------------------
@ckpt_available
@needs_torch
def test_frozen_backbone_default():
    _, ad = _load_adapter()
    frozen = [p.requires_grad for p in ad._model.backbone.parameters()]
    assert frozen and not any(frozen)
    assert all(p.requires_grad for p in ad._model.fight_head.parameters())


# 20. deterministic preprocessing ------------------------------------------------------------------
def test_deterministic_preprocessing():
    from core.stgcn_adapter import preprocess_pair
    from core.pairs import PairManager, generate_candidate_pairs
    from core.skeleton import normalize_person
    from tests.conftest import make_person
    pm = PairManager()
    p1, p2 = make_person(1, x0=40.0), make_person(2, x0=220.0)
    for i in range(4):
        sk = {1: normalize_person(p1), 2: normalize_person(p2)}
        from config import FightConfig
        fc = FightConfig()
        cands = generate_candidate_pairs([p1, p2], sk, fc.pair_max_distance,
                                         fc.min_person_visibility, fc.max_candidate_pairs)
        pm.update(cands, sk, i * 0.1)
    buf = pm.open_tracks()[0].buffer
    a = preprocess_pair(buf, 0.5, 10)
    b = preprocess_pair(buf, 0.5, 10)
    assert np.array_equal(a, b) and a.shape == (2, 10, 17, 3)


# 21. trained fight checkpoint ------------------------------------------------------
@trained_available
@needs_torch
def test_trained_fight_checkpoint_loads():
    ad = STGCNPPAdapter(temporal_length=100, window_seconds=4.0)
    info = ad.load_model(TRAINED_CKPT)
    assert info.loaded and info.missing_keys == [] and info.unexpected_keys == []
    assert ad.head_trained
    x = np.random.rand(1, 2, 100, 17, 3).astype(np.float32)
    p = ad.predict_detailed([(1, 0, 2, 0)], x)[0]
    assert p.head_status == "TRAINED_HEAD"
    assert abs(sum(p.probabilities) - 1.0) < 1e-5
