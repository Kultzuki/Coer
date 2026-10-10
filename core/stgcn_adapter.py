"""ST-GCN++ adapter. Framework-isolated pair-violence interface.

Pipeline view: pair_sequence -> preprocess_pair() -> tensor -> predict()
-> pair violence probability. PYSKL specifics stay inside this module; the
rest of the pipeline never imports torch, pyskl, or mmaction2.

HONESTY CONTRACT: a pretrained action-recognition ST-GCN++ checkpoint is NOT
a FIGHT/NO_FIGHT classifier. This adapter only guarantees the backbone loads
and accepts the VIGLANCE representation. predict() with no loaded model
raises instead of inventing scores. Fine-tuning the head is a later stage.

Expected model tensor (PYSKL 2D convention): (N, M, T, V, C) with M=2
people, V=17 COCO joints, C=3 (x, y, confidence). Per-pair preprocess
returns (2, T, 17, C); predict_batch stacks to (N, 2, T, 17, C).

Model-input normalization is PYSKL `PreNormalize2D` ('fix'): frame-size
mapping to [-1, 1] with the raw score channel untouched. The internal
body-scale `CanonicalSkeleton` is NOT model input (released checkpoints
were trained on frame-normalized coordinates); it remains the
representation for candidate selection, features, and diagnostics.

No Kinetics-Skeleton ST-GCN++ exists in either framework (PYSKL Kinetics
support is PoseC3D-only); the fine-tune path starts from NTU60-pretrained
weights with a replaced classification head.

numpy-only until load_model() is called (torch imported lazily there).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from core.logging_utils import get_logger
from core.pairs import PairTrack
from core.skeleton import (NUM_JOINTS, bones_pyskl, frame_normalize,
                           motion_sequence)

log = get_logger("stgcn")

MODEL_TENSOR_ORDER = "(N, M, T, V, C)"
ADAPTER_PEOPLE = 2
ADAPTER_JOINTS = NUM_JOINTS
ADAPTER_CHANNELS = 3

MODALITIES = ("joint", "bone", "joint_motion", "bone_motion")


class ModelNotLoadedError(RuntimeError):
    pass


def resample_buffer(buffer: List[dict], window_seconds: float,
                    temporal_length: int) -> List[dict]:
    if not buffer or temporal_length < 1:
        return []
    last_t = float(buffer[-1]["t"])
    start = last_t - float(window_seconds)
    grid = [start + (i + 0.5) * float(window_seconds) / temporal_length
            for i in range(temporal_length)]
    times = [float(e["t"]) for e in buffer]
    out = []
    for g in grid:
        best = min(range(len(times)), key=lambda i: abs(times[i] - g))
        out.append(buffer[best])
    return out


def _raw_pixels(skel) -> tuple:
    raw = skel.xy * skel.scale + skel.root
    return raw.astype(np.float32), np.asarray(skel.confidence, dtype=np.float32)


def _pair_sequences(buffer: List[dict], window_seconds: float,
                    temporal_length: int, img_hw) -> List[np.ndarray]:
    seq = resample_buffer(buffer, window_seconds, temporal_length)
    streams = []
    for key in ("a", "b"):
        norm = np.zeros((len(seq), NUM_JOINTS, ADAPTER_CHANNELS), dtype=np.float32)
        for i, e in enumerate(seq):
            raw, conf = _raw_pixels(e[key])
            norm[i] = frame_normalize(raw, conf, img_hw)
        streams.append(norm)
    return streams


def preprocess_pair(buffer: List[dict], window_seconds: float,
                    temporal_length: int, modality: str = "joint",
                    img_hw=(1080.0, 1920.0)) -> np.ndarray:
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality}")
    streams = _pair_sequences(buffer, window_seconds, temporal_length, img_hw)
    t = len(streams[0])
    if modality == "joint":
        return np.stack(streams, axis=0)
    if modality == "bone":
        return np.stack([
            np.stack([bones_pyskl(f[:, :2], f[:, 2]) for f in s], axis=0)
            for s in streams], axis=0)
    if modality == "joint_motion":
        return np.stack([motion_sequence(s) for s in streams], axis=0)
    return np.stack([
        motion_sequence(np.stack([bones_pyskl(f[:, :2], f[:, 2]) for f in s], axis=0))
        for s in streams], axis=0)


def is_buffer_ready(track: PairTrack, window_seconds: float,
                    min_coverage: float = 1.0, min_frames: int = 4) -> bool:
    if track.frame_count < min_frames or len(track.buffer) < 2:
        return False
    span = float(track.buffer[-1]["t"]) - float(track.buffer[0]["t"])
    required = float(window_seconds) * float(min_coverage)
    return span + 1e-6 >= required


@dataclass
class CheckpointInfo:
    path: str
    framework: str = "pyskl"
    config_name: str = ""
    num_classes: int = 0
    loaded: bool = False
    missing_keys: List[str] = None
    unexpected_keys: List[str] = None
    ignored_head_keys: List[str] = None

    def __post_init__(self):
        self.missing_keys = list(self.missing_keys or [])
        self.unexpected_keys = list(self.unexpected_keys or [])
        self.ignored_head_keys = list(self.ignored_head_keys or [])


@dataclass
class PairPrediction:
    pair_key: tuple
    logits: List[float]            # raw 2-class logits [NO_FIGHT, FIGHT]
    probabilities: List[float]     # softmax(logits)
    fight_probability: float       # probabilities[1]; UNTRAINED_HEAD until fine-tuned
    timestamp: float = 0.0
    device: str = "cpu"
    inference_time_ms: float = 0.0
    head_status: str = "UNTRAINED_HEAD"


def normalize_state_dict(obj, target: str = "pyskl") -> dict:
    """Unwrap + cross-framework head rename. Pure python, no torch needed.

    Accepts three containers: PYSKL bare OrderedDict state dict, MMAction2
    {'meta', 'state_dict'} wrap, and VIGLANCE training artifacts carrying
    weights under 'model_state_dict'. Head keys differ (`cls_head.fc_cls.*`
    vs `cls_head.fc.*`); the 690 backbone keys are shape-identical and pass
    through untouched, as do VIGLANCE `fight_head.*` keys. Returns a plain
    dict ready for load_state_dict.
    """
    if isinstance(obj, dict) and "model_state_dict" in obj:
        sd = obj["model_state_dict"]
    else:
        sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    out = {str(k).removeprefix("module."): v for k, v in dict(sd).items()}
    src, dst = (("cls_head.fc.", "cls_head.fc_cls.") if target == "pyskl"
                else ("cls_head.fc_cls.", "cls_head.fc."))
    for tail in ("weight", "bias"):
        if src + tail in out and dst + tail not in out:
            out[dst + tail] = out.pop(src + tail)
    return out


class STGCNPPAdapter:
    def __init__(self, temporal_length: int = 100, window_seconds: float = 4.0,
                 modality: str = "joint",
                 scorer: Optional[Callable[[np.ndarray], List[float]]] = None,
                 img_hw=(1080.0, 1920.0), device: str = "auto",
                 freeze_backbone: bool = True, num_classes: int = 2):
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality}")
        if num_classes != 2:
            raise ValueError("VIGLANCE head is binary (num_classes=2)")
        self.temporal_length = int(temporal_length)
        self.window_seconds = float(window_seconds)
        self.modality = modality
        self.img_hw = (float(img_hw[0]), float(img_hw[1]))
        self.device_name = device
        self.freeze_backbone = bool(freeze_backbone)
        self.num_classes = int(num_classes)
        self._scorer = scorer
        self._model = None
        self._device = "cpu"
        self.head_trained = False  # flips only after fight fine-tuning
        self.checkpoint = CheckpointInfo(path="")

    @property
    def is_ready(self) -> bool:
        return self._model is not None or self._scorer is not None

    def resolve_device(self, want: Optional[str] = None) -> str:
        want = want or self.device_name
        if want == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
                return "cuda"
            except ImportError as exc:
                raise ImportError("torch is required for device handling.") from exc
        if want == "cpu":
            return "cpu"
        if want == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        raise ValueError(f"device must be auto/cpu/cuda, got {want!r}")

    def load_model(self, path: str, config_name: str = "",
                   num_classes: int = 2, device: Optional[str] = None,
                   freeze_backbone: Optional[bool] = None) -> CheckpointInfo:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load an ST-GCN++ checkpoint. "
                              "Install it (CPU builds suffice for smoke tests).") from exc
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ST-GCN++ checkpoint not found: {path}\n"
                f"Download the official PYSKL NTU60 2D joint weights "
                f"(stgcnpp_ntu60_xsub_hrnet/j.pth) into models/stgcnpp/ "
                f"or set fight.model.checkpoint / VIGILANCE_FIGHT_MODEL_CHECKPOINT.")
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            log.warning("event=stgcn_weights_only_retry path=%s", path)
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise ValueError(
                    f"ST-GCN++ checkpoint at {path} could not be parsed "
                    f"(tried weights_only True and False): {exc}") from exc
        sd = normalize_state_dict(obj, target="pyskl")
        metadata = obj if isinstance(obj, dict) else {}
        metadata_checks = {
            "temporal_length": self.temporal_length,
            "num_people": ADAPTER_PEOPLE,
            "num_keypoints": ADAPTER_JOINTS,
            "input_channels": ADAPTER_CHANNELS,
        }
        for name, expected in metadata_checks.items():
            actual = metadata.get(name)
            if actual is not None and int(actual) != expected:
                raise ValueError(
                    f"checkpoint metadata {name}={actual} does not match "
                    f"adapter value {expected}")
        class_names = metadata.get("class_names")
        if class_names is not None:
            normalized_classes = [
                str(name).strip().lower().replace("_", "").replace("-", "")
                for name in class_names
            ]
            no_fight_names = {"nofight", "nonfight", "normal", "nonviolent"}
            fight_names = {"fight", "violent", "violence"}
            if (len(normalized_classes) != 2
                    or normalized_classes[0] not in no_fight_names
                    or normalized_classes[1] not in fight_names):
                raise ValueError(
                    "checkpoint class_names must map index 0 to no-fight and "
                    "index 1 to fight")
        backbone_sd = {k[len("backbone."):]: v for k, v in sd.items()
                       if k.startswith("backbone.")}
        head_keys = sorted(k for k in sd if k.startswith("cls_head."))
        if not backbone_sd:
            raise ValueError(
                f"ST-GCN++ checkpoint at {path} contains no backbone.* keys "
                f"({len(sd)} keys total); refusing to load an incompatible file.")
        from core.stgcn_model import build_model
        freeze = self.freeze_backbone if freeze_backbone is None else bool(freeze_backbone)
        model = build_model(num_classes=num_classes, freeze_backbone=freeze)
        res = model.backbone.load_state_dict(backbone_sd, strict=False)
        missing = sorted(res.missing_keys)
        unexpected = sorted(k for k in res.unexpected_keys)
        if missing:
            raise ValueError(
                f"ST-GCN++ backbone has {len(missing)} unexplained missing keys "
                f"(showing 10): {missing[:10]}. Checkpoint incompatible; aborting.")
        if unexpected:
            raise ValueError(
                f"ST-GCN++ checkpoint has {len(unexpected)} unexpected backbone keys "
                f"(showing 10): {unexpected[:10]}. Checkpoint incompatible; aborting.")
        head_sd = {k: v for k, v in sd.items() if k.startswith("fight_head.")}
        head_trained = False
        if head_sd:
            expected = {"fight_head.weight": tuple(model.fight_head.weight.shape),
                        "fight_head.bias": tuple(model.fight_head.bias.shape)}
            bad = [k for k, s in expected.items()
                   if k not in head_sd or tuple(head_sd[k].shape) != s]
            if bad:
                raise ValueError(
                    f"ST-GCN++ checkpoint fight head incompatible with "
                    f"num_classes={num_classes}: {bad}. Aborting.")
            res_h = model.fight_head.load_state_dict(
                {k[len("fight_head."):]: v for k, v in head_sd.items()},
                strict=True)
            assert not res_h.missing_keys and not res_h.unexpected_keys
            head_trained = True
        dev = self.resolve_device(device)
        model.to(dev)
        model.eval()
        self._model = model
        self._device = dev
        self.checkpoint = CheckpointInfo(
            path=path, config_name=config_name, num_classes=num_classes,
            loaded=True, missing_keys=missing, unexpected_keys=unexpected,
            ignored_head_keys=head_keys)
        self.head_trained = head_trained
        log.info("event=stgcn_loaded path=%s device=%s frozen=%s ignored_head=%s head=%s",
                 path, dev, freeze, head_keys,
                 "TRAINED_HEAD" if head_trained else "UNTRAINED_HEAD")
        return self.checkpoint

    def preprocess_pair(self, track: PairTrack) -> np.ndarray:
        return preprocess_pair(track.buffer, self.window_seconds,
                               self.temporal_length, self.modality, self.img_hw)

    def _validate_batch(self, batch: np.ndarray) -> np.ndarray:
        arr = np.asarray(batch, dtype=np.float32)
        if arr.ndim != 5:
            raise ValueError(
                f"ST-GCN++ batch must be (N, M, T, V, C), got shape {arr.shape}")
        n, m, t, v, c = (int(d) for d in arr.shape)
        if m != ADAPTER_PEOPLE:
            raise ValueError(f"wrong person count M={m}, expected {ADAPTER_PEOPLE}")
        if v != ADAPTER_JOINTS:
            raise ValueError(f"wrong joint count V={v}, expected {ADAPTER_JOINTS}")
        if c != ADAPTER_CHANNELS:
            raise ValueError(f"wrong channel count C={c}, expected {ADAPTER_CHANNELS}")
        if t != self.temporal_length:
            raise ValueError(
                f"wrong temporal length T={t}, expected {self.temporal_length} "
                f"(resample the buffer first)")
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite values in ST-GCN++ input tensor")
        return arr

    def predict(self, tensor: np.ndarray) -> float:
        scores = self.predict_batch(tensor[np.newaxis, ...])
        return float(scores[0])

    def predict_batch(self, batch: np.ndarray) -> List[float]:
        return [p.fight_probability for p in self.predict_detailed(None, batch)]

    def predict_pairs(self, tracks) -> List[PairPrediction]:
        tracks = list(tracks)
        if not tracks:
            return []
        batch = np.stack([self.preprocess_pair(t) for t in tracks], axis=0)
        return self.predict_detailed([t.key for t in tracks], batch)

    def predict_detailed(self, keys, batch: np.ndarray) -> List[PairPrediction]:
        import time
        if self._scorer is not None:
            arr = self._validate_batch(batch)
            scores = [float(s) for s in self._scorer(arr)]
            now = 0.0
            return [PairPrediction(
                pair_key=k, logits=[0.0, 0.0], probabilities=[1.0 - s, s],
                fight_probability=s, timestamp=now, device="scorer",
                inference_time_ms=0.0,
                head_status="INJECTED_SCORER") for k, s in
                zip(keys if keys is not None else [None] * len(scores), scores)]
        if self._model is None:
            raise ModelNotLoadedError(
                "no ST-GCN++ model or scorer configured; refusing to invent scores")
        arr = self._validate_batch(batch)
        import torch
        keys = list(keys) if keys is not None else [None] * len(arr)
        self._model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), 8):
                chunk = torch.from_numpy(arr[i:i + 8]).to(self._device)
                t0 = time.perf_counter()
                logits = self._model(chunk)
                dt = (time.perf_counter() - t0) * 1000.0 / max(len(chunk), 1)
                probs = torch.softmax(logits, dim=1).cpu().tolist()
                lg = logits.cpu().tolist()
                for k, l, p in zip(keys[i:i + 8], lg, probs):
                    out.append(PairPrediction(
                        pair_key=k, logits=[float(l[0]), float(l[1])],
                        probabilities=[float(p[0]), float(p[1])],
                        fight_probability=float(p[1]), timestamp=0.0,
                        device=self._device, inference_time_ms=dt,
                        head_status=("TRAINED_HEAD" if self.head_trained
                                     else "UNTRAINED_HEAD")))
        for p in out:
            log.info("event=pair_predict pair=%s tensor=(1,2,%d,17,3) device=%s "
                     "logits=[%.4f,%.4f] fight_probability=%.4f [%s]",
                     p.pair_key, self.temporal_length, p.device,
                     p.logits[0], p.logits[1], p.fight_probability, p.head_status)
        return out
