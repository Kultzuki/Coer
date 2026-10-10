"""Violence/fight sequence classifier (2-layer BiLSTM-128 + attention).

The NEW fight architecture is explicitly OUT OF SCOPE for this foundation
pass (per task instruction #5). This module preserves the legacy model and
rule behavior exactly and fixes only foundational defects:
- No ``torch`` import at module import time (lazy inside ``_ensure_model``),
  so the package imports cleanly on machines without torch (tests, Kaggle
  CPU checks, config inspection).
- No model load at import time (legacy created ``_global_fight_detector``
  at import, which printed warnings and touched the filesystem on every
  ``import scenarios.fight_logic``). The global instance is now lazy.
- ``update(tracked_objects, timestamp)``: timestamp explicit (previously
  accepted but ignored ``time.time()`` from ``check()``).
- Accepts canonical :class:`TrackedPerson` AND legacy dicts.
- ``reset()`` clears the rolling buffer + counter for deterministic replay.
- ``logging`` instead of ``print``.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from core.logging_utils import get_logger

log = get_logger("fight")


def _build_model(input_size=102, hidden_size=128, num_layers=2, num_classes=2):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Attention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.w = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)

        def forward(self, lstm_output):
            u = torch.tanh(self.w(lstm_output))
            alphas = F.softmax(self.v(u), dim=1)
            return torch.sum(alphas * lstm_output, dim=1)

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True, bidirectional=True,
            )
            self.attention = Attention(hidden_size * 2)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(self.attention(lstm_out))

    return BiLSTMClassifier()


class FightDetector:
    def __init__(
        self,
        model_path: str = "models/fight_classifier.pt",
        sequence_length: int = 30,
        threshold: float = 0.65,
        confirm_frames: int = 5,
    ):
        self.model_path = model_path
        self.sequence_length = int(sequence_length)
        self.threshold = float(threshold)
        self.confirm_frames = int(confirm_frames)
        self.buffer: deque = deque(maxlen=self.sequence_length)
        self.fight_counter = 0
        self.model_loaded = False
        self._model = None
        self._device = None

    def reset(self) -> None:
        self.buffer.clear()
        self.fight_counter = 0

    # ---- lazy torch -----------------------------------------------------
    def _ensure_model(self) -> bool:
        if self.model_loaded:
            return True
        try:
            import torch
        except ImportError:
            log.warning("event=fight_no_torch skip_inference")
            return False
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = _build_model().to(self._device)
            if hasattr(self._model, "eval"):
                self._model.eval()
        if not os.path.exists(self.model_path):
            log.warning("event=fight_no_weights path=%s skip_inference", self.model_path)
            return False
        try:
            import torch.nn.functional as F  # noqa: F401 (kept local on purpose)
            self._model.load_state_dict(
                __import__("torch").load(self.model_path, map_location=self._device, weights_only=True)
            )
            self.model_loaded = True
            log.info("event=fight_weights_loaded path=%s device=%s", self.model_path, self._device)
        except Exception as exc:
            log.warning("event=fight_weights_failed path=%s error=%r", self.model_path, exc)
            return False
        return True

    # ---- inference --------------------------------------------------------
    def update(self, tracked_objects, timestamp: float) -> Dict[str, Any]:
        output: Dict[str, Any] = {"alert": False, "confidence": 0.0, "message": "Normal"}
        items = []
        for obj in tracked_objects or []:
            if hasattr(obj, "to_legacy_dict"):
                items.append(obj.to_legacy_dict())
            else:
                items.append(obj)
        stable = sorted(items, key=lambda x: x["track_id"])
        ordered = np.zeros((2, 51), dtype=np.float32)
        for idx, obj in enumerate(stable[:2]):
            ordered[idx] = np.asarray(obj["keypoints"], dtype=np.float32).reshape(-1)[:51]
        self.buffer.append(ordered.flatten())

        if len(self.buffer) < self.sequence_length:
            return output
        if not self._ensure_model():
            return output

        import torch
        import torch.nn.functional as F

        seq_array = np.array(self.buffer, dtype=np.float32)
        seq_tensor = torch.tensor(seq_array).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(seq_tensor)
            probs = F.softmax(logits, dim=1)
            fight_prob = float(probs[0, 1].item())

        if fight_prob > self.threshold:
            self.fight_counter += 1
        else:
            self.fight_counter = max(0, self.fight_counter - 1)
        if self.fight_counter >= self.confirm_frames:
            log.info("event=fight_alert conf=%.3f ts=%.3f", fight_prob, timestamp)
            output = {
                "alert": True,
                "confidence": round(fight_prob, 4),
                "message": f"Fight detected! Confidence: {fight_prob:.2f}",
            }
        return output


_global_fight_detector: Optional[FightDetector] = None


def get_detector() -> FightDetector:
    global _global_fight_detector
    if _global_fight_detector is None:
        _global_fight_detector = FightDetector()
    return _global_fight_detector


def reset() -> None:
    if _global_fight_detector is not None:
        _global_fight_detector.reset()


def check(tracked_objects, timestamp: Optional[float] = None) -> Dict[str, Any]:
    if timestamp is None:
        timestamp = time.time()
        log.debug("event=fight_check_no_timestamp fallback=time.time()")
    return get_detector().update(tracked_objects, float(timestamp))
