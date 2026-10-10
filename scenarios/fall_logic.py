"""Fall detection (pose rules: torso angle + wide aspect + stillness).

Ported from the legacy scenario module WITHOUT behavior change to the rule
itself. Foundational fixes only:
- ``update(tracked_objects, timestamp)``: timestamp is a required explicit
  argument (was ``time.time()`` captured inside ``check()``).
- Accepts canonical :class:`TrackedPerson` objects AND legacy dicts.
- No import-time side effects; global instance is created lazily.
- ``reset()`` clears per-ID history for deterministic replay.
- ``logging`` instead of ``print``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.logging_utils import get_logger

log = get_logger("fall")


def _as_legacy_list(tracked_objects) -> List[Dict[str, Any]]:
    items = []
    for obj in tracked_objects or []:
        if hasattr(obj, "to_legacy_dict"):
            items.append(obj.to_legacy_dict())
        else:
            items.append(obj)
    return items


class FallDetector:
    def __init__(
        self,
        angle_deg: float = 45.0,
        stillness_sec: float = 1.5,
        min_visible_sec: float = 0.5,
        min_torso_conf: float = 0.1,
    ):
        self.angle_deg = float(angle_deg)
        self.stillness_sec = float(stillness_sec)
        self.min_visible_sec = float(min_visible_sec)
        self.min_torso_conf = float(min_torso_conf)
        self.history: Dict[int, list] = {}

    def reset(self) -> None:
        self.history.clear()

    def update(self, tracked_objects, timestamp: float) -> Dict[str, Any]:
        best_alert: Dict[str, Any] = {
            "alert": False, "person_id": -1, "angle": 0.0,
            "confidence": 0.0, "message": "Normal",
        }
        for obj in _as_legacy_list(tracked_objects):
            person_id = obj["track_id"]
            kp = np.asarray(obj["keypoints"], dtype=np.float32).reshape(-1)
            if person_id not in self.history:
                self.history[person_id] = []
            if float(np.sum(kp)) == 0:
                self.history[person_id].clear()
                continue

            ls_x, ls_y, ls_c = kp[5 * 3:5 * 3 + 3]
            rs_x, rs_y, rs_c = kp[6 * 3:6 * 3 + 3]
            lh_x, lh_y, lh_c = kp[11 * 3:11 * 3 + 3]
            rh_x, rh_y, rh_c = kp[12 * 3:12 * 3 + 3]
            if min(float(ls_c), float(rs_c), float(lh_c), float(rh_c)) < self.min_torso_conf:
                continue

            shoulder_mid_x = (float(ls_x) + float(rs_x)) / 2.0
            shoulder_mid_y = (float(ls_y) + float(rs_y)) / 2.0
            hip_mid_x = (float(lh_x) + float(rh_x)) / 2.0
            hip_mid_y = (float(lh_y) + float(rh_y)) / 2.0

            dx = hip_mid_x - shoulder_mid_x
            dy = hip_mid_y - shoulder_mid_y
            angle = math.degrees(math.atan2(abs(dx), abs(dy)))

            xs = kp[0::3]; ys = kp[1::3]; confs = kp[2::3]
            valid = confs > 0.1
            w, h = 0.0, 0.0
            if int(np.sum(valid)) >= 2:
                w = float(np.max(xs[valid]) - np.min(xs[valid]))
                h = float(np.max(ys[valid]) - np.min(ys[valid]))
            aspect_ratio_wide = w > h

            self.history[person_id].append(
                {"timestamp": float(timestamp), "x": hip_mid_x, "y": hip_mid_y, "h": h}
            )
            self.history[person_id] = [
                item for item in self.history[person_id]
                if float(timestamp) - item["timestamp"] <= self.stillness_sec
            ]

            still = False
            if len(self.history[person_id]) > 3:
                start = self.history[person_id][0]
                elapsed = float(timestamp) - start["timestamp"]
                if elapsed > self.min_visible_sec:
                    displacement = math.hypot(hip_mid_x - start["x"], hip_mid_y - start["y"])
                    if displacement < max(h * 0.15, 15.0):
                        still = True

            alert = (angle > self.angle_deg) and aspect_ratio_wide and still
            confidence = min(1.0, max(0.0, (angle - self.angle_deg) / 45.0)) if alert else 0.0
            if alert:
                log.info("event=fall_alert person=%s angle=%.1f ts=%.3f", person_id, angle, timestamp)
                return {
                    "alert": True, "person_id": person_id,
                    "angle": round(angle, 2), "confidence": round(confidence, 2),
                    "message": f"Fall detected! Person {person_id} is down.",
                }
        return best_alert


_global_fall_detector: Optional[FallDetector] = None


def get_detector() -> FallDetector:
    global _global_fall_detector
    if _global_fall_detector is None:
        _global_fall_detector = FallDetector()
    return _global_fall_detector


def reset() -> None:
    if _global_fall_detector is not None:
        _global_fall_detector.reset()


def check(tracked_objects, timestamp: Optional[float] = None) -> Dict[str, Any]:
    if timestamp is None:
        timestamp = time.time()
        log.debug("event=fall_check_no_timestamp fallback=time.time()")
    return get_detector().update(tracked_objects, float(timestamp))
