"""Shared fakes: deterministic stand-ins so smoke tests never need torch/cv2/ultralytics."""

import numpy as np

from core.detector import DetectorOutput
from core.tracked_person import TrackedPerson


def standing_keypoints(x0=100.0, y0=100.0, step=20.0, conf=0.9):
    """Upright 17-keypoint skeleton (shoulders above hips)."""
    kps = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        kps[i] = [x0 + (i % 5) * 4.0, y0 + i * step * 0.35, conf]
    # shoulders (5,6) clearly above hips (11,12)
    kps[5] = [x0 - 12.0, y0 + 30.0, conf]
    kps[6] = [x0 + 12.0, y0 + 30.0, conf]
    kps[11] = [x0 - 10.0, y0 + 110.0, conf]
    kps[12] = [x0 + 10.0, y0 + 110.0, conf]
    return kps


def make_person(track_id, x0=100.0, timestamp=0.0, conf=0.9):
    return TrackedPerson(
        track_id=int(track_id),
        bbox=[x0, 80.0, x0 + 60.0, 260.0],
        keypoints=standing_keypoints(x0=x0, conf=conf),
        detection_conf=conf,
        timestamp=float(timestamp),
    )


class FakeDetector:
    """Drop-in for Detector: serves a scripted person-list per frame."""

    def __init__(self, script, det_results=None):
        # script: list where each entry is a list of (track_id, x0) tuples
        self.script = list(script)
        self.det_results = det_results if det_results is not None else []
        self.calls = 0

    def detect(self, frame, timestamp=0.0):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        persons = [
            make_person(tid, x0=x0, timestamp=timestamp)
            for tid, x0 in self.script[idx]
        ]
        for p in persons:
            p.timestamp = float(timestamp)
        return DetectorOutput(persons=persons, pose_results=None, det_results=self.det_results)

    def reset(self):
        self.calls = 0


def black_frames(n=8, h=240, w=320):
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
