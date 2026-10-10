"""Abandoned-bag / bag-snatch logic (person-bag distance + separation timer).

Ported WITHOUT behavior change to the rule itself. Foundational fixes:
- ``update(results, timestamp)``: timestamp is an explicit argument.
- No import-time side effects; global instance is lazy.
- ``reset()`` clears bag tracks for deterministic replay.
- ``logging`` instead of ``print``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from core.logging_utils import get_logger

log = get_logger("bag")


def calculate_distance(box1, box2) -> float:
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return math.hypot(c1_x - c2_x, c1_y - c2_y)


def calculate_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


class BagDetector:
    def __init__(self, proximity_threshold: float = 150.0, speed_threshold: float = 30.0):
        self.tracked_bags: Dict[int, dict] = {}
        self.next_bag_id = 1
        self.proximity_threshold = float(proximity_threshold)
        self.speed_threshold = float(speed_threshold)

    def reset(self) -> None:
        self.tracked_bags.clear()
        self.next_bag_id = 1

    def update(self, results, timestamp: float) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "alert": False, "bag_bbox": None, "person_bbox": None,
            "distance": 0.0, "message": "Normal",
        }
        if results is None or len(results) == 0:
            return output
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return output

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        person_boxes = [boxes[i] for i in range(len(boxes)) if int(classes[i]) == 0 and confs[i] > 0.3]
        bag_boxes = [boxes[i] for i in range(len(boxes)) if int(classes[i]) == 26 and confs[i] > 0.2]

        current_tracked_bags: Dict[int, dict] = {}
        for bag_bbox in bag_boxes:
            bag_center = calculate_center(bag_bbox)
            min_p_dist = float("inf")
            nearest_p_bbox = None
            for p_bbox in person_boxes:
                d = calculate_distance(bag_bbox, p_bbox)
                if d < min_p_dist:
                    min_p_dist = d
                    nearest_p_bbox = p_bbox

            best_match_id = None
            best_match_dist = float("inf")
            for t_id, t_data in self.tracked_bags.items():
                d = math.hypot(bag_center[0] - t_data["center"][0], bag_center[1] - t_data["center"][1])
                if d < 100.0 and d < best_match_dist:
                    best_match_dist = d
                    best_match_id = t_id

            if best_match_id is None:
                bag_id = self.next_bag_id
                self.next_bag_id += 1
                snatch_time = None
            else:
                bag_id = best_match_id
                prev_dist = self.tracked_bags[bag_id]["min_person_dist"]
                snatch_time = self.tracked_bags[bag_id]["snatch_start_time"]
                if prev_dist < self.proximity_threshold and min_p_dist > self.proximity_threshold:
                    if (min_p_dist - prev_dist) > self.speed_threshold and snatch_time is None:
                        snatch_time = float(timestamp)
                if min_p_dist < self.proximity_threshold:
                    snatch_time = None

            current_tracked_bags[bag_id] = {
                "center": bag_center,
                "min_person_dist": min_p_dist,
                "snatch_start_time": snatch_time,
                "bbox": bag_bbox.tolist(),
            }
            if snatch_time is not None and float(timestamp) - snatch_time > 1.5:
                log.info("event=bag_alert dist=%.1f ts=%.3f", min_p_dist, timestamp)
                output = {
                    "alert": True,
                    "bag_bbox": bag_bbox.tolist(),
                    "person_bbox": nearest_p_bbox.tolist() if nearest_p_bbox is not None else [],
                    "distance": round(float(min_p_dist), 2),
                    "message": "Bag snatching alert! Separation persisted > 1.5s.",
                }
        self.tracked_bags = current_tracked_bags
        return output


_global_bag_detector: Optional[BagDetector] = None


def get_detector() -> BagDetector:
    global _global_bag_detector
    if _global_bag_detector is None:
        _global_bag_detector = BagDetector()
    return _global_bag_detector


def reset() -> None:
    if _global_bag_detector is not None:
        _global_bag_detector.reset()


def check(results, timestamp: Optional[float] = None) -> Dict[str, Any]:
    if timestamp is None:
        timestamp = time.time()
        log.debug("event=bag_check_no_timestamp fallback=time.time()")
    return get_detector().update(results, float(timestamp))
