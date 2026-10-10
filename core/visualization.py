"""Multi-person visualization: bbox + track ID + COCO skeleton.

cv2 is imported lazily inside the draw call so importing this module never
requires opencv (tests, config inspection stay light).
"""

from __future__ import annotations


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 200),
    (200, 255, 0), (120, 120, 255), (0, 255, 200), (255, 200, 0),
]

# Mirrors GraphConfig.violence_edge_threshold, which visualization cannot import without a cycle.
PAIR_RISK_THRESHOLD = 0.6

PLAIN_ALERT_NAMES = {
    "Fall Detection": "Someone may have fallen",
    "Abandoned Bag": "Unattended bag",
    "Violence/Fight (legacy BiLSTM)": "Possible fight",
    "Weapon Threat": "Weapon detected",
    "Fire Threat": "Fire detected",
    "Smoke Threat": "Smoke detected",
    "Unattended Bag Candidate": "Unattended bag",
}

PLAIN_CATEGORY_NAMES = {
    "weapon": "Weapon",
    "fire": "Fire",
    "smoke": "Smoke",
    "bag": "Bag",
}

# verdict -> (banner text, fill colour, text colour); the text colour varies per entry so contrast holds on every fill.
# Only alert verdicts get a banner. NO_FIGHT is deliberately silent: a calm
# pair score can mean "nothing happening" OR "the pair history never filled"
# (short clips, tracker ID churn, 0-1 visible people), so a reassuring banner
# would be a false claim on the frame.
PLAIN_SCENE_BANNERS = {
    "FIGHT": ("POSSIBLE FIGHT - CHECK NOW", (0, 200, 255), (0, 0, 0)),
    "MULTI_PERSON_FIGHT": ("GROUP FIGHT - CHECK NOW", (0, 0, 255), (255, 255, 255)),
    "GROUP_ASSAULT": ("GROUP ASSAULT - CHECK NOW", (0, 0, 255), (255, 255, 255)),
}
PLAIN_SCENE_NO_BANNER = frozenset({"NO_FIGHT"})
PLAIN_SCENE_FALLBACK = ("WATCHING CLOSELY", (0, 200, 255), (0, 0, 0))

# Top y values chosen so the plain scene banner and plain alert rows clear the y=22 ui.py::_draw_model_banner text.
PLAIN_SCENE_BANNER_TOP = 40
PLAIN_ALERT_TOP = 136


def color_for(track_id: int):
    return PALETTE[int(track_id) % len(PALETTE)]


def title_case(raw) -> str:
    return " ".join(
        part[:1].upper() + part[1:]
        for part in str(raw).replace("_", " ").split() if part
    )


def plain_alert_name(name) -> str:
    if name in PLAIN_ALERT_NAMES:
        return PLAIN_ALERT_NAMES[name]
    return title_case(name)


def plain_category_name(category) -> str:
    if category in PLAIN_CATEGORY_NAMES:
        return PLAIN_CATEGORY_NAMES[category]
    return title_case(category)


def whole_percent(value) -> int:
    if value is None:
        return 0
    return int(float(value) * 100.0 + 0.5)


def _text_size(cv2, text: str, scale: float, thickness: int):
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return width, height, baseline


def _fit_scale(cv2, text: str, frame_width: int, thickness: int,
               max_scale: float = 0.9, min_scale: float = 0.35,
               margin: int = 48) -> float:
    scale = max_scale
    while scale > min_scale:
        width, _, _ = _text_size(cv2, text, scale, thickness)
        if width + margin <= frame_width:
            return scale
        scale -= 0.05
    return min_scale


def draw_label(cv2, frame, text: str, x: int, y: int, scale: float,
               thickness: int, color, chip: bool = False) -> None:
    if not chip:
        cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness)
        return
    width, height, baseline = _text_size(cv2, text, scale, thickness)
    left = max(0, int(x) - 5)
    top = max(0, int(y) - height - baseline + 2)
    cv2.rectangle(frame, (left, top),
                  (left + width + 10, int(y) + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


def draw_tracks(frame, persons, skeleton_conf: float = 0.3, *,
                plain: bool = False) -> None:
    import cv2
    for person in persons:
        color = color_for(person.track_id)
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if plain:
            draw_label(cv2, frame, f"Person {person.track_id}", x1,
                       max(15, y1 - 8), 0.6, 2, color, chip=True)
        else:
            cv2.putText(frame, f"ID {person.track_id} g{person.generation}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        kps = person.keypoints
        for x, y, c in kps:
            if c > skeleton_conf:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        for a, b in COCO_SKELETON:
            xa, ya, ca = kps[a]
            xb, yb, cb = kps[b]
            if ca > skeleton_conf and cb > skeleton_conf:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)


def draw_alerts(frame, alerts, *, plain: bool = False) -> None:
    import cv2
    y = PLAIN_ALERT_TOP if plain else 30
    for name in alerts:
        if plain:
            text = plain_alert_name(name)
            width, _, _ = _text_size(cv2, text, 0.8, 2)
            right = min(frame.shape[1], 15 + width + 10)
        else:
            text = f"{name} ALERT!"
            right = 400
        cv2.rectangle(frame, (10, y - 25), (right, y + 10), (0, 0, 255), -1)
        cv2.putText(frame, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 45


def draw_threats(frame, detections, *, plain: bool = False) -> None:
    import cv2
    colors = {
        "weapon": (0, 0, 255),
        "fire": (0, 80, 255),
        "smoke": (160, 160, 160),
        "bag": (255, 160, 0),
    }
    for detection in detections or []:
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        color = colors.get(detection.category, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if plain:
            raw = getattr(detection, "label", "") or ""
            name = title_case(raw) if str(raw).strip() else plain_category_name(detection.category)
            draw_label(cv2, frame, f"{name} {whole_percent(detection.confidence)}%",
                       x1, max(18, y1 - 6), 0.5, 2, color, chip=True)
        else:
            label = f"{detection.source}:{detection.label} {detection.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_pairs(frame, pair_tracks, persons_by_id, graph=None, *,
               plain: bool = False) -> None:
    import cv2
    for track in pair_tracks:
        a = persons_by_id.get(track.id_a)
        b = persons_by_id.get(track.id_b)
        if a is None or b is None:
            continue
        ax = int((a.bbox[0] + a.bbox[2]) / 2)
        ay = int((a.bbox[1] + a.bbox[3]) / 2)
        bx = int((b.bbox[0] + b.bbox[2]) / 2)
        by = int((b.bbox[1] + b.bbox[3]) / 2)
        score = None
        if graph is not None:
            e = graph.edges.get(track.key)
            score = e.score if e is not None else None
        if plain:
            if score is None or score < PAIR_RISK_THRESHOLD:
                continue
            cv2.line(frame, (ax, ay), (bx, by), (0, 0, 255), 2)
            draw_label(cv2, frame, f"Violence risk {whole_percent(score)}%",
                       (ax + bx) // 2, (ay + by) // 2, 0.5, 1, (0, 0, 255), chip=True)
            continue
        color = (0, 0, 255) if score is not None and score >= PAIR_RISK_THRESHOLD else (255, 255, 0)
        cv2.line(frame, (ax, ay), (bx, by), color, 2)
        dist = track.latest_features.get("normalized_center_distance", 0.0)
        label = f"P{track.id_a}<->P{track.id_b} d={dist:.1f} buf={track.frame_count}"
        if score is not None:
            label += f" s={score:.2f}"
        cv2.putText(frame, label, ((ax + bx) // 2, (ay + by) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_scene_panel(frame, verdict, graph, *, plain: bool = False) -> None:
    import cv2
    if plain:
        name = getattr(getattr(verdict, "scene", None), "value", None)
        if name in PLAIN_SCENE_NO_BANNER:
            return
        text, fill, ink = PLAIN_SCENE_BANNERS.get(name, PLAIN_SCENE_FALLBACK)
        frame_h, frame_w = frame.shape[:2]
        thickness = 2
        scale = _fit_scale(cv2, text, frame_w, thickness)
        width, height, baseline = _text_size(cv2, text, scale, thickness)
        x1 = max(0, (frame_w - width - 40) // 2)
        y1 = max(0, min(PLAIN_SCENE_BANNER_TOP, frame_h - height - baseline - 30))
        x2 = min(frame_w, x1 + width + 40)
        y2 = min(frame_h, y1 + height + baseline + 26)
        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.putText(frame, text, (x1 + 20, y2 - 13), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, ink, thickness)
        return
    from core.graph import overlay_lines
    y = frame.shape[0] - 20 - 18 * 8
    for line in overlay_lines(verdict, graph)[:9]:
        cv2.putText(frame, line[:80], (10, max(20, y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 18
