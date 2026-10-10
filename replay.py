"""Local-video replay: python replay.py --source data/videos/example.mp4.

Drives frames through Pipeline (models never open cameras themselves) and
prints per-frame inspection: tracked people, candidate pairs, pair buffers,
graph structure, scene candidate. --synthetic runs a scripted crowd without
any video file or ML weights. --demo-scores injects pair scores (as
"a-b:score") so graph reasoning can be inspected before a classifier exists.
No automatic web downloading. cv2 required only for real video / --viz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import VigilanceConfig
from core.graph import overlay_lines
from core.logging_utils import setup_logging
from core.pipeline import Pipeline
from core.tracked_person import TrackedPerson


class SyntheticCrowd:
    def __init__(self, script, h=240, w=320):
        self.script = list(script)
        self.h, self.w = h, w
        self.calls = 0

    def detect(self, frame, timestamp=0.0):
        from core.detector import DetectorOutput
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        persons = []
        for tid, x0 in self.script[idx]:
            kpts = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                kpts[i] = [x0 + (i % 5) * 3.0, 60.0 + i * 7.0, 0.9]
            kpts[5] = [x0 - 12.0, 90.0, 0.9]
            kpts[6] = [x0 + 12.0, 90.0, 0.9]
            kpts[11] = [x0 - 10.0, 170.0, 0.9]
            kpts[12] = [x0 + 10.0, 170.0, 0.9]
            persons.append(TrackedPerson(track_id=tid, bbox=[x0, 40.0, x0 + 60.0, 220.0],
                                         keypoints=kpts, detection_conf=0.9,
                                         timestamp=float(timestamp)))
        return DetectorOutput(persons=persons, pose_results=None, det_results=[])

    def reset(self):
        self.calls = 0

    def frame(self):
        return np.zeros((self.h, self.w, 3), dtype=np.uint8)


def parse_scores(spec: str):
    table = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        pair, val = item.split(":", 1)
        a, b = pair.split("-", 1)
        a, b = sorted((int(a), int(b)))
        table[(a, b)] = float(val)
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VIGLANCE replay inspector")
    ap.add_argument("--source", default=None, help="local MP4 / image / camera index")
    ap.add_argument("--synthetic", action="store_true", help="scripted crowd, no files needed")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--evidence", default=None, help="write JSONL evidence here")
    ap.add_argument("--viz", action="store_true", help="show debug visualization (needs display)")
    ap.add_argument("--demo-scores", default="", help='e.g. "1-2:0.92,1-3:0.88"')
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args(argv)

    cfg = VigilanceConfig.from_file(args.config) if args.config else VigilanceConfig()
    cfg.from_env()
    setup_logging(args.log_level or cfg.log_level)
    table = parse_scores(args.demo_scores)

    def demo_scorer(tracks):
        out = []
        for t in tracks:
            a, b = sorted((t.id_a, t.id_b))
            out.append(table.get((a, b), table.get((b, a), 0.0)))
        return out

    scorer = demo_scorer if table else None

    if args.synthetic:
        script = [[(1, 40.0), (2, 160.0)], [(1, 42.0), (2, 158.0), (3, 270.0)],
                  [(1, 44.0), (3, 268.0)], [(1, 46.0), (2, 154.0), (3, 266.0)]]
        fake = SyntheticCrowd(script * ((args.max_frames // len(script)) + 1))
        pipe = Pipeline(config=cfg, detector=fake, pair_scorer=scorer)
        fh = open(args.evidence, "w") if args.evidence else None
        n = min(args.max_frames, len(fake.script))
        for i in range(n):
            _inspect(pipe, fake.frame(), i / cfg.fps, i, fh, args.viz)
        if fh:
            fh.close()
            print(f"evidence -> {args.evidence}")
        return 0

    if not args.source:
        print("need --source or --synthetic")
        return 2
    try:
        import cv2
    except ImportError:
        print("opencv required for video replay: pip install opencv-python")
        return 2
    pipe = Pipeline(config=cfg, pair_scorer=scorer)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"cannot open {args.source}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps
    fh = open(args.evidence, "w") if args.evidence else None
    i = 0
    while i < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        _inspect(pipe, frame, i / fps, i, fh, args.viz)
        if args.viz:
            cv2.imshow("replay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1
    cap.release()
    if args.viz:
        cv2.destroyAllWindows()
    if fh:
        fh.close()
    return 0


def _inspect(pipe, frame, ts, idx, fh, viz):
    from core.visualization import draw_pairs, draw_scene_panel, draw_tracks
    res = pipe.process(frame, timestamp=ts)
    persons = {p.track_id: p for p in res.persons}
    graph = pipe.graph
    print(f"[f{idx} t={ts:.2f}] people={sorted(persons)} "
          f"pairs={len(res.pair_keys)} scene={res.scene.scene.value if res.scene else '?'}")
    for line in overlay_lines(res.scene, graph) if res.scene else []:
        print(f"    {line}")
    if fh:
        fh.write(json.dumps({"frame": idx, "timestamp": ts,
                             "people": sorted(persons),
                             "pairs": [list(k) for k in res.pair_keys],
                             "scene": res.scene.scene.value if res.scene else None,
                             "features": res.scene.features if res.scene else {}}) + "\n")
    if viz:
        draw_tracks(frame, res.persons)
        draw_pairs(frame, pipe.pair_manager.open_tracks(), persons, graph)
        if res.scene:
            draw_scene_panel(frame, res.scene, graph)


if __name__ == "__main__":
    raise SystemExit(main())
