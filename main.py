"""Thin CLI entrypoint. Visualization only -- all logic lives in core/scenarios."""

from __future__ import annotations

import argparse
import os

from config import VigilanceConfig
from core.logging_utils import get_logger, setup_logging
from core.pipeline import Pipeline

log = get_logger("main")


def build_config(args) -> VigilanceConfig:
    cfg = VigilanceConfig()
    if args.config:
        cfg = VigilanceConfig.from_file(args.config)
    cfg.from_env()
    cfg.apply_overrides(
        source=args.source,
        pose_weights=args.pose_weights,
        det_weights=args.det_weights,
        tracker=args.tracker,
        tracker_fallback=args.tracker_fallback,
        fps=args.fps,
        imgsz=args.imgsz,
        device=args.device,
        max_people=args.max_people,
        lost_timeout_sec=args.lost_timeout,
        lost_grace_sec=args.lost_grace,
        evidence_dir=args.evidence_dir,
        log_level=args.log_level,
    )
    if args.no_fall:
        cfg.enable_fall = False
    if args.no_bag:
        cfg.enable_bag = False
    if args.no_fight:
        cfg.enable_fight = False
    if args.legacy_fight:
        cfg.enable_legacy_fight = True
    cfg.validate()
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VIGLANCE - Local AI Surveillance (foundation build)")
    p.add_argument("--source", default=None, help="Video path, image path, or 0 for webcam (default: config file value, else webcam)")
    p.add_argument("--config", default=None, help="JSON (or YAML, if pyyaml installed) config file")
    p.add_argument("--pose-weights", default=None)
    p.add_argument("--det-weights", default=None)
    p.add_argument("--tracker", default=None)
    p.add_argument("--tracker-fallback", default=None)
    p.add_argument("--max-people", type=int, default=None)
    p.add_argument("--lost-timeout", type=float, default=None)
    p.add_argument("--lost-grace", type=float, default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--evidence-dir", default=None)
    p.add_argument("--log-level", default=None)
    p.add_argument("--no-fall", action="store_true")
    p.add_argument("--no-bag", action="store_true")
    p.add_argument("--no-fight", action="store_true")
    p.add_argument("--legacy-fight", action="store_true",
                   help="Opt in to the legacy BiLSTM fight module (OFF by default; ST-GCN++ is the fight path)")
    p.add_argument("--fight-checkpoint", default=None,
                   help="Post-training plug-in: ST-GCN++ fight checkpoint (.pth). "
                        "Absent model stays usable (graph structure only).")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (replay/smoke)")
    return p.parse_args(argv)


def draw_overlays(frame, persons, alerts) -> None:
    from core.visualization import draw_alerts, draw_tracks
    draw_tracks(frame, persons)
    draw_alerts(frame, alerts)


def run_video(cfg: VigilanceConfig, pipeline: Pipeline, max_frames=None) -> None:
    import cv2
    source: object = int(cfg.source) if str(cfg.source).isdigit() else cfg.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {cfg.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        cfg.fps = float(fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info("event=stream_start source=%s fps=%.2f", cfg.source, cfg.fps)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process(frame, timestamp=idx / float(cfg.fps))
            for name, data in result.alerts.items():
                log.warning("event=alert scenario=%s ts=%.3f data=%s", name, result.timestamp, data)
            draw_overlays(frame, result.persons, result.alerts)
            cv2.imshow("VIGLANCE - Live Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    log.info("event=stream_end frames=%d diagnostics=%s", idx, pipeline.track_diagnostics.summary())


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    setup_logging(args.log_level or cfg.log_level)
    pipeline = Pipeline(config=cfg)
    if args.fight_checkpoint:
        loaded = pipeline.load_fight_model(args.fight_checkpoint)
        if loaded is None:
            log.warning("event=no_fight_model (continuing without scores)")
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if isinstance(cfg.source, str) and cfg.source.lower().endswith(image_exts):
        import cv2
        if not os.path.exists(cfg.source):
            log.error("event=image_missing path=%s", cfg.source)
            return 2
        frame = cv2.imread(cfg.source)
        if frame is None:
            log.error("event=image_unreadable path=%s", cfg.source)
            return 2
        result = pipeline.process(frame, timestamp=0.0)
        for name, data in result.alerts.items():
            log.warning("event=alert scenario=%s data=%s", name, data)
        draw_overlays(frame, result.persons, result.alerts)
        cv2.imshow("VIGLANCE - Image View", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0
    run_video(cfg, pipeline, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
