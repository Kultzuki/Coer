"""VIGLANCE web UI: live MJPEG + status dashboard with zero web dependencies.

Usage:
    python ui.py --source 0                      # webcam + real YOLO detector
    python ui.py --source video.mp4              # MP4/AVI/MOV/etc. + detector
    python ui.py --config config/default.json     # default trained fight model
    python ui.py --synthetic                     # scripted demo, no camera/weights
    python ui.py --source 0 --fight-checkpoint models/stgcnpp/best_fight_stgcnpp_v1.pth

Then open http://localhost:8000 in a browser.

Endpoints: / (dashboard), /stream.mjpg (MJPEG), /api/status (JSON),
/snapshot.jpg (single frame), /api/upload (multipart video), and
/api/source (webcam/source switching). OpenCV handles MP4, AVI, MOV, MKV,
WebM, MPEG, and other locally supported codecs. The auxiliary YOLO threat
model (`xDevine/yolo11s-multiclass-surveillance-detection`) downloads from
Hugging Face on first use unless disabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from config import VigilanceConfig
from core.logging_utils import get_logger, setup_logging
from core.pipeline import Pipeline
from core.stgcn_adapter import is_buffer_ready
from core.video_upload import (MAX_UPLOAD_BYTES, VideoUploadError,
                               save_video_upload)

log = get_logger("ui")

#: Uploaded/file videos play at most this many times, then stop on an
#: "ended" status instead of looping forever. Webcams (digit sources) and
#: the synthetic demo are unaffected.
MAX_VIDEO_PLAYS = 3

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "opencv-python is required for the UI (capture + JPEG encoding)."
    ) from exc


# --------------------------------------------------------------------------
# shared live state
# --------------------------------------------------------------------------

class LiveState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.status: dict = {"running": False, "timeline": []}
        self.stop = False
        self.source_request = None
        self.source_generation = 0
        self.timeline = []
        self.active_event = None

    def request_source(self, source: str) -> None:
        with self.lock:
            self.source_generation += 1
            self.source_request = (self.source_generation, str(source))

    def reset_timeline(self) -> None:
        with self.lock:
            self.timeline.clear()
            self.active_event = None

    def record_event(self, timestamp: float, scene: str | None,
                     alerts: dict) -> None:
        active = (scene is not None and scene != "NO_FIGHT") or bool(alerts)
        with self.lock:
            if not active:
                self.active_event = None
                return
            alert_names = sorted(alerts)
            if self.active_event is None:
                self.active_event = {
                    "start": round(float(timestamp), 3),
                    "end": round(float(timestamp), 3),
                    "scene": scene or "THREAT",
                    "alerts": alert_names,
                }
                self.timeline.append(self.active_event)
                return
            self.active_event["end"] = round(float(timestamp), 3)
            if scene is not None and scene != "NO_FIGHT":
                self.active_event["scene"] = scene
            self.active_event["alerts"] = sorted(
                set(self.active_event["alerts"]) | set(alert_names)
            )

    def timeline_snapshot(self) -> list:
        with self.lock:
            return [dict(event) for event in self.timeline]


# --------------------------------------------------------------------------
# frame pump
# --------------------------------------------------------------------------

def _model_status(pipeline: Pipeline) -> dict:
    adapter = pipeline.pair_adapter
    ckpt = getattr(adapter, "checkpoint", None)
    return {
        "ready": bool(adapter.is_ready),
        "head_trained": bool(getattr(adapter, "head_trained", False)),
        "checkpoint": getattr(ckpt, "path", None) or None,
    }


def _alert_details(result_alerts, threat_alerts) -> dict:
    """Name -> {confidence, message} for every live alert (sidebar data)."""
    details = {}
    for source in (result_alerts or {}, threat_alerts or {}):
        for name, data in source.items():
            if not isinstance(data, dict):
                continue
            try:
                conf = data.get("confidence")
                conf = None if conf is None else round(float(conf), 3)
            except (TypeError, ValueError):
                conf = None
            details[str(name)] = {
                "confidence": conf,
                "message": str(data.get("message", "")),
            }
    return details


def _status_from_result(pipeline: Pipeline, result, fps: float, source: str,
                        threat_frame=None) -> dict:
    persons = [
        {"id": p.track_id, "gen": p.generation, "bbox": [round(float(v), 1) for v in p.bbox]}
        for p in result.persons
    ]
    pairs = []
    window = float(pipeline.config.fight_cfg.window_seconds)
    for t in pipeline.pair_manager.open_tracks():
        edge = pipeline.graph.edges.get(t.key)
        times = [float(e.get("t", 0.0)) for e in t.buffer] if t.buffer else []
        span = (max(times) - min(times)) if len(times) > 1 else 0.0
        pairs.append({
            "a": t.id_a, "b": t.id_b,
            "key": list(t.key),
            "score": (None if edge is None or edge.score is None else round(float(edge.score), 3)),
            "buf": t.frame_count,
            "span": round(span, 2),
            "ready": bool(is_buffer_ready(t, window)),
            "dist": round(float(t.latest_features.get("normalized_center_distance", 0.0)), 2),
        })
    scene = None
    if result.scene is not None:
        scene = {
            "verdict": result.scene.scene.value,
            "incidents": [
                {"members": inc.members,
                 "violent_edges": len(inc.violent_edges),
                 "candidate": inc.candidate.value}
                for inc in result.scene.incidents
            ],
        }
    threat_frame = threat_frame or {}
    threat_detections = threat_frame.get("detections", [])
    threat_alerts = threat_frame.get("alerts", {})
    return {
        "running": True,
        "source": str(source),
        "frame": result.frame_index,
        "ts": round(result.timestamp, 3),
        "fps": round(fps, 1),
        "people": persons,
        "pairs": pairs,
        "scene": scene,
        "model": _model_status(pipeline),
        "alerts": sorted(set(result.alerts) | set(threat_alerts)),
        "alert_details": _alert_details(result.alerts, threat_alerts),
        "threats": threat_detections,
        "threat_models": threat_frame.get("models", {}),
        "diagnostics": result.diagnostics,
    }


def _draw_model_banner(frame, pipeline: Pipeline) -> None:
    ms = _model_status(pipeline)
    if ms["ready"] and ms["head_trained"]:
        text, color = "Fight detection: active", (0, 200, 0)
    elif ms["ready"]:
        text, color = "Fight detection: not trained - results may be unreliable", (0, 200, 255)
    else:
        text, color = "Fight detection: not installed - movement only", (150, 150, 150)
    cv2.putText(frame, text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def open_capture(source):
    """Open a video source, trying alternate Windows backends for webcams.

    Returns (cap, backend_label) on success, else (None, error_text).
    A successful open is verified with a real frame grab: some laptop
    cameras report "opened" yet deliver nothing usable under the default
    backend, which previously surfaced as garbage frames downstream.
    """
    is_cam = str(source).isdigit()
    src: object = int(source) if is_cam else source
    backends = [None]
    if is_cam:
        backends += [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    tried = []
    for backend in backends:
        label = "default" if backend is None else str(backend)
        tried.append(label)
        try:
            cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
        except Exception as exc:
            log.warning("event=ui_capture_open_failed backend=%s error=%r", label, exc)
            continue
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok or frame is None or min(frame.shape[:2]) <= 0:
            cap.release()
            log.warning("event=ui_capture_no_frames backend=%s source=%r", label, source)
            continue
        if not is_cam:
            # The probe grab consumed frame 0; reopen so playback starts
            # at the beginning instead of silently dropping the first frame.
            cap.release()
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                continue
        log.info("event=ui_capture_open source=%r backend=%s size=%sx%s",
                 source, label, frame.shape[1], frame.shape[0])
        return cap, label
    return None, f"cannot open source {source!r} (tried backends: {', '.join(tried)}; " \
                 "close other apps using the camera and retry)"


def pump_frames(state: LiveState, pipeline: Pipeline, source, fps_fallback: float,
                synthetic_frames=None, threat_detector=None, notifier=None) -> None:
    from core.visualization import (draw_alerts, draw_pairs, draw_scene_panel,
                                    draw_threats, draw_tracks)

    cap = None
    current_source = str(source)
    current_generation = 0
    fps = float(fps_fallback)
    idx, last, ema = 0, time.perf_counter(), None
    plays, ended = 0, False
    try:
        while not state.stop:
            if synthetic_frames is None:
                with state.lock:
                    requested = state.source_request
                if requested is not None and requested[0] != current_generation:
                    current_generation, current_source = requested
                    if cap is not None:
                        cap.release()
                    cap = None
                    pipeline.reset()
                    if threat_detector is not None:
                        threat_detector.reset()
                    state.reset_timeline()
                    idx, ema = 0, None
                    plays, ended = 0, False

                if cap is None and not ended:
                    cap, open_error = open_capture(current_source)
                    if cap is None:
                        with state.lock:
                            state.status = {
                                "running": False,
                                "source": current_source,
                                "error": open_error,
                            }
                        time.sleep(0.1)
                        continue
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_fallback)
                    pipeline.reset()
                    if threat_detector is not None:
                        threat_detector.reset()
                if cap is None:
                    time.sleep(0.1)
                    continue
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    if str(current_source).isdigit():
                        timeline = state.timeline_snapshot()
                        with state.lock:
                            state.status = {
                                "running": False,
                                "source": current_source,
                                "timeline": timeline,
                            }
                        time.sleep(0.1)
                        continue
                    plays += 1
                    if plays >= MAX_VIDEO_PLAYS:
                        ended = True
                        timeline = state.timeline_snapshot()
                        with state.lock:
                            state.status = {
                                "running": False,
                                "source": current_source,
                                "timeline": timeline,
                                "ended": True,
                            }
                        log.info("event=ui_replay_finished source=%s plays=%d",
                                 current_source, plays)
                    time.sleep(0.1)
                    continue
            else:
                frame = synthetic_frames[idx % len(synthetic_frames)].copy()

            ts = idx / float(fps or fps_fallback)
            try:
                result = pipeline.process(frame, timestamp=ts)
            except Exception as exc:
                log.error("event=ui_frame_error frame=%d error=%r", idx, exc)
                idx += 1
                continue

            threat_data = {"detections": [], "alerts": {}, "models": {}}
            threat_detections = []
            if threat_detector is not None:
                threat_frame = threat_detector.detect(frame, result.persons, ts)
                threat_detections = threat_frame.detections
                threat_data = {
                    "detections": [d.to_dict() for d in threat_frame.detections],
                    "alerts": threat_frame.alerts,
                    "models": threat_frame.models,
                }
            alerts = dict(result.alerts)
            alerts.update(threat_data["alerts"])
            scene_name = result.scene.scene.value if result.scene is not None else None
            state.record_event(ts, scene_name, alerts)
            persons_by_id = {p.track_id: p for p in result.persons}
            draw_tracks(frame, result.persons, plain=True)
            draw_threats(frame, threat_detections, plain=True)
            draw_pairs(frame, pipeline.pair_manager.open_tracks(), persons_by_id,
                       pipeline.graph, plain=True)
            if result.scene is not None:
                draw_scene_panel(frame, result.scene, pipeline.graph, plain=True)
            draw_alerts(frame, alerts, plain=True)
            _draw_model_banner(frame, pipeline)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                idx += 1
                continue
            if notifier is not None:
                from core.alert_dispatch import incident_from_state
                notifier.maybe_notify(
                    incident_from_state(scene_name, alerts, ts), bytes(buf))
            now = time.perf_counter()
            ema = (now - last) if ema is None else 0.9 * ema + 0.1 * (now - last)
            last = now
            timeline = state.timeline_snapshot()
            with state.lock:
                state.jpeg = bytes(buf)
                state.status = _status_from_result(
                    pipeline, result, (1.0 / ema) if ema else 0.0,
                    current_source, dict(threat_data, alerts=alerts))
                state.status["timeline"] = timeline
                state.status["running"] = True
            idx += 1
    finally:
        if cap is not None:
            cap.release()
        with state.lock:
            state.status = dict(state.status, running=False)
        log.info("event=ui_pump_end frames=%d", idx)


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RakshaNetra — VIGLANCE AI Surveillance System</title>
<style>
:root {
  --bg-dark: #0a0f1d;
  --bg-sidebar: #090e1a;
  --bg-panel: #111828;
  --bg-card: #131c2e;
  --border-color: #1e293f;
  --border-subtle: #172133;
  --text-main: #f1f5f9;
  --text-muted: #7d8ea5;
  --text-sub: #94a3b8;
  --accent-cyan: #38bdf8;
  --accent-cyan-glow: rgba(56, 189, 248, 0.25);
  --accent-blue: #0284c7;
  --alert-red: #ef4444;
  --alert-red-dark: #991b1b;
  --alert-red-bg: rgba(239, 68, 68, 0.16);
  --alert-red-pill: rgba(220, 38, 38, 0.45);
  --status-green: #10b981;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg-dark);
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  height: 100vh;
  display: flex;
  overflow: hidden;
  font-size: 13px;
}

/* Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

/* LEFT SIDEBAR */
#sidebar {
  width: 220px;
  min-width: 220px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  padding: 16px 14px;
  user-select: none;
}
.brand-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 18px;
}
.brand-logo-icon {
  width: 46px;
  height: 46px;
  margin-bottom: 8px;
  filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.4));
}
.brand-title {
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.01em;
  color: #fff;
  line-height: 1.1;
}
.brand-title span { color: var(--accent-cyan); }
.brand-subtitle {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 4px;
  font-weight: 500;
  letter-spacing: 0.02em;
}

.nav-menu {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: 8px;
  color: var(--text-muted);
  text-decoration: none;
  font-weight: 600;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s ease;
  border: 1px solid transparent;
}
.nav-item:hover {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-main);
}
.nav-item.active {
  background: rgba(14, 165, 233, 0.14);
  color: var(--accent-cyan);
  border-color: rgba(56, 189, 248, 0.3);
  box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);
}
.nav-item svg {
  width: 17px;
  height: 17px;
  flex-shrink: 0;
  stroke: currentColor;
}

/* MAIN LAYOUT WRAPPER */
#main-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  min-width: 0;
  overflow: hidden;
}

/* TOP STATUS BAR */
#topbar {
  height: 52px;
  background: #0d1424;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  flex-shrink: 0;
}
.top-metrics {
  display: flex;
  align-items: center;
  gap: 20px;
  font-size: 13px;
  font-weight: 600;
}
.metric-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #cbd5e1;
}
.metric-divider {
  width: 1px;
  height: 16px;
  background: var(--border-color);
}
.status-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: var(--status-green);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--status-green);
}
.alert-pill-badge {
  background: var(--alert-red);
  color: #fff;
  font-size: 11px;
  font-weight: 800;
  padding: 1px 7px;
  border-radius: 10px;
  min-width: 18px;
  text-align: center;
}
.top-user {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
}
.user-avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #1e293b;
  border: 1px solid #38bdf8;
  display: flex;
  align-items: center;
  justify-content: center;
}
.user-name {
  font-weight: 600;
  color: #e2e8f0;
}

/* CONTENT CONTAINER */
#content-area {
  flex: 1;
  display: flex;
  min-height: 0;
  overflow: hidden;
}

/* CENTER DASHBOARD AREA */
#center-viewport {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 14px 16px;
  gap: 12px;
  overflow-y: auto;
  min-width: 0;
}

/* CAMERA 2x2 GRID */
.camera-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 12px;
  flex: 1;
  min-height: 480px;
}
.camera-card {
  background: #080d17;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
  transition: border-color 0.2s;
}
.camera-card:hover {
  border-color: #2e4162;
}
.camera-card.highlight {
  border-color: var(--accent-cyan);
  box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
}
.camera-header {
  height: 28px;
  background: rgba(13, 20, 36, 0.95);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 10px;
  font-size: 11.5px;
  font-weight: 600;
  color: #cbd5e1;
  border-bottom: 1px solid var(--border-subtle);
  flex-shrink: 0;
}
.camera-header-left {
  display: flex;
  align-items: center;
  gap: 6px;
}
.camera-time {
  color: var(--text-muted);
  font-weight: 500;
  font-size: 11px;
}

.camera-feed-box {
  flex: 1;
  position: relative;
  background: #000;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
.camera-feed-box img#liveStream {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* Overlay alert tags inside camera views */
.camera-overlay-alerts {
  position: absolute;
  top: 8px;
  right: 8px;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 5px;
  z-index: 5;
  pointer-events: none;
}
.cam-alert-tag {
  background: rgba(185, 28, 28, 0.85);
  border: 1px solid rgba(239, 68, 68, 0.9);
  color: #fee2e2;
  font-size: 10.5px;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 5px;
  backdrop-filter: blur(4px);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
}
.cam-alert-tag svg {
  width: 12px;
  height: 12px;
  fill: currentColor;
}

/* Camera Standby Placeholder View */
.cam-placeholder {
  width: 100%;
  height: 100%;
  background: radial-gradient(circle at center, #0c1629 0%, #060a13 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  border: 1px dashed rgba(56, 189, 248, 0.15);
}
.cctv-grid-overlay {
  position: absolute;
  inset: 0;
  background-image: linear-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
  background-size: 24px 24px;
  pointer-events: none;
}
.cctv-reticle {
  position: absolute;
  inset: 16px;
  border: 1px solid rgba(255, 255, 255, 0.04);
  pointer-events: none;
}
.cctv-reticle::before, .cctv-reticle::after {
  content: "";
  position: absolute;
  width: 10px;
  height: 10px;
  border-color: rgba(56, 189, 248, 0.3);
  border-style: solid;
}
.cctv-reticle::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
.cctv-reticle::after { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

.cam-slot-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  background: rgba(14, 165, 233, 0.1);
  border: 1px solid rgba(56, 189, 248, 0.35);
  color: var(--accent-cyan);
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 11.5px;
  font-weight: 700;
  cursor: pointer;
  z-index: 2;
  transition: all 0.15s;
}
.cam-slot-btn:hover {
  background: rgba(14, 165, 233, 0.25);
  border-color: var(--accent-cyan);
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
}
.cam-fullscreen-btn {
  position: absolute;
  right: 8px;
  bottom: 8px;
  z-index: 6;
  display: flex;
  align-items: center;
  background: rgba(13, 20, 36, 0.8);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  border-radius: 6px;
  padding: 6px;
  cursor: pointer;
}
.cam-fullscreen-btn:hover {
  color: var(--accent-cyan);
  border-color: var(--accent-cyan);
}
.cam-slot-label {
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 6px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* BOTTOM ANALYTICS ROW */
.bottom-analytics-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  height: 130px;
  min-height: 130px;
}
.analytics-card {
  background: #0d1424;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px 14px;
  display: flex;
  flex-direction: column;
}
.card-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.card-title {
  font-size: 13px;
  font-weight: 700;
  color: #f1f5f9;
}
.card-more-btn {
  color: var(--text-muted);
  background: transparent;
  border: 0;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
}

/* Heatmap Grid */
.heatmap-container {
  flex: 1;
  display: grid;
  grid-template-columns: repeat(18, 1fr);
  grid-template-rows: repeat(4, 1fr);
  gap: 3px;
  align-items: center;
}
.heatmap-cell {
  border-radius: 2px;
  width: 100%;
  height: 100%;
  min-height: 12px;
  transition: transform 0.1s;
}
.heatmap-cell:hover {
  transform: scale(1.15);
  z-index: 2;
}

/* SVG Line Chart */
.chart-container {
  flex: 1;
  position: relative;
  overflow: hidden;
}
.chart-container svg {
  width: 100%;
  height: 100%;
}

/* RIGHT SIDEBAR - ACTIVE ALERTS */
#alerts-sidebar {
  width: 320px;
  min-width: 320px;
  background: var(--bg-sidebar);
  border-left: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  padding: 16px 14px;
  overflow-y: auto;
  gap: 12px;
}
.alerts-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border-subtle);
}
.alerts-header-title {
  font-size: 15px;
  font-weight: 800;
  color: #fff;
}
.alerts-clock {
  font-size: 12px;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 5px;
  font-variant-numeric: tabular-nums;
}

.alerts-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
}

/* Alert Item Card */
.alert-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-left: 4px solid var(--alert-red);
  border-radius: 6px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: transform 0.15s, border-color 0.15s;
}
.alert-card:hover {
  border-color: #334460;
  border-left-color: var(--alert-red);
}
.alert-top {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.alert-title {
  font-size: 13.5px;
  font-weight: 700;
  color: #f8fafc;
}
.alert-cam {
  font-size: 11.5px;
  color: var(--text-muted);
  font-weight: 500;
}
.alert-confidence {
  font-size: 12px;
  color: #e2e8f0;
  font-weight: 600;
}
.alert-confidence span {
  color: var(--accent-cyan);
}
.alert-reasoning {
  font-size: 11.5px;
  color: var(--text-sub);
  line-height: 1.35;
}
.alert-actions {
  display: flex;
  gap: 8px;
  margin-top: 6px;
}
.btn-view-live {
  flex: 1;
  background: #0ea5e9;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 11.5px;
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-view-live:hover {
  background: #38bdf8;
}
.btn-dismiss {
  flex: 1;
  background: transparent;
  color: #f87171;
  border: 1px solid rgba(239, 68, 68, 0.4);
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 11.5px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.15s;
}
.btn-dismiss:hover {
  background: rgba(239, 68, 68, 0.15);
  border-color: var(--alert-red);
}

/* SOURCE CONFIG MODAL */
#sourceModal {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(4px);
  z-index: 100;
  align-items: center;
  justify-content: center;
}
.modal-content {
  background: #0f172a;
  border: 1px solid var(--border-color);
  border-radius: 10px;
  width: 440px;
  max-width: 90vw;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
}
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.modal-header h3 {
  font-size: 16px;
  color: #fff;
  font-weight: 700;
}
.modal-close-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 20px;
  cursor: pointer;
}
.modal-tabs {
  display: flex;
  gap: 8px;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 8px;
}
.modal-tab {
  background: transparent;
  border: none;
  color: var(--text-muted);
  padding: 6px 12px;
  font-weight: 600;
  font-size: 12.5px;
  cursor: pointer;
  border-radius: 4px;
}
.modal-tab.active {
  background: rgba(56, 189, 248, 0.15);
  color: var(--accent-cyan);
}
.modal-form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.modal-input {
  width: 100%;
  background: #090e1a;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px 12px;
  color: #fff;
  font-size: 13px;
}
.modal-submit-btn {
  background: var(--accent-cyan);
  color: #041426;
  border: none;
  padding: 10px 16px;
  border-radius: 6px;
  font-weight: 800;
  cursor: pointer;
  font-size: 13px;
  transition: opacity 0.15s;
}
.modal-submit-btn:hover { opacity: 0.9; }
#modalMessage {
  font-size: 12px;
  color: var(--accent-cyan);
  min-height: 18px;
}

/* TECH DRAWER */
#techDrawer {
  display: none;
  position: fixed;
  right: 0;
  top: 52px;
  bottom: 0;
  width: 420px;
  background: #090e1a;
  border-left: 1px solid var(--border-color);
  z-index: 90;
  padding: 18px 20px;
  overflow-y: auto;
  flex-direction: column;
  gap: 14px;
  box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
}
#techDrawer.open { display: flex; }
.drawer-title { font-size: 14px; font-weight: 700; color: #fff; display: flex; justify-content: space-between; }
.code-box {
  background: #060a12;
  border: 1px solid #1a2538;
  border-radius: 6px;
  padding: 10px 12px;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 11px;
  color: #93c5fd;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
</head>
<body>

<!-- LEFT NAVIGATION SIDEBAR -->
<aside id="sidebar">
  <div class="brand-box">
    <!-- RakshaNetra Eye-Shield Logo -->
    <svg class="brand-logo-icon" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M50 8L86 24V50C86 73 70.8 90 50 96C29.2 90 14 73 14 50V24L50 8Z" stroke="#38bdf8" stroke-width="4.5" fill="rgba(14, 165, 233, 0.12)"/>
      <ellipse cx="50" cy="50" rx="26" ry="17" stroke="#38bdf8" stroke-width="3.5" fill="none"/>
      <circle cx="50" cy="50" r="10" fill="#38bdf8"/>
      <circle cx="52" cy="48" r="3.5" fill="#ffffff"/>
    </svg>
    <div class="brand-title">Raksha<span>Netra</span></div>
    <div class="brand-subtitle">AI - सहायक सुरक्षा सिस्टम</div>
  </div>

  <nav class="nav-menu">
    <div class="nav-item active" id="navDashboard">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="7" height="7"></rect>
        <rect x="14" y="3" width="7" height="7"></rect>
        <rect x="14" y="14" width="7" height="7"></rect>
        <rect x="3" y="14" width="7" height="7"></rect>
      </svg>
      Dashboard
    </div>
    <div class="nav-item" id="navLive" onclick="openSourceModal('CAM-01')">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="23 7 16 12 23 17 23 7"></polygon>
        <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
      </svg>
      Live Monitoring
    </div>
    <div class="nav-item" onclick="toggleTechDrawer()">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <polyline points="12 6 12 12 16 14"></polyline>
      </svg>
      Incident History
    </div>
    <div class="nav-item" onclick="toggleTechDrawer()">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"></line>
        <line x1="12" y1="20" x2="12" y2="4"></line>
        <line x1="6" y1="20" x2="6" y2="14"></line>
      </svg>
      Analytics
    </div>
    <div class="nav-item" onclick="toggleTechDrawer()">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
      </svg>
      Camera Health
    </div>
    <div class="nav-item" onclick="toggleTechDrawer()">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
      </svg>
      Settings
    </div>
  </nav>
</aside>

<!-- MAIN APP AREA -->
<div id="main-wrapper">
  <!-- TOP STATUS BAR -->
  <header id="topbar">
    <div class="top-metrics">
      <div class="metric-item">
        <span class="status-indicator"></span>
        <span>System Active</span>
      </div>
      <div class="metric-divider"></div>
      <div class="metric-item">
        <span>Total Cameras Online: <strong id="camCount">—</strong></span>
      </div>
      <div class="metric-divider"></div>
      <div class="metric-item">
        <span>Active Alerts:</span>
        <span class="alert-pill-badge" id="topAlertCount">0</span>
      </div>
      <div class="metric-divider"></div>
      <div class="metric-item">
        <span>Latency: <strong id="latencyVal">—</strong></span>
      </div>
    </div>

    <div class="top-user" onclick="toggleTechDrawer()" title="Click to view system diagnostics">
      <div class="user-avatar">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
          <circle cx="12" cy="7" r="4"></circle>
        </svg>
      </div>
      <span class="user-name">Admin</span>
      <span style="font-size:10px; color:#64748b;">&#9662;</span>
    </div>
  </header>

  <!-- CONTENT -->
  <div id="content-area">
    <!-- CENTER VIEWPORT -->
    <main id="center-viewport">
      <!-- 2x2 CAMERA GRID -->
      <div class="camera-grid">
        <!-- CAM-01 (ACTIVE LIVE FEED) -->
        <div class="camera-card highlight" id="cardCam1">
          <div class="camera-header">
            <div class="camera-header-left">
              <span>CAM-01 Main Gate - <span class="live-clock">10:45 AM</span></span>
            </div>
            <span class="camera-time live-clock">10:45 AM</span>
          </div>
          <div class="camera-feed-box" onclick="openSourceModal('CAM-01')">
            <!-- Real Live Stream Feed from VIGLANCE Backend -->
            <img id="liveStream" src="/stream.mjpg" alt="CAM-01 Live Feed">
            <button class="cam-fullscreen-btn" onclick="toggleFullscreen(event, 'cardCam1')" title="Fullscreen">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
            </button>
            <!-- Overlaid Alert Pills (shown only during live backend events) -->
            <div class="camera-overlay-alerts" id="camOverlayAlerts"></div>
          </div>
        </div>

        <!-- CAM-02 (PARKING AREA) -->
        <div class="camera-card" id="cardCam2">
          <div class="camera-header">
            <div class="camera-header-left">
              <span>CAM-02 Parking Area - <span class="live-clock">10:45 AM</span></span>
            </div>
            <span class="camera-time live-clock">10:45 AM</span>
          </div>
          <div class="camera-feed-box">
            <div class="cam-placeholder">
              <div class="cctv-grid-overlay"></div>
              <div class="cctv-reticle"></div>
              <button class="cam-slot-btn" onclick="openSourceModal('CAM-02')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                + Add Camera Source / Feed
              </button>
              <span class="cam-slot-label">CAM-02 Standby / Connect Stream</span>
            </div>
          </div>
        </div>

        <!-- CAM-03 (RESIDENTIAL LANE) -->
        <div class="camera-card" id="cardCam3">
          <div class="camera-header">
            <div class="camera-header-left">
              <span>CAM-03 Residential Lane - Lane</span>
            </div>
            <span class="camera-time live-clock">10:45 AM</span>
          </div>
          <div class="camera-feed-box">
            <div class="cam-placeholder">
              <div class="cctv-grid-overlay"></div>
              <div class="cctv-reticle"></div>
              <button class="cam-slot-btn" onclick="openSourceModal('CAM-03')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                + Add Camera Source / Feed
              </button>
              <span class="cam-slot-label">CAM-03 Standby / Connect Stream</span>
            </div>
          </div>
        </div>

        <!-- CAM-04 (LOBBY / COMMUNITY HALL) -->
        <div class="camera-card" id="cardCam4">
          <div class="camera-header">
            <div class="camera-header-left">
              <span>CAM-04 Lobby / Community Hall</span>
            </div>
            <span class="camera-time live-clock">10:45 AM</span>
          </div>
          <div class="camera-feed-box">
            <div class="cam-placeholder">
              <div class="cctv-grid-overlay"></div>
              <div class="cctv-reticle"></div>
              <button class="cam-slot-btn" onclick="openSourceModal('CAM-04')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                + Add Camera Source / Feed
              </button>
              <span class="cam-slot-label">CAM-04 Standby / Connect Stream</span>
            </div>
          </div>
        </div>
      </div>

      <!-- BOTTOM ANALYTICS ROW -->
      <div class="bottom-analytics-row">
        <!-- INCIDENT ACTIVITY HEATMAP -->
        <div class="analytics-card">
          <div class="card-header-bar">
            <span class="card-title">Incident Activity</span>
            <button class="card-more-btn" onclick="toggleTechDrawer()">•••</button>
          </div>
          <div class="heatmap-container" id="heatmapGrid"></div>
        </div>

        <!-- AI CONFIDENCE TRENDLINE -->
        <div class="analytics-card">
          <div class="card-header-bar">
            <span class="card-title">AI Confidence Trend</span>
            <button class="card-more-btn" onclick="toggleTechDrawer()">•••</button>
          </div>
          <div class="chart-container">
            <svg viewBox="0 0 500 90" preserveAspectRatio="none">
              <defs>
                <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
                  <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
                </linearGradient>
              </defs>
              <!-- Grid guide lines -->
              <line x1="0" y1="20" x2="500" y2="20" stroke="#172238" stroke-dasharray="3 3"/>
              <line x1="0" y1="50" x2="500" y2="50" stroke="#172238" stroke-dasharray="3 3"/>
              <line x1="0" y1="80" x2="500" y2="80" stroke="#172238" stroke-dasharray="3 3"/>
              <!-- Area fill -->
              <path id="trendArea" d=""/>
              <!-- Live pair-score curve (was a static demo path) -->
              <path id="trendLine" d="" fill="none" stroke="#38bdf8" stroke-width="2.6" stroke-linecap="round"/>
            </svg>
          </div>
        </div>
      </div>
    </main>

    <!-- RIGHT SIDEBAR - ACTIVE ALERTS -->
    <aside id="alerts-sidebar">
      <div class="alerts-header">
        <span class="alerts-header-title">Active Alerts</span>
        <div class="alerts-clock">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          <span class="live-clock">10:45 AM</span>
        </div>
      </div>

      <!-- Rendered live from /api/status scene verdict + alert_details -->
      <div class="alerts-list" id="alertsList"></div>
    </aside>
  </div>
</div>

<!-- ADD CAMERA SOURCE / VIDEO MODAL -->
<div id="sourceModal">
  <div class="modal-content">
    <div class="modal-header">
      <h3 id="modalTitle">Configure Camera Source</h3>
      <button class="modal-close-btn" onclick="closeSourceModal()">&times;</button>
    </div>

    <div class="modal-tabs">
      <button class="modal-tab active" id="tabVideo" onclick="setModalTab('video')">Play Video File</button>
      <button class="modal-tab" id="tabWebcam" onclick="setModalTab('webcam')">Computer Webcam</button>
      <button class="modal-tab" id="tabRtsp" onclick="setModalTab('rtsp')">RTSP / Stream</button>
    </div>

    <!-- Video Upload Form -->
    <form id="uploadVideoForm" class="modal-form" onsubmit="handleUploadVideo(event)">
      <p style="font-size:12px; color:#94a3b8;">Upload or choose an MP4/AVI/MKV video to feed this surveillance channel:</p>
      <input type="file" id="videoFileInput" class="modal-input" accept="video/*,.mp4,.avi,.mov,.mkv,.webm" required>
      <button type="submit" class="modal-submit-btn">Play Video Feed</button>
    </form>

    <!-- Webcam Form -->
    <form id="webcamForm" class="modal-form" style="display:none;" onsubmit="handleSwitchWebcam(event)">
      <p style="font-size:12px; color:#94a3b8;">Connect to a local hardware webcam index:</p>
      <input type="number" id="webcamIndex" class="modal-input" value="0" min="0" placeholder="Camera Index (default 0)">
      <button type="submit" class="modal-submit-btn">Connect Webcam</button>
    </form>

    <!-- RTSP Form -->
    <form id="rtspForm" class="modal-form" style="display:none;" onsubmit="handleRtspStream(event)">
      <p style="font-size:12px; color:#94a3b8;">Enter RTSP / HTTP video stream URL:</p>
      <input type="text" id="rtspUrl" class="modal-input" placeholder="rtsp://admin:password@192.168.1.100:554/stream1">
      <button type="submit" class="modal-submit-btn">Connect RTSP Stream</button>
    </form>

    <div id="modalMessage"></div>
  </div>
</div>

<!-- TECHNICAL DIAGNOSTICS SLIDE-OVER DRAWER -->
<div id="techDrawer">
  <div class="drawer-title">
    <span>Technical Diagnostics</span>
    <button style="background:none; border:none; color:#94a3b8; cursor:pointer;" onclick="toggleTechDrawer()">&times;</button>
  </div>
  <div>
    <div style="font-size:11px; color:#64748b; margin-bottom:4px; text-transform:uppercase; font-weight:700;">Live Stream Status</div>
    <div class="code-box" id="techMeta">Connecting...</div>
  </div>
  <div>
    <div style="font-size:11px; color:#64748b; margin-bottom:4px; text-transform:uppercase; font-weight:700;">AI Models & Checkpoint</div>
    <div class="code-box" id="techModels">Loading models...</div>
  </div>
  <div>
    <div style="font-size:11px; color:#64748b; margin-bottom:4px; text-transform:uppercase; font-weight:700;">Tracked Persons & Pairs</div>
    <div class="code-box" id="techPairs">No open pairs</div>
  </div>
  <div>
    <div style="font-size:11px; color:#64748b; margin-bottom:4px; text-transform:uppercase; font-weight:700;">Pipeline Diagnostics</div>
    <div class="code-box" id="techDiag">{}</div>
  </div>
</div>

<script>
// Live Clocks
function updateClocks() {
  const now = new Date();
  let hours = now.getHours();
  const minutes = String(now.getMinutes()).padStart(2, "0");
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12 || 12;
  const timeStr = `${hours}:${minutes} ${ampm}`;
  document.querySelectorAll(".live-clock").forEach(el => el.textContent = timeStr);
}
setInterval(updateClocks, 1000);
updateClocks();

// Render Heatmap Matrix
const heatmapGrid = document.getElementById("heatmapGrid");
const heatColors = [
  "#141f33", "#141f33", "#141f33", "#0e7490", "#0891b2",
  "#06b6d4", "#f59e0b", "#f97316", "#ef4444", "#f43f5e"
];
// Matrix pattern resembling the density heatmap in reference image
const heatPattern = [
  [0, 1, 2, 0, 1, 3, 5, 8, 7, 6, 2, 1, 0, 0, 1, 0, 0, 0],
  [1, 1, 2, 2, 3, 5, 7, 9, 8, 5, 3, 1, 1, 0, 0, 1, 0, 0],
  [0, 2, 3, 4, 6, 7, 9, 8, 7, 4, 2, 1, 0, 1, 0, 0, 0, 0],
  [0, 1, 1, 2, 3, 4, 6, 8, 6, 4, 1, 0, 0, 0, 0, 1, 0, 0]
];
for (let r = 0; r < 4; r++) {
  for (let c = 0; c < 18; c++) {
    const val = heatPattern[r][c];
    const cell = document.createElement("div");
    cell.className = "heatmap-cell";
    cell.style.backgroundColor = heatColors[val];
    cell.title = `Slot ${r+1}:${c+1} — Incidents: ${val}`;
    heatmapGrid.appendChild(cell);
  }
}

// Active Alert Counts
function updateAlertCounter() {
  const count = document.querySelectorAll(".alert-card:not([style*='display: none'])").length;
  document.getElementById("topAlertCount").textContent = count;
}

// Dismissed alert cards stay dismissed until replaced by newer backend state
const dismissedAlerts = new Set();
function dismissAlert(id) {
  dismissedAlerts.add(id);
  const card = document.getElementById(id);
  if (card) {
    card.style.display = "none";
    updateAlertCounter();
  }
}
function alertCardId(kind, name) {
  return "dyn-" + kind + "-" + String(name).toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function focusCamera(cardId) {
  document.querySelectorAll(".camera-card").forEach(c => c.classList.remove("highlight"));
  const target = document.getElementById(cardId);
  if (target) {
    target.classList.add("highlight");
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function toggleFullscreen(e, cardId) {
  if (e) e.stopPropagation();
  const target = document.getElementById(cardId);
  if (!target) return;
  const active = document.fullscreenElement || document.webkitFullscreenElement;
  if (active) {
    if (document.exitFullscreen) document.exitFullscreen();
    else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
  } else if (target.requestFullscreen) {
    target.requestFullscreen().catch(() => {});
  } else if (target.webkitRequestFullscreen) {
    target.webkitRequestFullscreen();
  }
}

// Modal Management
let targetSlot = "CAM-01";
function openSourceModal(slot) {
  targetSlot = slot;
  document.getElementById("modalTitle").textContent = `Configure Source for ${slot}`;
  document.getElementById("modalMessage").textContent = "";
  document.getElementById("sourceModal").style.display = "flex";
}
function closeSourceModal() {
  document.getElementById("sourceModal").style.display = "none";
}
function setModalTab(tab) {
  document.querySelectorAll(".modal-tab").forEach(t => t.classList.remove("active"));
  document.getElementById("uploadVideoForm").style.display = "none";
  document.getElementById("webcamForm").style.display = "none";
  document.getElementById("rtspForm").style.display = "none";

  if (tab === "video") {
    document.getElementById("tabVideo").classList.add("active");
    document.getElementById("uploadVideoForm").style.display = "flex";
  } else if (tab === "webcam") {
    document.getElementById("tabWebcam").classList.add("active");
    document.getElementById("webcamForm").style.display = "flex";
  } else {
    document.getElementById("tabRtsp").classList.add("active");
    document.getElementById("rtspForm").style.display = "flex";
  }
}

async function handleUploadVideo(e) {
  e.preventDefault();
  const file = document.getElementById("videoFileInput").files[0];
  if (!file) return;
  const msg = document.getElementById("modalMessage");
  msg.textContent = "Uploading video feed...";
  try {
    const formData = new FormData();
    formData.append("video", file, file.name);
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (res.ok) {
      msg.textContent = `Streaming ${file.name} successfully!`;
      setTimeout(closeSourceModal, 1000);
    } else {
      msg.textContent = "Error loading video file.";
    }
  } catch (err) {
    msg.textContent = "Server communication error.";
  }
}

async function handleSwitchWebcam(e) {
  e.preventDefault();
  const idx = document.getElementById("webcamIndex").value || "0";
  const msg = document.getElementById("modalMessage");
  msg.textContent = "Connecting to webcam...";
  try {
    const res = await fetch("/api/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: idx })
    });
    if (res.ok) {
      msg.textContent = `Webcam ${idx} connected!`;
      setTimeout(closeSourceModal, 1000);
    } else {
      msg.textContent = "Could not open specified camera.";
    }
  } catch (err) {
    msg.textContent = "Server communication error.";
  }
}

async function handleRtspStream(e) {
  e.preventDefault();
  const url = document.getElementById("rtspUrl").value.trim();
  const msg = document.getElementById("modalMessage");
  if (!url) {
    msg.textContent = "Please provide an RTSP stream URL.";
    return;
  }
  msg.textContent = "Connecting to stream...";
  try {
    const res = await fetch("/api/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: url })
    });
    if (res.ok) {
      msg.textContent = `Streaming from configured source.`;
      setTimeout(closeSourceModal, 1200);
    } else {
      msg.textContent = "Could not open that stream URL.";
    }
  } catch (err) {
    msg.textContent = "Server communication error.";
  }
}

// Toggle Tech Drawer
function toggleTechDrawer() {
  document.getElementById("techDrawer").classList.toggle("open");
}

// ---- Live alert + trend state (all values below come from /api/status) ----
const FIGHT_SCENES = ["FIGHT", "MULTI_PERSON_FIGHT", "GROUP_ASSAULT"];
const FIGHT_TITLES = {
  FIGHT: "Possible Fight Detected",
  MULTI_PERSON_FIGHT: "Group Fight Detected",
  GROUP_ASSAULT: "Group Assault Detected"
};
const ALERT_WORDS = {
  "Fall Detection": "Fall Detected",
  "Abandoned Bag": "Unattended Bag",
  "Unattended Bag Candidate": "Unattended Bag",
  "Violence/Fight (legacy BiLSTM)": "Possible Fight",
  "Weapon Threat": "Weapon Detected",
  "Fire Threat": "Fire Detected",
  "Smoke Threat": "Smoke Detected"
};
function alertWords(key) {
  if (Object.prototype.hasOwnProperty.call(ALERT_WORDS, key)) return ALERT_WORDS[key];
  return String(key).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}
function escHtml(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function alertCardHTML(id, title, cam, confPct, reason) {
  const conf = (confPct === null || confPct === undefined)
    ? "" : `<div class="alert-confidence">Confidence: <span>${confPct}%</span></div>`;
  return `<div class="alert-card" id="${id}">` +
    `<div class="alert-top"><span class="alert-title">${escHtml(title)}</span>` +
    `<span class="alert-cam">${escHtml(cam)}</span></div>${conf}` +
    `<div class="alert-reasoning">Reasoning: ${escHtml(reason)}</div>` +
    `<div class="alert-actions">` +
    `<button class="btn-view-live" onclick="focusCamera('cardCam1')">View Live</button>` +
    `<button class="btn-dismiss" onclick="dismissAlert('${id}')">Dismiss</button></div></div>`;
}
function sceneReason(s) {
  const incidents = (s.scene && s.scene.incidents) || [];
  const parts = incidents.map(inc =>
    `${(inc.members || []).length} people, ${inc.violent_edges || 0} violent link(s)`);
  return parts.length ? parts.join("; ") : "Violent interaction detected";
}
function topPairScore(s) {
  let top = null;
  for (const p of (s.pairs || [])) {
    if (p.score !== null && p.score !== undefined && (top === null || p.score > top)) top = p.score;
  }
  return top;
}
function renderAlerts(s) {
  const list = document.getElementById("alertsList");
  const details = s.alert_details || {};
  const verdict = s.scene ? s.scene.verdict : "NO_FIGHT";
  let html = "";
  let count = 0;
  function push(id, title, cam, confPct, reason) {
    if (dismissedAlerts.has(id)) return;
    count++;
    html += alertCardHTML(id, title, cam, confPct, reason);
  }
  if (FIGHT_SCENES.indexOf(verdict) >= 0) {
    const top = topPairScore(s);
    push("dyn-fight", FIGHT_TITLES[verdict] || FIGHT_TITLES.FIGHT,
      "(CAM-01 Live Feed)", top === null ? null : Math.round(top * 100),
      sceneReason(s));
  }
  Object.keys(details).forEach(function (name) {
    const d = details[name] || {};
    const conf = (d.confidence === null || d.confidence === undefined)
      ? null : Math.round(d.confidence * 100);
    push(alertCardId("alert", name), alertWords(name), "(CAM-01 Live Feed)",
      conf, d.message || alertWords(name));
  });
  if (!count) {
    html = `<div class="alert-card calm"><div class="alert-top">` +
      `<span class="alert-title">No active alerts</span></div>` +
      `<div class="alert-reasoning">Monitoring is running. New detections will appear here.</div></div>`;
  }
  list.innerHTML = html;
  document.getElementById("topAlertCount").textContent = count;
}

// Camera overlay pills: only the live CAM-01 slot, only during real events
function renderOverlayPills(s) {
  const box = document.getElementById("camOverlayAlerts");
  if (!box) return;
  const verdict = s.scene ? s.scene.verdict : "NO_FIGHT";
  const names = Object.keys(s.alert_details || {});
  let html = "";
  if (FIGHT_SCENES.indexOf(verdict) >= 0) {
    const top = topPairScore(s);
    html += `<div class="cam-alert-tag"><svg viewBox="0 0 24 24"><path d="M12 2L1 21h22L12 2zm0 4l7.5 13h-15L12 6zm-1 5v4h2v-4h-2zm0 6v2h2v-2h-2z"/></svg>` +
      `Alert: Possible Fight${top === null ? "" : " " + Math.round(top * 100) + "%"}</div>`;
  }
  names.forEach(function (name) {
    html += `<div class="cam-alert-tag"><svg viewBox="0 0 24 24"><path d="M12 2L1 21h22L12 2zm0 4l7.5 13h-15L12 6zm-1 5v4h2v-4h-2zm0 6v2h2v-2h-2z"/></svg>` +
      `Alert: ${escHtml(alertWords(name))}</div>`;
  });
  box.innerHTML = html;
}

// AI confidence trendline from live top pair scores (last 40 polls)
const scoreHistory = [];
function renderTrend() {
  const line = document.getElementById("trendLine");
  const area = document.getElementById("trendArea");
  if (!line) return;
  const N = 40, W = 500, H = 90;
  const hist = scoreHistory.slice(-N);
  while (hist.length < 2) hist.unshift(0);
  const full = hist.slice();
  while (full.length < N) full.unshift(full[0]);
  let d = "";
  full.forEach(function (v, i) {
    const x = (i / (N - 1)) * W;
    const y = H - 6 - Math.max(0, Math.min(1, v)) * (H - 16);
    d += (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
  });
  line.setAttribute("d", d);
  if (area) area.setAttribute("d", d + `L ${W} ${H} L 0 ${H} Z`);
}

// Dynamic Backend Polling (/api/status)
async function pollBackend() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const s = await res.json();

    // Update Latency / FPS
    if (s.fps) {
      const lat = (1.0 / Math.max(s.fps, 1)).toFixed(2);
      document.getElementById("latencyVal").textContent = `${lat}s`;
    }

    document.getElementById("camCount").textContent = (s.running !== false) ? "1" : "0";

    renderAlerts(s);
    renderOverlayPills(s);

    const top = topPairScore(s);
    scoreHistory.push(top === null ? 0 : top);
    if (scoreHistory.length > 120) scoreHistory.splice(0, scoreHistory.length - 120);
    renderTrend();

    // Populate Tech Drawer info
    document.getElementById("techMeta").textContent =
      `Frame: ${s.frame || 0} | FPS: ${s.fps || 0} | Source: ${s.source || "None"}
Scene Verdict: ${s.scene ? s.scene.verdict : "NO_FIGHT"}`;
    document.getElementById("techModels").textContent = JSON.stringify(s.model || {}, null, 2);
    document.getElementById("techPairs").textContent =
      `Active Persons: ${(s.people || []).length}
Open Pairs: ${(s.pairs || []).length}
` +
      JSON.stringify(s.pairs || [], null, 2);
    document.getElementById("techDiag").textContent = JSON.stringify(s.diagnostics || {}, null, 2);

  } catch (e) {
    // Polling failure / starting up
  }
}
setInterval(pollBackend, 500);
pollBackend();
</script>
</body>
</html>
"""


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ViglanceUI/1.0"

        def _send(self, code: int, ctype: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict, code: int = 200):
            self._send(code, "application/json", json.dumps(payload).encode())

        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self._send(200, "text/html", DASHBOARD_HTML.encode())
            elif self.path == "/api/status":
                with state.lock:
                    body = json.dumps(state.status).encode()
                self._send(200, "application/json", body)
            elif self.path == "/snapshot.jpg":
                with state.lock:
                    jpeg = state.jpeg
                if jpeg is None:
                    self._send(503, "text/plain", b"no frame yet")
                else:
                    self._send(200, "image/jpeg", jpeg)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while not state.stop:
                        with state.lock:
                            jpeg = state.jpeg
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                         + jpeg + b"\r\n")
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_UPLOAD_BYTES:
                self._send_json({"error": "upload exceeds the 256 MiB limit"}, 413)
                return
            if self.path == "/api/source":
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    source = str(payload.get("source", "0"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    self._send_json({"error": "invalid JSON source request"}, 400)
                    return
                state.request_source(source)
                self._send_json({"ok": True, "message": f"source requested: {source}"})
                return
            if self.path == "/api/upload":
                content_type = self.headers.get("Content-Type", "")
                upload_dir = Path(tempfile.gettempdir()) / "vigilance_uploads"
                try:
                    destination = save_video_upload(
                        self.rfile, content_type, length, upload_dir
                    )
                except VideoUploadError as exc:
                    self._send_json({"error": str(exc)}, exc.status_code)
                    return
                state.request_source(str(destination))
                self._send_json({
                    "ok": True,
                    "message": "video uploaded; playback requested",
                    "source": str(destination),
                })
                return
            self._send_json({"error": "not found"}, 404)

        def log_message(self, *args):  # quiet favicon noise
            pass

    return Handler


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VIGLANCE web UI (MJPEG + dashboard)")
    p.add_argument("--source", default=None, help="webcam index, video file, or image")
    p.add_argument("--config", default=None)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--synthetic", action="store_true",
                   help="scripted demo persons; no camera or YOLO weights needed")
    p.add_argument("--fight-checkpoint", default=None,
                   help="post-training plug-in: ST-GCN++ fight checkpoint (.pth)")
    p.add_argument("--no-fight", action="store_true")
    p.add_argument("--no-threats", action="store_true",
                   help="disable the auxiliary YOLO threat model")
    p.add_argument("--no-fire", action="store_true",
                   help="disable fire/smoke alerts from the threat model")
    p.add_argument("--threats", action="store_true",
                   help="opt in to the auxiliary YOLO threat model (OFF by default; violence-only focus)")
    p.add_argument("--enable-weapon-alerts", action="store_true",
                   help="enable weapon alerts (disabled by default for demo stability)")
    p.add_argument("--no-threat-download", action="store_true",
                   help="use only the local auxiliary YOLO weights")
    p.add_argument("--surveillance-weights", default=None,
                   help="local weights for the surveillance YOLO model")
    p.add_argument("--telegram", action="store_true",
                   help="enable Telegram incident alerts (needs chat id + token)")
    p.add_argument("--telegram-chat-id", default=None)
    p.add_argument("--telegram-bot-token", default=None,
                   help="prefer VIGILANCE_TELEGRAM_BOT_TOKEN env var instead")
    p.add_argument("--telegram-cooldown", type=float, default=None,
                   help="minimum seconds between Telegram messages")
    p.add_argument("--telegram-test", action="store_true",
                   help="send one test photo+message via Telegram and exit")
    p.add_argument("--log-level", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = VigilanceConfig()
    if args.config:
        cfg = VigilanceConfig.from_file(args.config)
    cfg.from_env()
    cfg.apply_overrides(source=None if args.synthetic else args.source)
    setup_logging(args.log_level or cfg.log_level)

    if args.telegram_test:
        from core.alert_dispatch import Incident, TelegramConfig, TelegramNotifier
        token = args.telegram_bot_token or cfg.telegram_bot_token
        tcfg = TelegramConfig(
            enabled=True, bot_token=token,
            chat_id=args.telegram_chat_id or cfg.telegram_chat_id,
            cooldown_sec=0.0,
        )
        notifier = TelegramNotifier(tcfg)
        if not notifier.ready:
            print("telegram-test ABORT: need --telegram-chat-id and "
                  "VIGILANCE_TELEGRAM_BOT_TOKEN")
            return 2
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "VIGLANCE telegram test", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", frame)
        incident = Incident(key="telegram-test", title="TEST: fight detected",
                            scene="FIGHT", alerts=["Violence/Fight"],
                            timestamp=0.0)
        sent = notifier.maybe_notify(incident,
                                     bytes(buf) if ok else None, now=0.0)
        print("telegram-test " + ("SENT - check your chat" if sent
                                  else "FAILED - see log"))
        return 0 if sent else 1

    from tests.conftest import FakeDetector, black_frames
    fire_on = cfg.enable_fire and not args.no_fire and not args.synthetic
    threats_on = ((args.threats or cfg.enable_threats)
                  and not args.no_threats and not args.synthetic)
    use_threat_detector = threats_on or fire_on
    use_auxiliary_threats = threats_on
    if use_threat_detector:
        cfg.enable_bag = False
    if args.synthetic:
        script = [[(1, 60.0), (2, 220.0)], [(1, 64.0), (2, 216.0), (3, 380.0)],
                  [(1, 68.0), (3, 376.0)], [(1, 72.0), (2, 208.0), (3, 372.0)]] * 25
        detector, frames = FakeDetector(script), black_frames(len(script), h=480, w=640)
        source_label = "synthetic"
    else:
        from core.detector import Detector  # real YOLO; raises informatively if missing
        try:
            detector = Detector(
                pose_model_path=cfg.pose_weights, det_model_path=cfg.det_weights,
                tracker=cfg.tracker, fallback_tracker=cfg.tracker_fallback,
                conf=cfg.conf, imgsz=cfg.imgsz, device=cfg.device,
                max_people=cfg.max_people,
                load_object_model=not use_auxiliary_threats)
        except Exception as exc:
            log.error("event=ui_detector_failed error=%r (try --synthetic)", exc)
            print(f"Detector failed: {exc}\nTry: python ui.py --synthetic")
            return 2
        frames, source_label = None, cfg.source

    if args.no_fight:
        cfg.enable_fight = False
    cfg.validate()
    pipeline = Pipeline(config=cfg, detector=detector)
    if args.fight_checkpoint:  # post-training plug-in; absent model stays usable
        loaded = pipeline.load_fight_model(args.fight_checkpoint)
        if loaded is None:
            log.warning("event=ui_no_fight_model (continuing without scores)")
    elif (not args.synthetic and Path(cfg.fight_cfg.model.checkpoint).exists()
          and not pipeline.pair_adapter.is_ready):
        pipeline.load_fight_model()

    threat_detector = None
    if use_threat_detector:
        from core.threat_detection import SurveillanceThreatDetector
        threat_detector = SurveillanceThreatDetector(
            surveillance_weights=args.surveillance_weights,
            download_missing=not args.no_threat_download,
            conf=cfg.conf,
            imgsz=cfg.imgsz,
            device=cfg.device,
            bag_proximity_px=cfg.bag_proximity_px,
            enable_weapon_alerts=args.enable_weapon_alerts,
            enable_fire_alerts=fire_on,
            enable_bag_alerts=False,
        )

    state = LiveState()
    notifier = None
    if args.telegram or cfg.telegram_enabled:
        from core.alert_dispatch import TelegramConfig, TelegramNotifier
        token = args.telegram_bot_token or cfg.telegram_bot_token
        tcfg = TelegramConfig(
            enabled=True,
            bot_token=token,
            chat_id=args.telegram_chat_id or cfg.telegram_chat_id,
            cooldown_sec=(args.telegram_cooldown
                          if args.telegram_cooldown is not None
                          else cfg.telegram_cooldown_sec),
            notify_fight=True, notify_fall=True, notify_fire=True,
            notify_bag=cfg.telegram_notify_bag,
            notify_weapon=cfg.telegram_notify_weapon,
        )
        notifier = TelegramNotifier(tcfg)
        if notifier.ready:
            log.info("event=telegram_ready cooldown=%.0fs", tcfg.cooldown_sec)
        else:
            log.warning("event=telegram_not_configured "
                        "(need --telegram-chat-id and VIGILANCE_TELEGRAM_BOT_TOKEN)")
    pump = threading.Thread(target=pump_frames,
                            args=(state, pipeline,
                                  cfg.source if frames is None else source_label,
                                  cfg.fps, frames, threat_detector, notifier),
                            daemon=True)
    pump.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    log.info("event=ui_ready url=http://127.0.0.1:%d source=%s", args.port, source_label)
    print(f"VIGLANCE UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop = True
        server.shutdown()
        pump.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
