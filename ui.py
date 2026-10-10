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


def pump_frames(state: LiveState, pipeline: Pipeline, source, fps_fallback: float,
                synthetic_frames=None, threat_detector=None) -> None:
    from core.visualization import (draw_alerts, draw_pairs, draw_scene_panel,
                                    draw_threats, draw_tracks)

    cap = None
    current_source = str(source)
    current_generation = 0
    fps = float(fps_fallback)
    idx, last, ema = 0, time.perf_counter(), None
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

                if cap is None:
                    src: object = (int(current_source)
                                   if str(current_source).isdigit()
                                   else current_source)
                    cap = cv2.VideoCapture(src)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        with state.lock:
                            state.status = {
                                "running": False,
                                "source": current_source,
                                "error": f"cannot open source {current_source!r}",
                            }
                        time.sleep(0.1)
                        continue
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_fallback)
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    timeline = state.timeline_snapshot()
                    with state.lock:
                        state.status = {
                            "running": False,
                            "source": current_source,
                            "timeline": timeline,
                        }
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
<title>VIGLANCE - Camera Watch</title>
<style>
:root{
  --bg:#080b11;
  --surface:#111823;
  --surface-2:#0c111a;
  --line:#223047;
  --text:#eaf0f8;
  --muted:#a3b1c6;
  --calm:#5fd08a;
  --warning:#f6b93f;
  --danger:#ff8177;
  --neutral:#a3b1c6;
  --radius:16px;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg);
  color:var(--text);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px;
  line-height:1.5;
  overflow-x:hidden;
}
.masthead{
  display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;
  padding:18px 26px;border-bottom:1px solid var(--line);background:var(--surface-2);
}
.brand{font-size:19px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
.brand span{color:var(--muted);font-weight:600;letter-spacing:.03em;text-transform:none;font-size:14px;margin-left:10px}
.srcpill{
  font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:7px 15px;
  overflow-wrap:anywhere;max-width:100%;
}
.layout{
  display:grid;grid-template-columns:minmax(0,1.6fr) minmax(320px,1fr);
  gap:20px;align-items:stretch;max-width:1580px;margin:0 auto;padding:22px 26px;
}
@media (max-width:1080px){.layout{grid-template-columns:minmax(0,1fr)}}
.stage{min-width:0}
.console{min-width:0;display:flex;flex-direction:column}
.console .activitycard{flex:1 1 auto;display:flex;flex-direction:column;min-height:200px}
.card{
  background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:22px 24px;margin-bottom:20px;
}
.label{
  font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);margin:0 0 14px;
}
.hero{
  position:relative;display:flex;align-items:center;gap:20px;overflow:hidden;
  border:1px solid var(--line);border-left:8px solid var(--accent);
  border-radius:var(--radius);padding:26px 28px;margin-bottom:20px;
  background-color:var(--surface);
  background-image:linear-gradient(100deg,var(--tint) 0%,transparent 58%);
}
.hero[data-severity="calm"]{--accent:var(--calm);--tint:rgba(95,208,138,.13)}
.hero[data-severity="warning"]{--accent:var(--warning);--tint:rgba(246,185,63,.13)}
.hero[data-severity="danger"]{--accent:var(--danger);--tint:rgba(255,129,119,.15)}
.hero[data-severity="neutral"]{--accent:var(--neutral);--tint:rgba(163,177,198,.10)}
#heroIcon{flex:none;display:block}
#heroIcon .ico{width:58px;height:58px;color:var(--accent)}
.herotext{min-width:0}
.statechip{
  display:inline-block;font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);border:1px solid currentColor;border-radius:999px;
  padding:4px 11px;margin-bottom:12px;
}
#heroTitle{
  margin:0;font-weight:800;letter-spacing:-.02em;line-height:1.06;
  font-size:clamp(26px,3.6vw,46px);overflow-wrap:anywhere;
}
#heroDetail{margin:10px 0 0;font-size:clamp(14px,1.3vw,17px);color:var(--muted);overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px}
.chip{
  font-size:13px;font-weight:700;border-radius:999px;padding:7px 15px;
  border:1px solid currentColor;color:var(--danger);
}
img#stream{
  display:block;width:100%;max-width:100%;aspect-ratio:16/9;object-fit:contain;
  background:#000;border:1px solid var(--line);border-radius:12px;
}
#stage[data-off="1"] img#stream{opacity:.22;filter:grayscale(1) brightness(.75) contrast(.8)}
.playerbar{display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;margin:14px 0 0}
.badge{
  flex:none;font-size:10px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;
  border:1px solid currentColor;border-radius:999px;padding:5px 12px;
  color:var(--calm);background:rgba(95,208,138,.12);
}
#stage[data-off="1"] .badge{color:var(--danger);background:rgba(255,129,119,.14)}
.videonote{margin:0;flex:1 1 220px;min-width:0;font-size:13px;color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.tile{background:var(--surface-2);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.tile .n{font-size:30px;font-weight:800;line-height:1.1;letter-spacing:-.02em;overflow-wrap:anywhere}
.tile .n.small{font-size:18px;line-height:1.3;letter-spacing:0;word-break:normal;overflow-wrap:break-word}
.tile .k{font-size:10px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-top:9px}
.tile .s{font-size:13px;color:var(--muted);margin-top:6px;overflow-wrap:anywhere}
.activity{list-style:none;margin:0;padding:0;flex:1 1 auto;min-height:0;overflow-y:auto;overscroll-behavior:contain}
.activity li{display:flex;align-items:center;gap:12px;padding:13px 0;border-bottom:1px solid var(--line)}
.activity li:last-child{border-bottom:0;padding-bottom:0}
.activity .tag{
  flex:none;font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;
  border:1px solid currentColor;border-radius:999px;padding:4px 10px;
}
.activity .grow{flex:1 1 auto;min-width:0;overflow-wrap:anywhere}
.activity .time{flex:none;font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}
li.sev-danger .tag{color:var(--danger)}
li.sev-warning .tag{color:var(--warning)}
li.sev-calm .tag{color:var(--calm)}
li.sev-neutral .tag{color:var(--neutral)}
.actions{display:grid;gap:12px}
.actions form{display:grid;gap:12px}
button{
  width:100%;min-height:62px;padding:18px 20px;border:0;border-radius:14px;
  font-family:inherit;font-size:16px;font-weight:800;color:#04120b;
  background:var(--calm);cursor:pointer;transition:filter .18s ease,transform .18s ease;
}
button:hover{filter:brightness(1.08)}
button:active{transform:translateY(1px)}
button.ghost{background:var(--surface-2);color:var(--text);border:1px solid var(--line)}
button:focus-visible,summary:focus-visible{outline:3px solid var(--warning);outline-offset:3px}
input[type=file]{
  width:100%;font-family:inherit;font-size:14px;color:var(--muted);
  background:var(--surface-2);border:1px dashed var(--line);border-radius:14px;padding:14px;
}
input[type=file]::file-selector-button{
  font-family:inherit;font-weight:700;color:var(--text);background:var(--surface);
  border:1px solid var(--line);border-radius:10px;padding:9px 14px;margin-right:12px;cursor:pointer;
}
#sourceMessage{margin:14px 0 0;font-size:14px;color:var(--muted);min-height:1.5em;overflow-wrap:anywhere}
details.tech{max-width:1580px;margin:0 auto;padding:0 26px 48px}
details.tech>summary{
  cursor:pointer;list-style:none;display:flex;align-items:center;gap:12px;
  padding:18px 24px;border:1px solid var(--line);border-radius:var(--radius);
  background:var(--surface);color:var(--muted);
  font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;
}
details.tech>summary::-webkit-details-marker{display:none}
.chev{
  width:9px;height:9px;flex:none;border-right:2px solid currentColor;border-bottom:2px solid currentColor;
  transform:rotate(-45deg);transition:transform .18s ease;
}
details.tech[open]>summary{border-bottom-left-radius:0;border-bottom-right-radius:0;color:var(--text)}
details.tech[open]>summary .chev{transform:rotate(45deg)}
.techbody{
  border:1px solid var(--line);border-top:0;
  border-bottom-left-radius:var(--radius);border-bottom-right-radius:var(--radius);
  background:var(--surface-2);padding:22px 24px;
}
.tgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:22px}
.tcell{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.pre{
  margin:0;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#c3d0e2;
  white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;
}
.scroll-x{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}
th{font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.empty{margin:10px 0 0;font-size:13px;color:var(--muted)}
.toast{
  position:fixed;right:20px;bottom:20px;z-index:9;max-width:min(92vw,420px);
  background:var(--danger);color:#1b0a09;padding:15px 20px;border-radius:12px;
  box-shadow:0 18px 40px rgba(0,0,0,.5);font-weight:800;overflow-wrap:anywhere;
}
@media (max-width:560px){
  .masthead{padding:14px 16px}
  .layout{padding:16px}
  .hero{padding:20px 18px;gap:14px;border-left-width:6px}
  #heroIcon .ico{width:40px;height:40px}
  .card{padding:18px 16px}
  .tiles{grid-template-columns:minmax(0,1fr)}
  details.tech{padding:0 16px 36px}
  .techbody{padding:16px}
  .toast{right:12px;left:12px;bottom:12px;max-width:none}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<header class="masthead">
  <div class="brand">Viglance<span>Camera watch</span></div>
  <div class="srcpill" id="srcPill">Starting up</div>
</header>

<main class="layout">
  <section class="stage" id="stage" data-off="0">
    <div class="hero" id="hero" data-severity="neutral" aria-live="polite">
      <span id="heroIcon" aria-hidden="true"><svg class="ico" viewBox="0 0 24 24" focusable="false"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg></span>
      <div class="herotext">
        <span class="statechip" id="heroState">Status: live</span>
        <h1 id="heroTitle">Watching</h1>
        <p id="heroDetail">Analysing the video…</p>
      </div>
    </div>
    <div class="card">
      <div class="chips" id="threatChips"></div>
      <img id="stream" src="/stream.mjpg" alt="Live camera view">
      <div class="playerbar">
        <span class="badge" id="liveBadge">Live feed</span>
        <p class="videonote" id="videoNote">Live view from the camera. Every frame is processed on this computer.</p>
      </div>
    </div>
  </section>

  <section class="console">
    <div class="card">
      <h2 class="label">At a glance</h2>
      <div class="tiles">
        <div class="tile">
          <div class="n" id="statPeople">0</div>
          <div class="k">People in view</div>
          <div class="s" id="statPeopleSub">No one in view</div>
        </div>
        <div class="tile">
          <div class="n" id="statAlerts">0</div>
          <div class="k">Alerts right now</div>
          <div class="s" id="statAlertsSub">Nothing to worry about</div>
        </div>
        <div class="tile">
          <div class="n" id="statIncidents">0</div>
          <div class="k">Recent incidents</div>
          <div class="s" id="statIncidentsSub">Nothing recorded yet</div>
        </div>
        <div class="tile">
<div class="n small" id="statSource">Starting up</div>
<div class="k" id="statSourceLabel">Watching</div>
          <div class="s" id="statSourceSub">Connecting to the camera</div>
        </div>
      </div>
    </div>

    <div class="card activitycard">
      <h2 class="label">Recent activity</h2>
      <ul class="activity" id="activity">
        <li class="sev-neutral"><span class="tag">Live</span><span class="grow">Nothing to report yet. Activity will appear here.</span></li>
      </ul>
    </div>

    <div class="card">
      <h2 class="label">What you can do</h2>
      <div class="actions">
        <button id="webcam" type="button">Use the computer camera</button>
        <form id="uploadForm">
          <input id="videoFile" type="file" accept="video/*,.avi,.mkv,.mov,.mp4,.mpeg,.mpg,.webm,.wmv" required>
          <button type="submit">Play a video file</button>
        </form>
        <button id="notifications" type="button" class="ghost">Turn on fight alerts on this device</button>
      </div>
      <p id="sourceMessage" role="status"></p>
    </div>
  </section>
</main>

<details class="tech">
  <summary><span class="chev" aria-hidden="true"></span>Technical details (for engineers)</summary>
  <div class="techbody">
    <pre class="pre" id="meta">frame - t=- fps=- src=-</pre>
    <div class="tgrid">
      <div class="tcell">
        <h3 class="label">Model state</h3>
        <pre class="pre" id="model">ready=- head_trained=- ckpt=none</pre>
      </div>
      <div class="tcell">
        <h3 class="label">Threat models</h3>
        <pre class="pre" id="threatModels">{}</pre>
      </div>
      <div class="tcell">
        <h3 class="label">People</h3>
        <div class="scroll-x" id="people"></div>
      </div>
      <div class="tcell">
        <h3 class="label">Pairs</h3>
        <div class="scroll-x" id="pairs"></div>
      </div>
      <div class="tcell">
        <h3 class="label">Threat detections</h3>
        <div class="scroll-x" id="threats"></div>
      </div>
      <div class="tcell">
        <h3 class="label">Diagnostics</h3>
        <pre class="pre" id="diag">{}</pre>
      </div>
    </div>
  </div>
</details>

<div id="notification" class="toast" role="status" hidden></div>
<script>
const SCENES={
  NO_FIGHT:{title:"All clear",detail:"No fighting has been detected.",severity:"calm"},
  FIGHT:{title:"Possible fight detected",detail:"Two people appear to be fighting. Check this area.",severity:"danger"},
  MULTI_PERSON_FIGHT:{title:"Group fight detected",detail:"Several people appear to be fighting.",severity:"danger"},
  GROUP_ASSAULT:{title:"Group assault detected",detail:"A group appears to be attacking someone.",severity:"danger"}
};
const WATCHING={title:"Watching",detail:"Analysing the video…",severity:"neutral"};
const LOST={title:"Video connection lost",detail:"The video feed stopped. Check the camera or video file, then try again.",severity:"danger"};
const ALERT_WORDS={
  "Fall Detection":"Someone may have fallen",
  "Abandoned Bag":"Unattended bag",
  "Violence/Fight (legacy BiLSTM)":"Possible fight",
  "Weapon Threat":"Weapon detected",
  "Fire Threat":"Fire detected",
  "Smoke Threat":"Smoke detected",
  "Unattended Bag Candidate":"Unattended bag"
};
const CATEGORY_WORDS={weapon:"Weapon",fire:"Fire",smoke:"Smoke",bag:"Bag"};
const STATE_WORDS={calm:"Status: clear",warning:"Status: check",danger:"Status: alert",neutral:"Status: live"};
const STATE_TAGS={calm:"Clear",warning:"Check",danger:"Alert",neutral:"Live"};
const FIGHT_SCENES=["FIGHT","MULTI_PERSON_FIGHT","GROUP_ASSAULT"];
const ICONS={
 calm:'<svg class="ico" viewBox="0 0 24 24" focusable="false"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M8 12.4l2.7 2.7L16.2 9.4" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
 warning:'<svg class="ico" viewBox="0 0 24 24" focusable="false"><path d="M12 3.4l9 16.2H3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 9.6v5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><circle cx="12" cy="17.2" r="1.2" fill="currentColor"/></svg>',
 danger:'<svg class="ico" viewBox="0 0 24 24" focusable="false"><path d="M8.5 2.8h7L21.2 8.5v7L15.5 21.2h-7L2.8 15.5v-7z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 7.4v5.6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><circle cx="12" cy="16.3" r="1.2" fill="currentColor"/></svg>',
 neutral:'<svg class="ico" viewBox="0 0 24 24" focusable="false"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>'
};
function put(id,text){document.getElementById(id).textContent=text;}
function message(text){document.getElementById("sourceMessage").textContent=text;}
function titleCase(raw){
  const parts=String(raw===undefined||raw===null?"":raw).split("_").join(" ").split(" ");
  return parts.filter(function(w){return w.length>0;})
    .map(function(w){return w.charAt(0).toUpperCase()+w.slice(1);}).join(" ");
}
function alertWords(key){
  return Object.prototype.hasOwnProperty.call(ALERT_WORDS,key)?ALERT_WORDS[key]:titleCase(key);
}
function categoryWords(key){
  return Object.prototype.hasOwnProperty.call(CATEGORY_WORDS,key)?CATEGORY_WORDS[key]:titleCase(key);
}
function percent(value){
  const n=Number(value);
  if(!isFinite(n)) return "unknown";
  return Math.round(n*100)+"%";
}
function allDigits(text){
  if(!text.length) return false;
  for(let i=0;i<text.length;i++){const c=text.charCodeAt(i);if(c<48||c>57) return false;}
  return true;
}
function lastSegment(text){
  let part="",best="",seen=false;
  for(let i=0;i<text.length;i++){
    const code=text.charCodeAt(i);
    if(code===47||code===92){if(part.length) best=part;part="";seen=true;}
    else part+=text.charAt(i);
  }
  if(part.length) best=part;
  return best.length?best:(seen?text:text);
}
function sourceWords(raw){
  const text=String(raw===undefined||raw===null?"":raw);
  if(!text.length) return "Not connected";
  if(allDigits(text)) return "Computer camera";
  if(text==="synthetic") return "Demo video";
  return lastSegment(text);
}
function clock(seconds){
  const n=Number(seconds);
  const total=isFinite(n)&&n>0?n:0;
  const m=Math.floor(total/60);
  const s=Math.floor(total%60);
  return m+":"+(s<10?"0":"")+s;
}
function severityOf(event){
  if(event.scene && event.scene!=="NO_FIGHT") return "danger";
  return (event.alerts||[]).length?"warning":"calm";
}
const LIVE_CAPTION="Live view from the camera. Every frame is processed on this computer.";
const DEAD_CAPTION="No live picture. Choose the camera or a video file to start watching again.";
function renderFriendly(s){
  const running=s.running!==false;
  const view=!running?LOST:((s.scene&&SCENES[s.scene.verdict])||WATCHING);
  const hero=document.getElementById("hero");
  hero.setAttribute("data-severity",view.severity);
  document.getElementById("heroIcon").innerHTML=ICONS[view.severity];
  put("heroState",STATE_WORDS[view.severity]);
  put("heroTitle",view.title);
  put("heroDetail",view.detail);
  document.getElementById("stage").setAttribute("data-off",running?"0":"1");
  put("liveBadge",running?"Live feed":"Stopped");
  put("videoNote",running?LIVE_CAPTION:DEAD_CAPTION);

  const people=s.people||[];
  put("statPeople",String(people.length));
  put("statPeopleSub",people.length?(people.length===1?"1 person being tracked":"people being tracked"):"No one in view");

  const alerts=(s.alerts||[]).map(alertWords);
  put("statAlerts",String(alerts.length));
  put("statAlertsSub",alerts.length?alerts.join(", "):"Nothing to worry about");

  const events=s.timeline||[];
  put("statIncidents",String(events.length));
  put("statIncidentsSub",events.length?(events.length===1?"1 incident recorded":"incidents recorded"):"Nothing recorded yet");

  const source=sourceWords(s.source);
  put("statSource",source);
  put("statSourceSub",running?"Connected and watching":"Not connected right now");
  put("statSourceLabel",running?"Watching":"Stopped");
  put("srcPill",running?source:source+" - stopped");

  const chips=document.getElementById("threatChips");
  chips.innerHTML=(s.threats||[]).length
    ? s.threats.map(function(t){
        return `<span class="chip">${categoryWords(t.category)} spotted &middot; ${percent(t.confidence)}</span>`;
      }).join("")
    : "";

  const list=document.getElementById("activity");
  if(!events.length){
    const emptyTag=running?"Live":"Offline";
    const emptyText=running?"Nothing to report yet. Activity will appear here.":"Monitoring is paused until a video source is connected.";
    list.innerHTML=`<li class="sev-${running?"neutral":"danger"}"><span class="tag">${emptyTag}</span><span class="grow">${emptyText}</span></li>`;
  }else{
    list.innerHTML=events.slice(-6).reverse().map(function(event){
      const known=SCENES[event.scene];
      const headline=known?known.title:"Activity recorded";
      const words=(event.alerts||[]).map(alertWords).filter(function(w){
        return headline.toLowerCase().indexOf(w.toLowerCase())<0;
      });
      const severity=severityOf(event);
      const body=headline+(words.length?" - "+words.join(", "):"");
      return `<li class="sev-${severity}"><span class="tag">${STATE_TAGS[severity]}</span>`+
        `<span class="grow">${body}</span><span class="time">${clock(event.start)}</span></li>`;
    }).join("");
  }
}
function dash(value){return value===undefined||value===null?"-":value;}
function renderTechnical(s){
  const m=s.model||{};
  put("model",`ready=${dash(m.ready)}
head_trained=${dash(m.head_trained)}
ckpt=${m.checkpoint||"none"}`);
  put("threatModels",JSON.stringify(s.threat_models||{},null,1));
  put("diag",JSON.stringify(s.diagnostics||{},null,1));
  put("meta",`frame ${dash(s.frame)}  t=${dash(s.ts)}s  fps=${dash(s.fps)}  src=${s.source||""}${s.error?"  ERR "+s.error:""}`);

  const people=s.people||[];
  document.getElementById("people").innerHTML=people.length
    ? `<table><tr><th>ID</th><th>gen</th><th>bbox</th></tr>`+
      people.map(function(p){return `<tr><td>${p.id}</td><td>${p.gen}</td><td class="mono">${p.bbox}</td></tr>`;}).join("")+
      `</table>`
    : `<p class="empty">no active tracks</p>`;

  const pairs=s.pairs||[];
  document.getElementById("pairs").innerHTML=pairs.length
    ? `<table><tr><th>pair</th><th>score</th><th>buf</th><th>span s</th><th>ready</th><th>dist</th></tr>`+
      pairs.map(function(p){
        return `<tr><td>P${p.a}&harr;P${p.b}</td><td>${p.score===null||p.score===undefined?"—":p.score}</td><td>${p.buf}</td><td>${p.span===undefined?"—":p.span}</td><td>${p.ready===undefined?"—":(p.ready?"yes":"no")}</td><td>${p.dist}</td></tr>`;
      }).join("")+`</table>`
    : `<p class="empty">no open pairs</p>`;

  const threats=s.threats||[];
  document.getElementById("threats").innerHTML=threats.length
    ? `<table><tr><th>source</th><th>label</th><th>category</th><th>confidence</th></tr>`+
      threats.map(function(t){
        return `<tr><td>${t.source}</td><td>${t.label}</td><td>${t.category}</td>`+
          `<td>${percent(t.confidence)} <span class="mono">(${t.confidence})</span></td></tr>`;
      }).join("")+`</table>`
    : `<p class="empty">no threat detections</p>`;
}
let lastFightNotification="";
let toastTimer=null;
async function enableNotifications(){
  if("Notification" in window && Notification.permission==="default") await Notification.requestPermission();
}
function notifyFight(s){
  const events=s.timeline||[];
  const event=events.length?events[events.length-1]:null;
  if(!event||FIGHT_SCENES.indexOf(event.scene)<0) return;
  const key=event.start+":"+event.scene;
  if(key===lastFightNotification) return;
  lastFightNotification=key;
  const text=(SCENES[event.scene]?SCENES[event.scene].title:"Possible fight detected")+" at "+clock(event.start);
  const toast=document.getElementById("notification"); toast.textContent=text; toast.hidden=false;
  clearTimeout(toastTimer); toastTimer=setTimeout(function(){toast.hidden=true;},8000);
  if("Notification" in window && Notification.permission==="granted") new Notification("VIGLANCE fight alert",{body:text});
}
document.getElementById("notifications").onclick=async()=>{
  await enableNotifications();
  if(!("Notification" in window)){message("This browser cannot show alerts. Watch the status panel instead.");return;}
  if(Notification.permission==="granted") message("Fight alerts are on for this device.");
  else if(Notification.permission==="denied") message("This browser is blocking alerts. Allow notifications for this page in your browser settings to turn them on.");
  else message("Alerts are still off. Click again and choose Allow when your browser asks.");
};
document.getElementById("webcam").onclick=async()=>{
  await enableNotifications();
  message("Switching to the computer camera…");
  try{
    const r=await fetch("/api/source",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({source:"0"})});
    if(!r.ok){message("Could not open the camera. Close anything else using it, then try again.");return;}
    message("Now watching the computer camera.");
  }catch(e){
    message("Could not reach the dashboard server. Check that it is still running.");
  }
};
document.getElementById("uploadForm").onsubmit=async(e)=>{
  e.preventDefault();
  const file=document.getElementById("videoFile").files[0];
  if(!file){message("Choose a video file first.");return;}
  await enableNotifications();
  message("Opening "+file.name+"…");
  try{
    const body=new FormData(); body.append("video",file,file.name);
    const r=await fetch("/api/upload",{method:"POST",body});
    if(!r.ok){message("That video could not be opened. Try an MP4, MOV, AVI or MKV file.");return;}
    message("Now playing "+file.name+".");
  }catch(e){
    message("Could not reach the dashboard server. Check that it is still running.");
  }
};
async function tick(){
  try{
    const s=await (await fetch("/api/status")).json();
    notifyFight(s);
    renderFriendly(s);
    renderTechnical(s);
  }catch(e){/* stream still starting */}
}
setInterval(tick,500); tick();
</script></body></html>
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
    p.add_argument("--threats", action="store_true",
                   help="opt in to the auxiliary YOLO threat model (OFF by default; violence-only focus)")
    p.add_argument("--enable-weapon-alerts", action="store_true",
                   help="enable weapon alerts (disabled by default for demo stability)")
    p.add_argument("--no-threat-download", action="store_true",
                   help="use only the local auxiliary YOLO weights")
    p.add_argument("--surveillance-weights", default=None,
                   help="local weights for the surveillance YOLO model")
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

    from tests.conftest import FakeDetector, black_frames
    threats_on = ((args.threats or cfg.enable_threats)
                  and not args.no_threats and not args.synthetic)
    use_auxiliary_threats = threats_on
    if use_auxiliary_threats:
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
    if threats_on:
        from core.threat_detection import SurveillanceThreatDetector
        threat_detector = SurveillanceThreatDetector(
            surveillance_weights=args.surveillance_weights,
            download_missing=not args.no_threat_download,
            conf=cfg.conf,
            imgsz=cfg.imgsz,
            device=cfg.device,
            bag_proximity_px=cfg.bag_proximity_px,
            enable_weapon_alerts=args.enable_weapon_alerts,
        )

    state = LiveState()
    pump = threading.Thread(target=pump_frames,
                            args=(state, pipeline,
                                  cfg.source if frames is None else source_label,
                                  cfg.fps, frames, threat_detector),
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
