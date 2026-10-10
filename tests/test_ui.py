"""UI endpoint tests: synthetic mode only, no camera or YOLO weights needed."""

import json
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

cv2 = pytest.importorskip("cv2")

from ui import LiveState, make_handler, parse_args, pump_frames


@pytest.fixture()
def live_state():
    from http.server import ThreadingHTTPServer

    from config import VigilanceConfig
    from core.pipeline import Pipeline
    from tests.conftest import FakeDetector, black_frames

    cfg = VigilanceConfig()
    script = [[(1, 60.0), (2, 220.0)], [(1, 64.0), (2, 216.0)]]
    pipe = Pipeline(config=cfg, detector=FakeDetector(script))
    state = LiveState()
    pump = threading.Thread(target=pump_frames,
                            args=(state, pipe, "synthetic", cfg.fps,
                                  black_frames(len(script), h=240, w=320)),
                            daemon=True)
    pump.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(100):
        with state.lock:
            ready = state.jpeg is not None
        if ready:
            break
        time.sleep(0.05)
    yield port, state
    state.stop = True
    server.shutdown()


def test_status_endpoint_shape(live_state):
    port, _ = live_state
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as r:
        assert r.status == 200
        body = json.loads(r.read())
    for key in ("running", "frame", "fps", "people", "pairs", "scene",
                "model", "alerts", "threats", "threat_models", "timeline",
                "diagnostics"):
        assert key in body, key
    assert body["running"] is True
    assert isinstance(body["people"], list) and len(body["people"]) >= 1
    assert set(body["model"]) >= {"ready", "head_trained", "checkpoint"}


def test_snapshot_is_jpeg(live_state):
    port, _ = live_state
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot.jpg", timeout=5) as r:
        assert r.status == 200
        assert r.headers.get_content_type() == "image/jpeg"
        data = r.read()
    assert data[:2] == b"\xff\xd8" and len(data) > 1000


def test_dashboard_serves_html(live_state):
    port, _ = live_state
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
        html = r.read().decode()
    assert ("VIGLANCE" in html and "/stream.mjpg" in html
            and "/api/status" in html and "/api/upload" in html)


def test_ui_source_defaults_to_config_value():
    assert parse_args([]).source is None
    assert parse_args(["--source", "clip.avi"]).source == "clip.avi"


def test_threats_opt_in_not_default():
    args = parse_args([])
    assert args.threats is False and args.no_threats is False


def test_pair_payload_carries_readiness(live_state):
    port, _ = live_state
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as r:
        body = json.loads(r.read())
    assert len(body["pairs"]) >= 1
    for pair in body["pairs"]:
        assert set(pair) >= {"a", "b", "key", "score", "buf", "span", "ready", "dist"}
        assert isinstance(pair["ready"], bool)
        assert pair["span"] >= 0.0


def test_timeline_groups_contiguous_fight_frames():
    state = LiveState()
    state.record_event(2.0, "FIGHT", {})
    state.record_event(2.1, "FIGHT", {})
    state.record_event(2.2, "NO_FIGHT", {})
    assert state.timeline_snapshot() == [{
        "start": 2.0,
        "end": 2.1,
        "scene": "FIGHT",
        "alerts": [],
    }]


def test_upload_endpoint_accepts_avi(live_state):
    port, _ = live_state
    boundary = "----Viglance" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"video\"; filename=\"demo.avi\"\r\n"
        "Content-Type: video/x-msvideo\r\n\r\n"
        "not-a-real-video\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        payload = json.loads(response.read())
    assert payload["ok"] is True and payload["source"].endswith(".avi")


def test_unknown_route_404(live_state):
    port, _ = live_state
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("expected 404")
