"""Telegram dispatcher tests: no network, transport faked."""

import io
import json

import pytest

from core.alert_dispatch import (TelegramConfig, TelegramNotifier,
                                 incident_from_state)


def _cfg(**kw):
    base = dict(enabled=True, bot_token="TOKEN", chat_id="123",
                cooldown_sec=60.0)
    base.update(kw)
    return TelegramConfig(**base)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_transport(monkeypatch, calls, ok=True):
    import core.alert_dispatch as ad

    def fake(req, timeout=None):
        calls.append((req.full_url, req.data[:60] if req.data else b""))
        return _Resp({"ok": ok})

    monkeypatch.setattr(ad, "urlopen", fake)


def test_incident_from_state():
    assert incident_from_state("NO_FIGHT", {}, 1.0) is None
    assert incident_from_state(None, {}, 1.0) is None
    inc = incident_from_state("GROUP_ASSAULT", {}, 2.5)
    assert inc.title == "Group assault detected" or "Group" in inc.title
    inc2 = incident_from_state(None, {"Fall Detection": {}}, 3.0)
    assert "Fall" in inc2.title


def test_wanted_flags():
    n = TelegramNotifier(_cfg(notify_fire=False))
    assert n._wanted(incident_from_state("FIGHT", {}, 0.0))
    assert not n._wanted(incident_from_state(None, {"Fire Threat": {}}, 0.0))
    assert n._wanted(incident_from_state(None, {"Fall Detection": {}}, 0.0))


def test_sends_once_per_incident(monkeypatch):
    calls = []
    _fake_transport(monkeypatch, calls)
    n = TelegramNotifier(_cfg(cooldown_sec=0))
    inc = incident_from_state("FIGHT", {}, 10.0)
    assert n.maybe_notify(inc, jpeg=None, now=100.0) is True
    assert n.maybe_notify(inc, jpeg=None, now=101.0) is False
    assert len(calls) == 1
    assert "sendMessage" in calls[0][0]


def test_cooldown_between_incidents(monkeypatch):
    calls = []
    _fake_transport(monkeypatch, calls)
    n = TelegramNotifier(_cfg(cooldown_sec=60.0))
    assert n.maybe_notify(incident_from_state("FIGHT", {}, 1.0), now=100.0)
    assert not n.maybe_notify(incident_from_state("FIGHT", {}, 2.0), now=130.0)
    assert n.maybe_notify(incident_from_state("FIGHT", {}, 3.0), now=161.0)
    assert len(calls) == 2


def test_photo_path_used_with_jpeg(monkeypatch):
    calls = []
    _fake_transport(monkeypatch, calls)
    n = TelegramNotifier(_cfg(cooldown_sec=0))
    assert n.maybe_notify(incident_from_state("FIGHT", {}, 1.0),
                          jpeg=b"\xff\xd8fake", now=0.0)
    assert "sendPhoto" in calls[0][0]


def test_not_ready_never_sends(monkeypatch):
    calls = []
    _fake_transport(monkeypatch, calls)
    n = TelegramNotifier(_cfg(enabled=False))
    assert not n.ready
    assert n.maybe_notify(incident_from_state("FIGHT", {}, 1.0)) is False
    n2 = TelegramNotifier(_cfg(bot_token=""))
    assert not n2.ready
    assert calls == []


def test_network_error_never_raises(monkeypatch):
    import core.alert_dispatch as ad

    def boom(req, timeout=None):
        raise OSError("offline")

    monkeypatch.setattr(ad, "urlopen", boom)
    n = TelegramNotifier(_cfg(cooldown_sec=0))
    assert n.maybe_notify(incident_from_state("FIGHT", {}, 1.0)) is False
