"""Outbound alert dispatch: Telegram incident notifications.

One notifier, one rule: a timeline incident notifies ONCE (keyed by its
start time + scene), with a global minimum gap between messages so a
flapping scene cannot spam a phone. Delivery failures never propagate —
they log and the pump keeps running.

Token hygiene: the bot token must come from the ``VIGILANCE_TELEGRAM_BOT_TOKEN``
environment variable (or an explicit CLI flag). It is never written to disk,
logs, or evidence files by this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import Request, urlopen

from core.logging_utils import get_logger

log = get_logger("alerts")

TELEGRAM_API = "https://api.telegram.org"
TIMEOUT_SEC = 10

FIGHT_SCENES = ("FIGHT", "MULTI_PERSON_FIGHT", "GROUP_ASSAULT")


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    cooldown_sec: float = 60.0
    notify_fight: bool = True
    notify_fall: bool = True
    notify_fire: bool = True
    notify_bag: bool = True
    notify_weapon: bool = True


@dataclass
class Incident:
    key: str            # start + scene: one notification per incident
    title: str
    scene: Optional[str]
    alerts: list = field(default_factory=list)
    timestamp: float = 0.0


def incident_from_state(scene: Optional[str], alerts: dict,
                        timestamp: float) -> Optional[Incident]:
    """Decide whether the current frame state deserves a notification."""
    names = sorted(alerts or {})
    fight = scene is not None and scene in FIGHT_SCENES
    if not fight and not names:
        return None
    if fight:
        title = {"FIGHT": "Possible fight detected",
                 "MULTI_PERSON_FIGHT": "Group fight detected",
                 "GROUP_ASSAULT": "Group assault detected"}.get(scene, "Fight detected")
    else:
        title = "Alert: " + ", ".join(names)
    key = f"{round(float(timestamp), 3)}:{scene or ','.join(names)}"
    return Incident(key=key, title=title, scene=scene,
                    alerts=names, timestamp=float(timestamp))


class TelegramNotifier:
    def __init__(self, config: TelegramConfig):
        self.config = config
        self._notified_keys: set = set()
        self._last_sent: float = 0.0

    @property
    def ready(self) -> bool:
        return bool(self.config.enabled and self.config.bot_token
                    and self.config.chat_id)

    def _wanted(self, incident: Incident) -> bool:
        if incident.scene in FIGHT_SCENES:
            return self.config.notify_fight
        names = set(incident.alerts)
        if any("Fall" in n for n in names):
            return self.config.notify_fall
        if any("Fire" in n or "Smoke" in n for n in names):
            return self.config.notify_fire
        if any("Bag" in n or "bag" in n for n in names):
            return self.config.notify_bag
        if any("Weapon" in n for n in names):
            return self.config.notify_weapon
        return True

    def _post(self, method: str, payload: dict) -> dict:
        url = f"{TELEGRAM_API}/bot{self.config.bot_token}/{method}"
        req = Request(url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode())

    def _send(self, incident: Incident, jpeg: Optional[bytes]) -> bool:
        caption = (f"🚨 VIGLANCE: {incident.title}\n"
                   f"t={incident.timestamp:.1f}s")
        if incident.alerts:
            caption += "\n" + ", ".join(incident.alerts)
        if jpeg:
            return self._send_photo(incident, caption, jpeg)
        result = self._post("sendMessage", {
            "chat_id": self.config.chat_id, "text": caption})
        return bool(result.get("ok", False))

    def _send_photo(self, incident: Incident, caption: str, jpeg: bytes) -> bool:
        boundary = "viglance-boundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="chat_id"\r\n\r\n{self.config.chat_id}\r\n'
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="caption"\r\n\r\n{caption}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; '
            f'filename="alert.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode() + jpeg + f"\r\n--{boundary}--\r\n".encode()
        url = f"{TELEGRAM_API}/bot{self.config.bot_token}/sendPhoto"
        req = Request(url, data=body,
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urlopen(req, timeout=TIMEOUT_SEC) as resp:
            return bool(json.loads(resp.read().decode()).get("ok", False))

    def maybe_notify(self, incident: Optional[Incident],
                     jpeg: Optional[bytes] = None,
                     now: Optional[float] = None) -> bool:
        """Send at most once per incident; never raises."""
        if incident is None or not self.ready or not self._wanted(incident):
            return False
        if incident.key in self._notified_keys:
            return False
        now = time.monotonic() if now is None else now
        if now - self._last_sent < self.config.cooldown_sec:
            log.info("event=telegram_cooldown_skip key=%s", incident.key)
            return False
        try:
            sent = self._send(incident, jpeg)
        except Exception as exc:
            log.warning("event=telegram_send_failed key=%s error=%r",
                        incident.key, exc)
            return False
        if sent:
            self._notified_keys.add(incident.key)
            self._last_sent = now
            log.info("event=telegram_sent key=%s title=%s",
                     incident.key, incident.title)
        return sent
