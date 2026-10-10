"""On-frame overlay tests: cv2 is the only dependency (skipped without it)."""

import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np

from core import visualization


class _Scene:
    def __init__(self, value):
        self.value = value


class _Verdict:
    def __init__(self, value):
        self.scene = _Scene(value)


def _frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_no_fight_draws_no_banner_on_frame():
    frame = _frame()
    visualization.draw_scene_panel(frame, _Verdict("NO_FIGHT"), None, plain=True)
    assert not frame.any()


def test_fight_verdicts_draw_a_banner():
    for verdict in ("FIGHT", "MULTI_PERSON_FIGHT", "GROUP_ASSAULT"):
        frame = _frame()
        visualization.draw_scene_panel(frame, _Verdict(verdict), None, plain=True)
        assert frame.any(), verdict


def test_no_reassuring_banner_exists_for_calm_verdict():
    assert "NO_FIGHT" not in visualization.PLAIN_SCENE_BANNERS
    assert "NO_FIGHT" in visualization.PLAIN_SCENE_NO_BANNER
