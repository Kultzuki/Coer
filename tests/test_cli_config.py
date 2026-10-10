"""CLI/config precedence tests: unset flags must never clobber file values."""

import json

import pytest

from config import VigilanceConfig


def _write_config(tmp_path, **overrides):
    cfg = {"source": "video.mp4", "log_level": "DEBUG"}
    cfg.update(overrides)
    path = tmp_path / "vig.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_config_file_source_survives_empty_cli(tmp_path):
    from main import build_config, parse_args
    args = parse_args(["--config", _write_config(tmp_path)])
    cfg = build_config(args)
    assert cfg.source == "video.mp4"
    assert cfg.log_level == "DEBUG"


def test_explicit_cli_still_wins(tmp_path):
    from main import build_config, parse_args
    args = parse_args(["--config", _write_config(tmp_path),
                       "--source", "other.mp4", "--log-level", "ERROR"])
    cfg = build_config(args)
    assert cfg.source == "other.mp4"
    assert cfg.log_level == "ERROR"


def test_no_config_falls_back_to_defaults():
    from main import build_config, parse_args
    cfg = build_config(parse_args([]))
    assert cfg.source == VigilanceConfig().source == "0"
    assert cfg.log_level == "INFO"


def test_legacy_fight_flag_and_env(tmp_path, monkeypatch):
    from main import build_config, parse_args
    cfg = build_config(parse_args([]))
    assert cfg.enable_legacy_fight is False
    cfg = build_config(parse_args(["--legacy-fight"]))
    assert cfg.enable_legacy_fight is True
    monkeypatch.setenv("VIGILANCE_ENABLE_LEGACY_FIGHT", "1")
    cfg = build_config(parse_args([]))
    assert cfg.enable_legacy_fight is True

@pytest.mark.parametrize("path", ["config/default.json"])
def test_default_config_is_violence_fire_fall_only(path):
    cfg = VigilanceConfig.from_file(path)
    cfg.validate()
    assert cfg.enable_fight is True
    assert cfg.enable_fire is True
    assert cfg.enable_fall is True
    assert cfg.enable_bag is False
    assert cfg.enable_legacy_fight is False
    assert cfg.enable_threats is False


def test_violence_fire_fall_pipeline_registers_no_other_scenarios():
    from core.pipeline import Pipeline
    from tests.conftest import FakeDetector
    pipe = Pipeline(config=VigilanceConfig(), detector=FakeDetector([[]]))
    assert sorted(name for name, _, _ in pipe._scenarios) == ["Fall Detection"]

def test_fire_enabled_by_default_with_env_override(monkeypatch):
    from config import VigilanceConfig
    assert VigilanceConfig().enable_fire is True
    monkeypatch.setenv("VIGILANCE_ENABLE_FIRE", "0")
    assert VigilanceConfig().from_env().enable_fire is False
