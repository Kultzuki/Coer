"""Structured logging for VIGLANCE. Replaces all print() statements.

Usage:
    from core.logging_utils import get_logger, setup_logging
    setup_logging("INFO")
    log = get_logger(__name__)
    log.info("event=frame_processed frame=12 persons=2")
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO", stream=None) -> logging.Logger:
    """Configure the root ``vigilance`` logger once; safe to call repeatedly."""
    global _CONFIGURED
    root = logging.getLogger("vigilance")
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(lvl)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(handler)
        _CONFIGURED = True
    else:
        root.setLevel(lvl)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vigilance.{name}")
