"""Logging setup: console logging plus one log file per run."""
from __future__ import annotations

import logging
import os
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | None = None) -> None:
    root = logging.getLogger()
    lvl = getattr(logging, (level or os.getenv("PROMO_LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root.setLevel(lvl)
    if not any(getattr(h, "_promo_console", False) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(FORMAT))
        h._promo_console = True  # type: ignore[attr-defined]
        root.addHandler(h)
    for noisy in ("urllib3", "watchdog", "PIL", "pypdf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_run_file_handler(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter(FORMAT))
    h.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(h)
    return h


def remove_handler(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
