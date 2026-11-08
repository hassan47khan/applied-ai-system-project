"""Logging config: full detail to logs/vibefinder.log, warnings and up to the console.

Set VIBEFINDER_LOG_DIR to write the log somewhere else (the test suite does, so running
the tests never touches logs/).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .recommender import ROOT

LOG_DIR = Path(os.environ.get("VIBEFINDER_LOG_DIR") or ROOT / "logs")
LOG_FILE = LOG_DIR / "vibefinder.log"
LOG_MAX_BYTES = 1_000_000    # rotate at ~1 MB, keeping 3 old files, so the log can't grow forever
LOG_BACKUPS = 3


def setup_logging(console_level: int = logging.WARNING) -> None:
    root = logging.getLogger("vibefinder")
    if root.handlers:  # already configured (e.g. called twice in tests)
        return
    root.setLevel(logging.INFO)
    root.propagate = False

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not open %s; logging to console only", LOG_FILE)
