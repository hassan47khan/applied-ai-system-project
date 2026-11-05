"""Logging config: full detail to logs/vibefinder.log, warnings and up to the console."""

import logging
from pathlib import Path

from .recommender import ROOT

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "vibefinder.log"


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
        LOG_DIR.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not open %s; logging to console only", LOG_FILE)
