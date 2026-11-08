"""Keep test runs out of the real logs/ folder.

Set before any test module imports src.log_setup, and inherited by the CLI subprocesses
some tests start, so neither the suite nor those processes append to logs/vibefinder.log.
"""

import logging
import os
import shutil
import tempfile

_LOG_DIR = tempfile.mkdtemp(prefix="vibefinder-test-logs-")
os.environ["VIBEFINDER_LOG_DIR"] = _LOG_DIR


def pytest_unconfigure(config):
    logger = logging.getLogger("vibefinder")
    for handler in logger.handlers[:]:      # Windows can't delete a file that is still open
        handler.close()
        logger.removeHandler(handler)
    shutil.rmtree(_LOG_DIR, ignore_errors=True)
