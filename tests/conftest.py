"""Test-suite environment, set before any test module imports src.

* Logs go to a temp dir, so neither the suite nor the CLI subprocesses some tests start
  append to logs/vibefinder.log.
* The suite runs against the original 18-song catalog and 13 context cards in
  tests/fixtures/. Many tests pin exact titles and scores from that small world; the real
  580-track catalog in data/ is exercised by the evaluation harness instead.
* Gemini is switched off and the key is blanked (so a developer's .env can't leak in):
  the suite never touches the network. Tests that need Gemini inject a fake client.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"

_LOG_DIR = tempfile.mkdtemp(prefix="vibefinder-test-logs-")
_CACHE_DIR = tempfile.mkdtemp(prefix="vibefinder-test-cache-")
os.environ["VIBEFINDER_LOG_DIR"] = _LOG_DIR
os.environ["VIBEFINDER_CACHE_DIR"] = _CACHE_DIR
os.environ["VIBEFINDER_SONGS"] = str(FIXTURES / "songs_small.csv")
os.environ["VIBEFINDER_KNOWLEDGE"] = str(FIXTURES / "knowledge")
os.environ["VIBEFINDER_INDEX"] = str(FIXTURES / "index")
os.environ["VIBEFINDER_RETRIEVAL"] = "tfidf"
os.environ["VIBEFINDER_USE_GEMINI"] = "0"
os.environ["VIBEFINDER_RERANK"] = "0"
os.environ["GEMINI_API_KEY"] = ""
os.environ["VIBEFINDER_NO_MEDIA"] = "1"     # no Deezer calls from tests


def pytest_unconfigure(config):
    logger = logging.getLogger("vibefinder")
    for handler in logger.handlers[:]:      # Windows can't delete a file that is still open
        handler.close()
        logger.removeHandler(handler)
    shutil.rmtree(_LOG_DIR, ignore_errors=True)
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)
