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


# How many tests the full suite collected, and whether this was a full run. Stashed on the
# pytest `config` object rather than in a module global: pytest can import conftest under two
# names ("conftest" and "tests.conftest"), and a test that imports it by name would get a
# second, empty copy. tests/test_docs_and_diagram.py reads this through the pytestconfig
# fixture to check the counts quoted in README.md, model_card.md and the architecture diagram
# against reality. Two commits in this repo's history exist only to correct a test count that
# had been written by hand; this makes the docs self-checking.

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(session, config, items):
    selected = bool(config.getoption("keyword") or config.getoption("markexpr"))
    paths = [a for a in config.args if not a.startswith("-")]
    whole_suite = not paths or all(
        Path(p).resolve() in (ROOT_DIR, TESTS_DIR) for p in paths
    )
    config.vibefinder_collected = {
        "count": len(items),
        "full_run": whole_suite and not selected,
    }
