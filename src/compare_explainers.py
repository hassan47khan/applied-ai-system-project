"""
Baseline vs. specialized explanation comparison (backs model_card.md section 4).

Three things are measured, all offline and deterministic:

  1. BASELINE   - the Module 3 explanation string, run from the actual historical
                  commit (12213e3) rather than re-typed from memory. It is
                  `"; ".join(reasons)`: score arithmetic, no prose, no grounding.
  2. SPECIALIZED - VibeFinder 2.0's constrained explainer: a fixed sentence shape
                  grounded in the score reasons plus the retrieved card's guidance.
  3. VALIDATOR  - what the constraint layer in src/llm.py actually rejects when an
                  LLM reply does not match the required style. Uses a stub client,
                  so no API key and no network are needed.

Run:
    python -m src.compare_explainers
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from .agent import VibeFinderAgent
from .llm import MAX_EXPLANATION_WORDS, GeminiExplainer, TemplateExplainer
from .recommender import load_songs

BASELINE_COMMIT = "12213e3"
BASELINE_PATH = "src/recommender.py"
BASELINE_PREFS = {"genre": "lofi", "mood": "chill", "energy": 0.25}
REQUEST = "I need something calm for studying late at night"


# --------------------------------------------------------------------- helpers
def load_baseline_module():
    """Load the Module 3 scorer straight out of git history."""
    repo_root = Path(__file__).resolve().parent.parent
    try:
        source = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{BASELINE_PATH}"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  (skipped: could not read {BASELINE_COMMIT}:{BASELINE_PATH} from git - {exc})")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline_recommender.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("baseline_recommender", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, text: str):
        self._text = text

    def generate_content(self, model, contents):  # noqa: ARG002 - stub signature
        return _FakeResponse(self._text)


class _FakeClient:
    """Stands in for google-genai so the validator can be exercised offline."""

    def __init__(self, text: str):
        self.models = _FakeModels(text)


def _words(text: str) -> int:
    return len(text.split())


# ----------------------------------------------------------------------- parts
def show_baseline(songs: List[Dict]) -> None:
    print("1. BASELINE - Module 3 explanation string (commit "
          f"{BASELINE_COMMIT}, prefs={BASELINE_PREFS})")
    baseline = load_baseline_module()
    if baseline is None:
        return
    song, _score, explanation = baseline.recommend_songs(BASELINE_PREFS, songs, k=1)[0]
    print(f"   top song : {song['title']} [{song['genre']}]")
    print(f"   text     : {explanation}")
    print(f"   words    : {_words(explanation)}")
    print("   grounding: score arithmetic only - no song name, no prose, no listening guidance")
    print()


def show_specialized():
    print(f"2. SPECIALIZED - VibeFinder 2.0 constrained explainer (request: {REQUEST!r})")
    result = VibeFinderAgent(k=1).run(REQUEST)
    top = result.recommendations[0]
    print(f"   top song : {top.song['title']} [{top.song['genre']}]")
    print(f"   text     : {top.explanation}")
    print(f"   words    : {_words(top.explanation)}")
    print("   grounding: names the song, cites every scored reason, ends with the "
          "retrieved card's guidance")
    print()
    return top


def show_validator(top) -> None:
    song, reasons = top.song, list(top.reasons)
    title = song["title"]
    candidates = {
        "compliant: short and names the song":
            f"{title} keeps the energy low and steady, which suits late-night studying.",
        "omits the song title":
            "This track keeps the energy low and steady, which suits late-night studying.",
        "over-long: exceeds the word cap":
            (f"{title} is an absolutely breathtaking masterpiece that has taken the world by "
             "storm, beloved by millions of listeners across every continent, and it is widely "
             "considered one of the greatest achievements in the history of recorded music, a "
             "timeless classic that will surely be remembered for generations to come, so press "
             "play right now and let it change your entire life forever and ever, because nothing "
             "else you hear this year will ever come even remotely close to matching the sheer "
             "brilliance and emotional depth on display throughout every single moment."),
        "invents facts but stays short and names the song":
            f"{title} won a Grammy in 2019 and was recorded in Reykjavik over three years.",
    }

    print(f"3. VALIDATOR - constraints in src/llm.py (max {MAX_EXPLANATION_WORDS} words, "
          "must name the song)")
    for label, text in candidates.items():
        out = GeminiExplainer(_FakeClient(text), fallback=TemplateExplainer()).explain(
            song, reasons, None
        )
        verdict = "KEPT" if out == text else "REJECTED -> template fallback"
        print(f"   [{_words(text):>3} words] {label}: {verdict}")
    print()
    print("   Note: the last case is KEPT. The validator enforces length and grounding in the")
    print("   song title; it cannot detect invented claims. That is why the LLM is optional and")
    print("   the offline template is the default (see model_card.md section 6).")


def main() -> int:
    songs = load_songs()
    print("=" * 78)
    print("  Explanation specialization: baseline vs. constrained output")
    print("=" * 78)
    print()
    show_baseline(songs)
    top = show_specialized()
    show_validator(top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
