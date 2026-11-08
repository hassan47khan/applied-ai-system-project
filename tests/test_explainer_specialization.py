"""
Locks down the specialization claims in model_card.md section 4.

These assert the *measurable differences* the model card reports, so the table there
cannot drift away from the code: the specialized explanation is grounded (names its
song, carries the scored reasons, ends with the retrieved card's guidance), the output
validator rejects replies that break the required style, and the historical Module 3
baseline really does lack that grounding.
"""

import pytest

from src.agent import VibeFinderAgent
from src.compare_explainers import (
    BASELINE_PREFS,
    _FakeClient,
    load_baseline_module,
)
from src.llm import MAX_EXPLANATION_WORDS, GeminiExplainer, TemplateExplainer
from src.recommender import load_songs
from src.retrieval import Card

STUDY_REQUEST = "I need something calm for studying late at night"


@pytest.fixture(scope="module")
def top_pick():
    return VibeFinderAgent(k=1).run(STUDY_REQUEST).recommendations[0]


# ---- specialized output is grounded -------------------------------------------------
def test_specialized_explanation_names_its_song(top_pick):
    assert top_pick.song["title"] in top_pick.explanation


def test_specialized_explanation_carries_the_scored_reasons(top_pick):
    """Every reason that moved the score has to appear, not a vague summary."""
    assert top_pick.reasons
    for reason in top_pick.reasons[:3]:
        assert reason in top_pick.explanation


def test_specialized_explanation_ends_with_card_guidance(top_pick):
    assert "Context (" in top_pick.explanation


def test_trade_offs_get_their_own_clause():
    card = Card(id="c", title="T", keywords=[], genres=[], moods=[], ranges={}, body="Guidance here.")
    song = {"title": "Storm Runner", "artist": "Voltline", "genre": "rock", "mood": "intense"}
    text = TemplateExplainer().explain(song, ["energy 0.91 fits target 0.92 (+1.47)",
                                              "acousticness 0.80 misses target 0.10 (-1.40)"], card)
    assert "Trade-off:" in text
    fits, tradeoff = text.split("Trade-off:")
    assert "misses" in tradeoff and "misses" not in fits
    assert "energy 0.91 fits" in fits


def test_llm_flag_without_a_key_falls_back_to_the_template(monkeypatch):
    from src.llm import make_explainer
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(make_explainer(True), TemplateExplainer)
    assert isinstance(make_explainer(False), TemplateExplainer)


# ---- the validator actually enforces the style --------------------------------------
def _explain_with_reply(reply, song, reasons):
    return GeminiExplainer(_FakeClient(reply), fallback=TemplateExplainer()).explain(
        song, reasons, None
    )


def test_validator_keeps_a_compliant_reply(top_pick):
    reply = f"{top_pick.song['title']} keeps the energy low and steady for late-night studying."
    assert _explain_with_reply(reply, top_pick.song, list(top_pick.reasons)) == reply


def test_validator_rejects_a_reply_that_omits_the_song_title(top_pick):
    reply = "This track keeps the energy low and steady for late-night studying."
    out = _explain_with_reply(reply, top_pick.song, list(top_pick.reasons))
    assert out != reply
    assert out == TemplateExplainer().explain(top_pick.song, list(top_pick.reasons), None)


def test_validator_rejects_a_reply_over_the_word_cap(top_pick):
    reply = top_pick.song["title"] + " is amazing" + " and wonderful" * MAX_EXPLANATION_WORDS
    assert len(reply.split()) > MAX_EXPLANATION_WORDS
    out = _explain_with_reply(reply, top_pick.song, list(top_pick.reasons))
    assert out != reply


def test_validator_cannot_catch_invented_facts(top_pick):
    """Documented limitation (model_card.md section 4), asserted so it stays honest."""
    reply = f"{top_pick.song['title']} won a Grammy in 2019 and was recorded in Reykjavik."
    assert _explain_with_reply(reply, top_pick.song, list(top_pick.reasons)) == reply


# ---- the historical baseline really is less grounded --------------------------------
def test_module3_baseline_explanation_lacks_grounding():
    baseline = load_baseline_module()
    if baseline is None:
        pytest.skip("git history unavailable")
    song, _score, explanation = baseline.recommend_songs(BASELINE_PREFS, load_songs(), k=1)[0]
    assert song["title"] not in explanation      # cannot be attributed to a song
    assert "Context (" not in explanation        # no listening guidance
    assert len(explanation.split()) < 15         # score arithmetic only
