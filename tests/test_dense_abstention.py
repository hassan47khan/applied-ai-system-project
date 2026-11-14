"""Regression tests for the dense-retrieval abstention bug.

The bug: on the Gemini path the agent stopped refusing. Re-running the harness with only
the retriever changed showed lexical retrieval refused correctly and dense retrieval did
not. Two causes, each covered here:

1. **An empty query was embedded.** "anything but jazz" retrieves on the text left once the
   rejection is removed, which is "". BM25 returns nothing for "", but the dense arm embedded
   it, found it "similar" to every card, and the retrieved cards' genres were counted as the
   user's preferences, so `exclusion_only` could never fire.

2. **Dense evidence was an absolute cosine.** Short and generic text is uniformly similar to
   every card in a 768-d space (measured: "" scores 0.66 against its best card and
   "good songs" 0.72, both above real requests such as "debugging a nasty bug all day" at
   0.62). Dense now scores a card by how far it stands out from the query's mean similarity
   to all cards (centred cosine). That keeps the card *order* identical and changes only
   which cards clear the floor.
"""

from __future__ import annotations

import logging
import math

import pytest

from src.agent import VibeFinderAgent
from src.embeddings import load_calibration, save_calibration
from src.llm import TemplateExplainer
from src.retrieval import (
    BM25Retriever, Calibration, DenseRetriever, HybridRetriever, has_content, load_cards,
)
from tests.fakes import concept_vector

CENTRED = Calibration(0.0635, 0.1382, signal="centered")


def _unit(vec):
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


@pytest.fixture()
def fixture_cards():
    return load_cards()          # conftest points this at the 13-card fixture world


def _dense(cards, embed=concept_vector):
    vectors = {c.id: concept_vector(" ".join(c.keywords)) for c in cards}
    return DenseRetriever(cards, vectors, embed, calibration=CENTRED)


# ---- cause 1: never embed an empty query ------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\t\n", "?!", "--"])
def test_has_content_is_false_for_nothing_to_retrieve_on(text):
    assert not has_content(text)


@pytest.mark.parametrize("text", ["a", "jazz", "música", "5k", "Lo-fi"])
def test_has_content_is_true_for_real_text(text):
    assert has_content(text)


@pytest.mark.parametrize("text", ["", "   ", "?!"])
def test_dense_never_embeds_an_empty_query(fixture_cards, text):
    calls = []

    def spy(q):
        calls.append(q)
        return concept_vector(q)

    assert _dense(fixture_cards, embed=spy).retrieve(text) == []
    assert calls == [], "an empty query must not reach the embedding model at all"


def test_hybrid_returns_nothing_for_an_empty_query(fixture_cards):
    hybrid = HybridRetriever(BM25Retriever(fixture_cards), _dense(fixture_cards))
    assert hybrid.retrieve("") == []


@pytest.mark.parametrize("request_text", ["anything but jazz", "no jazz", "not happy"])
def test_exclusion_only_request_is_flagged_with_dense_retrieval(fixture_cards, request_text):
    """The end-to-end regression: this answered unflagged at confidence 0.65 before the fix."""
    hybrid = HybridRetriever(BM25Retriever(fixture_cards), _dense(fixture_cards))
    agent = VibeFinderAgent(cards=fixture_cards, retriever=hybrid, understander=None,
                            explainer=TemplateExplainer())
    result = agent.run(request_text)
    assert result.retrieved == [], "no context card should be retrieved for an exclusion-only request"
    assert "exclusion_only" in result.issues
    assert result.needs_review


# ---- cause 2: evidence must mean "this card stands out", not "this cosine is high" -------

def test_a_query_equally_similar_to_every_card_is_not_evidence(fixture_cards):
    """All cosines 1.0: an absolute floor would accept every card; centring accepts none."""
    same = _unit([1.0, 2.0, 3.0])
    dense = DenseRetriever(fixture_cards, {c.id: same for c in fixture_cards},
                           lambda q: same, calibration=CENTRED)
    assert all(s == pytest.approx(1.0) for _, s in dense.cosine_scores("anything"))
    assert dense.retrieve("anything") == []


def test_a_card_that_stands_out_is_still_retrieved(fixture_cards):
    target, rest = fixture_cards[0], fixture_cards[1:]
    vectors = {c.id: _unit([0.0, 1.0, 0.2]) for c in rest}
    vectors[target.id] = _unit([1.0, 0.0, 0.2])
    dense = DenseRetriever(fixture_cards, vectors, lambda q: _unit([1.0, 0.0, 0.2]),
                           calibration=CENTRED)
    hits = dense.retrieve("anything", k=3)
    assert hits and hits[0][0].id == target.id
    assert len(hits) == 1, "only the card that stands out clears the floor"


def test_centring_is_a_per_query_shift_so_the_ranking_is_unchanged(fixture_cards):
    """Centring must change which cards clear the floor, never the order of the cards."""
    dense = _dense(fixture_cards)
    for query in ("rainy sad afternoon", "gym workout", "feeling bummed and lonely", "study exam"):
        raw = dict((c.id, s) for c, s in dense.cosine_scores(query))
        centred = dict((c.id, s) for c, s in dense.native_scores(query))
        shifts = {round(raw[cid] - centred[cid], 12) for cid in raw}
        assert len(shifts) == 1, f"centring was not a constant shift for {query!r}"
        assert sorted(raw, key=lambda c: (-raw[c], c)) == sorted(centred, key=lambda c: (-centred[c], c))


# ---- calibration units must match the score they are applied to ---------------------------

def test_dense_refuses_a_calibration_fitted_on_raw_cosine(fixture_cards, caplog):
    vectors = {c.id: concept_vector(" ".join(c.keywords)) for c in fixture_cards}
    with caplog.at_level(logging.WARNING, logger="vibefinder.retrieval"):
        dense = DenseRetriever(fixture_cards, vectors, concept_vector,
                               calibration=Calibration(0.621, 0.7189))     # the old cosine fit
    assert dense.calibration == DenseRetriever.DEFAULT_CALIBRATION
    assert "python -m src.index calibrate" in caplog.text


def test_calibration_unit_round_trips_through_the_file(tmp_path):
    save_calibration({"bm25": {"floor": 0.1, "full": 0.5, "signal": "native"},
                      "dense": {"floor": 0.06, "full": 0.14, "signal": "centered"}}, tmp_path)
    loaded = load_calibration(tmp_path)
    assert loaded["dense"] == Calibration(0.06, 0.14, "centered")
    assert loaded["bm25"].signal == "native"


def test_a_calibration_file_without_a_unit_is_read_as_native(tmp_path):
    """Files written before the unit existed were raw cosine; they must not pass as centred."""
    save_calibration({"dense": {"floor": 0.621, "full": 0.7189}}, tmp_path)
    assert load_calibration(tmp_path)["dense"].signal == "native"
