"""VibeFinder 3.0 retrieval: BM25, dense, RRF + MMR hybrid, the index, calibration, rerank.

Runs fully offline: dense retrieval uses tests/fakes.py, whose embeddings put synonyms on
the same axis, so these tests can check that the hybrid recovers what lexical misses.
"""

import json
import math

import pytest

from src.config import Settings
from src.embeddings import (
    build_index, card_sha, load_calibration, load_index, make_retriever, save_calibration,
)
from src.gemini import GeminiUnavailable
from src.rerank import RerankingRetriever
from src.retrieval import (
    BM25Retriever, Calibration, DenseRetriever, EVIDENCE_FLOOR, EVIDENCE_FULL, HybridRetriever,
    load_cards, mmr_select, parse_card, reciprocal_rank_fusion,
)
from src.retrieval_eval import (
    evaluate_retriever, fit_calibration, ndcg_at_k, recall_at_k, reciprocal_rank,
)
from tests.fakes import FakeSDK, concept_vector, fake_client


def write_card(directory, cid, keywords, body="Some guidance here.", title=None):
    path = directory / f"{cid}.md"
    path.write_text(f"---\nid: {cid}\ntitle: {title or cid}\nkeywords: {keywords}\n---\n{body}\n",
                    encoding="utf-8")
    return parse_card(path)


@pytest.fixture()
def cards(tmp_path):
    return [
        write_card(tmp_path, "sad", "rain, rainy, sad, cry", "Grey days pair with soft songs."),
        write_card(tmp_path, "gym", "gym, workout, lifting, weights", "Loud music for hard training."),
        write_card(tmp_path, "study", "study, homework, exam, library", "Quiet music for focus."),
        write_card(tmp_path, "cook", "cooking, kitchen, recipe", "Groovy feel-good music for the counter."),
    ]


# ---- BM25 ----------------------------------------------------------------------------
def test_bm25_ranks_the_matching_card_first(cards):
    hits = BM25Retriever(cards).retrieve("rainy sad afternoon", k=3)
    assert hits[0][0].id == "sad"


def test_bm25_refuses_off_topic_text(cards):
    assert BM25Retriever(cards).retrieve("asdf qwerty zxcv") == []


def test_bm25_discounts_a_match_found_only_in_the_body(cards):
    bm25 = BM25Retriever(cards)
    body_only = dict((c.id, s) for c, s in bm25.native_scores("good"))["cook"]
    keyword = dict((c.id, s) for c, s in bm25.native_scores("kitchen"))["cook"]
    assert 0 < body_only < keyword * 0.5


def test_bm25_scores_are_normalised_to_at_most_one(cards):
    for _, score in BM25Retriever(cards).native_scores("gym workout lifting weights"):
        assert 0.0 <= score <= 1.0


def test_calibration_maps_floor_and_full_onto_the_agent_scale():
    cal = Calibration(floor=0.2, full=0.6)
    assert cal.to_evidence(0.2) == pytest.approx(EVIDENCE_FLOOR)
    assert cal.to_evidence(0.6) == pytest.approx(EVIDENCE_FULL)
    assert cal.to_evidence(0.0) >= 0.0


# ---- fusion and diversity ---------------------------------------------------------------
def test_rrf_rewards_documents_ranked_well_by_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == fused["b"] > fused["c"] and fused["c"] == fused["d"]


def test_mmr_skips_a_near_duplicate():
    relevance = {"a": 1.0, "a2": 0.99, "b": 0.6}
    same = {frozenset({"a", "a2"}): 1.0}
    chosen = mmr_select(["a", "a2", "b"], relevance, lambda x, y: same.get(frozenset({x, y}), 0.0),
                        k=2, lam=0.5)
    assert chosen == ["a", "b"]


def test_mmr_with_full_relevance_weight_keeps_relevance_order():
    relevance = {"a": 1.0, "a2": 0.99, "b": 0.6}
    assert mmr_select(["a", "a2", "b"], relevance, lambda x, y: 1.0, k=3, lam=1.0) == ["a", "a2", "b"]


# ---- dense and hybrid ----------------------------------------------------------------
def dense_for(cards, embed=None):
    vectors = {c.id: concept_vector(" ".join(c.keywords)) for c in cards}
    return DenseRetriever(cards, vectors, embed or concept_vector, calibration=Calibration(0.5, 0.9))


def test_dense_finds_a_synonym_lexical_misses(cards):
    assert BM25Retriever(cards).retrieve("feeling bummed and lonely") == []
    assert dense_for(cards).retrieve("feeling bummed and lonely")[0][0].id == "sad"


def test_hybrid_uses_both_arms(cards):
    hybrid = HybridRetriever(BM25Retriever(cards), dense_for(cards))
    assert hybrid.retrieve("bummed out")[0][0].id == "sad"
    assert hybrid.retrieve("gym workout")[0][0].id == "gym"
    assert hybrid.last_info == {"mode": "hybrid"}


def test_hybrid_falls_back_to_lexical_and_says_why(cards):
    def broken(_):
        raise GeminiUnavailable("ClientError 429")
    hybrid = HybridRetriever(BM25Retriever(cards), dense_for(cards, embed=broken))
    assert hybrid.retrieve("gym workout")[0][0].id == "gym"
    assert hybrid.last_info == {"mode": "lexical", "fallback": "ClientError 429"}


def test_hybrid_without_a_dense_arm_is_the_lexical_retriever(cards):
    hybrid = HybridRetriever(BM25Retriever(cards), None)
    assert [c.id for c, _ in hybrid.retrieve("homework exam")] == ["study"]
    assert hybrid.last_info["fallback"] == "no embedding index"


def test_hybrid_reports_the_best_calibrated_evidence(cards):
    bm25 = BM25Retriever(cards)
    hybrid = HybridRetriever(bm25, dense_for(cards))
    lexical = dict((c.id, s) for c, s in bm25.retrieve("gym workout", k=8))
    fused = dict((c.id, s) for c, s in hybrid.retrieve("gym workout"))
    assert fused["gym"] >= lexical["gym"]


# ---- index ---------------------------------------------------------------------------
def test_index_build_then_load_round_trips(tmp_path, cards):
    client = fake_client(tmp_path)
    build_index(cards, client, tmp_path / "index")
    index = load_index(cards, tmp_path / "index")
    assert index.usable and set(index.vectors) == {c.id for c in cards}
    assert index.model == "gemini-embedding-2"
    assert all(math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-4) for vec in index.vectors.values())


def test_index_detects_a_card_edited_after_the_build(tmp_path, cards):
    build_index(cards, fake_client(tmp_path), tmp_path / "index")
    edited = write_card(tmp_path, "gym", "gym, workout, lifting, weights", "Different guidance now.")
    index = load_index([c if c.id != "gym" else edited for c in cards], tmp_path / "index")
    assert index.stale == ["gym"] and not index.usable


def test_index_reports_a_card_with_no_vector(tmp_path, cards):
    build_index(cards[:2], fake_client(tmp_path), tmp_path / "index")
    assert set(load_index(cards, tmp_path / "index").missing) == {"study", "cook"}


def test_missing_index_loads_as_none(tmp_path, cards):
    assert load_index(cards, tmp_path / "nowhere") is None


def test_card_sha_changes_with_the_text(tmp_path):
    a = write_card(tmp_path, "x", "one, two", "Body A.")
    b = write_card(tmp_path, "x", "one, two", "Body B.")
    assert card_sha(a) != card_sha(b)


def test_embeddings_are_cached_on_disk(tmp_path, cards):
    sdk = FakeSDK()
    client = fake_client(tmp_path, sdk)
    client.embed(["hello"], kind="query")
    client.embed(["hello"], kind="query")
    assert len(sdk.embed_calls) == 1 and client.stats.cache_hits == 1


def test_embedding_2_gets_the_task_prefix_and_one_content_per_text(tmp_path):
    sdk = FakeSDK()
    fake_client(tmp_path, sdk).embed(["a", "b"], kind="query")
    contents = sdk.embed_calls[0]["contents"]
    assert len(contents) == 2
    assert contents[0].parts[0].text == "task: search result | query: a"


# ---- factory -------------------------------------------------------------------------
def test_factory_downgrades_and_explains_when_there_is_no_index(tmp_path, cards):
    settings = Settings(index_dir=tmp_path / "none", cache_dir=tmp_path / "c", api_key="k", use_gemini=True)
    retriever, notes = make_retriever(cards, mode="hybrid", settings=settings, client=fake_client(tmp_path))
    assert isinstance(retriever, HybridRetriever) and retriever.dense is None
    assert any("no embedding index" in n for n in notes)


def test_factory_builds_a_working_hybrid_with_an_index(tmp_path, cards):
    client = fake_client(tmp_path)
    build_index(cards, client, tmp_path / "index")
    settings = Settings(index_dir=tmp_path / "index", cache_dir=tmp_path / "c")
    retriever, notes = make_retriever(cards, mode="hybrid", settings=settings, client=client)
    assert retriever.dense is not None and notes == []


def test_factory_offline_modes_need_no_client(tmp_path, cards):
    settings = Settings(index_dir=tmp_path, cache_dir=tmp_path)
    assert make_retriever(cards, mode="bm25", settings=settings)[0].name == "bm25"
    assert type(make_retriever(cards, mode="tfidf", settings=settings)[0]).__name__ == "Retriever"


def test_calibration_file_round_trips_and_ignores_garbage(tmp_path):
    save_calibration({"bm25": {"floor": 0.1, "full": 0.5}, "dense": {"floor": "x"}}, tmp_path)
    loaded = load_calibration(tmp_path)
    assert loaded == {"bm25": Calibration(0.1, 0.5)}


# ---- metrics and calibration fitting ----------------------------------------------
def test_ranking_metrics():
    assert recall_at_k(["a", "x", "b"], ["a", "b"]) == 1.0
    assert recall_at_k(["x", "y", "z", "a"], ["a"]) == 0.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert ndcg_at_k(["a"], ["a"]) == 1.0
    assert ndcg_at_k(["x", "a"], ["a"]) == pytest.approx(1 / math.log2(3))


def test_evaluate_retriever_counts_abstentions_and_kinds():
    queries = [{"q": "hit", "relevant": ["a"], "kind": "direct"},
               {"q": "miss", "relevant": ["a"], "kind": "synonym"},
               {"q": "noise", "relevant": [], "kind": "off_topic"}]
    answers = {"hit": ["a"], "miss": ["b"], "noise": []}
    rep = evaluate_retriever("t", lambda q: answers[q], queries)
    assert rep.hit1 == 0.5 and rep.abstain == 1.0
    assert rep.by_kind == {"direct": 1.0, "synonym": 0.0}
    assert rep.misses == [("miss", ["b"], ["a"])]


def test_fit_calibration_separates_positives_from_negatives():
    queries = [{"q": f"p{i}", "relevant": ["a"]} for i in range(4)] + \
              [{"q": f"n{i}", "relevant": []} for i in range(4)]
    scores = {"p0": 0.6, "p1": 0.7, "p2": 0.8, "p3": 0.9, "n0": 0.1, "n1": 0.2, "n2": 0.3, "n3": 0.35}
    fit = fit_calibration(lambda q: {"a": scores[q]}, queries)
    assert 0.35 < fit["floor"] < 0.6 and fit["balanced_accuracy"] == 1.0
    assert fit["full"] == pytest.approx(0.75)


def test_committed_calibration_uses_only_held_out_queries():
    gold = {q["q"] for q in json.load(open("data/eval/retrieval_gold.json", encoding="utf-8"))["queries"]}
    held = {q["q"] for q in json.load(open("data/eval/calibration_queries.json", encoding="utf-8"))["queries"]}
    assert not gold & held


# ---- rerank --------------------------------------------------------------------------
def test_rerank_reorders_and_drops_low_ratings(tmp_path, cards):
    sdk = FakeSDK(json_replies={"ratings": [{"id": "cook", "score": 9}, {"id": "study", "score": 2}]})
    base = BM25Retriever(cards)
    rr = RerankingRetriever(base, fake_client(tmp_path, sdk))
    hits = rr.retrieve("homework in the kitchen", k=3)
    assert [c.id for c, _ in hits] == ["cook"]
    assert rr.last_info["mode"].endswith("+rerank")


def test_rerank_ignores_ids_it_was_not_given(tmp_path, cards):
    sdk = FakeSDK(json_replies={"ratings": [{"id": "evil", "score": 10}, {"id": "gym", "score": 8}]})
    hits = RerankingRetriever(BM25Retriever(cards), fake_client(tmp_path, sdk)).retrieve("gym workout")
    assert [c.id for c, _ in hits] == ["gym"]


def test_rerank_failure_keeps_the_base_ranking(tmp_path, cards):
    sdk = FakeSDK(errors=[ValueError("boom")])
    rr = RerankingRetriever(BM25Retriever(cards), fake_client(tmp_path, sdk))
    assert rr.retrieve("gym workout")[0][0].id == "gym"
    assert "rerank_fallback" in rr.last_info


# ---- real data integrity -------------------------------------------------------------
def test_real_knowledge_cards_only_name_catalog_genres_and_moods():
    import csv
    from src.config import ROOT
    rows = list(csv.DictReader(open(ROOT / "data" / "songs.csv", encoding="utf-8")))
    genres, moods = {r["genre"] for r in rows}, {r["mood"] for r in rows}
    real_cards = load_cards(ROOT / "data" / "knowledge")
    assert len(real_cards) >= 30
    for card in real_cards:
        assert set(card.genres) <= genres, card.id
        assert set(card.moods) <= moods, card.id


def test_gold_labels_only_reference_real_cards():
    from src.config import ROOT
    ids = {c.id for c in load_cards(ROOT / "data" / "knowledge")}
    for name in ("retrieval_gold.json", "calibration_queries.json"):
        for item in json.load(open(ROOT / "data" / "eval" / name, encoding="utf-8"))["queries"]:
            assert set(item["relevant"]) <= ids, item
