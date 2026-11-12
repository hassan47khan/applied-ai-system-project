"""
Retrieval metrics and score calibration, shared by `src.evaluate` and `src.index`.

Metrics over a labelled query set (data/eval/retrieval_gold.json):
  Recall@3   share of a query's relevant cards found in the top 3 (averaged over on-topic queries)
  Hit@1      the top card is relevant
  MRR        1 / rank of the first relevant card
  nDCG@3     rank-discounted gain, binary relevance
  Abstain    off-topic queries for which the retriever correctly returned nothing
"""

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from .config import ROOT

GOLD_PATH = ROOT / "data" / "eval" / "retrieval_gold.json"
CALIBRATION_QUERIES_PATH = ROOT / "data" / "eval" / "calibration_queries.json"


def load_queries(path: Path = GOLD_PATH) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["queries"]


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int = 3) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int = 3) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class RetrievalReport:
    name: str
    recall3: float = 0.0
    hit1: float = 0.0
    mrr: float = 0.0
    ndcg3: float = 0.0
    abstain: float = 0.0
    n_on_topic: int = 0
    n_off_topic: int = 0
    by_kind: Dict[str, float] = field(default_factory=dict)   # Hit@1 per query kind
    fallbacks: int = 0
    misses: List[Tuple[str, List[str], List[str]]] = field(default_factory=list)


def evaluate_retriever(name: str, retrieve: Callable[[str], List[str]], queries: List[Dict],
                       fallback_of: Callable[[], bool] = lambda: False) -> RetrievalReport:
    """`retrieve(query) -> ranked card ids`. `fallback_of()` says whether the last call degraded."""
    report = RetrievalReport(name)
    recalls, hits, rrs, ndcgs, abstains = [], [], [], [], []
    kinds: Dict[str, List[float]] = {}
    for item in queries:
        ranked = retrieve(item["q"])
        report.fallbacks += bool(fallback_of())
        relevant = item["relevant"]
        if not relevant:
            abstains.append(1.0 if not ranked else 0.0)
            if ranked:
                report.misses.append((item["q"], ranked[:3], []))
            continue
        hit = 1.0 if ranked[:1] and ranked[0] in relevant else 0.0
        recalls.append(recall_at_k(ranked, relevant))
        hits.append(hit)
        rrs.append(reciprocal_rank(ranked, relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant))
        kinds.setdefault(item.get("kind", "other"), []).append(hit)
        if not hit:
            report.misses.append((item["q"], ranked[:3], relevant))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0   # noqa: E731
    report.recall3, report.hit1, report.mrr, report.ndcg3 = mean(recalls), mean(hits), mean(rrs), mean(ndcgs)
    report.abstain = mean(abstains)
    report.n_on_topic, report.n_off_topic = len(hits), len(abstains)
    report.by_kind = {k: mean(v) for k, v in sorted(kinds.items())}
    return report


def fit_calibration(native_scores: Callable[[str], Dict[str, float]], queries: List[Dict]) -> Dict:
    """Fit a retriever's (floor, full) on held-out queries.

    positives: the best native score among each on-topic query's relevant cards
    negatives: the best native score of any card for each off-topic query
    floor:     the threshold with the best balanced accuracy (ties -> the lower threshold)
    full:      the median positive score, so a typical correct match counts as full evidence
    """
    positives, negatives = [], []
    for item in queries:
        scores = native_scores(item["q"])
        if item["relevant"]:
            positives.append(max((scores.get(c, 0.0) for c in item["relevant"]), default=0.0))
        else:
            negatives.append(max(scores.values(), default=0.0))
    candidates = sorted(set(positives + negatives))
    best, best_acc = 0.0, -1.0
    for t in candidates:
        tpr = sum(p >= t for p in positives) / len(positives) if positives else 0.0
        tnr = sum(n < t for n in negatives) / len(negatives) if negatives else 0.0
        acc = (tpr + tnr) / 2
        if acc > best_acc + 1e-12:
            best, best_acc = t, acc
    # Put the floor halfway into the gap below the chosen positive, not on top of it.
    below = [s for s in candidates if s < best]
    floor = (best + below[-1]) / 2 if below else best
    full = max(statistics.median(positives) if positives else floor, floor + 0.02)
    return {"floor": round(floor, 4), "full": round(full, 4), "balanced_accuracy": round(best_acc, 3),
            "n_positive": len(positives), "n_negative": len(negatives)}
