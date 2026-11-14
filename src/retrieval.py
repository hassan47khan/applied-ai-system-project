"""
Retrieval layer (the RAG half of VibeFinder 2.0).

Knowledge lives in data/knowledge/*.md as "context cards": a small front-matter block
(keywords, matching genres/moods, feature ranges) plus a short guidance paragraph.
A TF-IDF retriever finds the cards that best match a free-text request. The profile
builder then turns the retrieved cards into scoring targets, so retrieval directly
changes which songs rank first (it is not printed next to a standard answer).
"""

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .recommender import NUMERIC_ATTRS, ROOT

log = logging.getLogger("vibefinder.retrieval")

DEFAULT_KNOWLEDGE_DIR = Path(os.environ.get("VIBEFINDER_KNOWLEDGE") or ROOT / "data" / "knowledge")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it me my of on or so some "
    "that the their to up was we what when with you your need want like give something "
    "music songs song play listen listening something some for".split()
)
# Filler that would otherwise dilute a request now that unknown words count against it.
# Query-only, so card vectors (and the thresholds tuned on them) are unaffected.
_QUERY_STOPWORDS = _STOPWORDS | frozenset(
    "am do does s t m im this then please can will just get getting going".split()
)

# At least this share of a request's words must appear somewhere in the knowledge base.
# Otherwise it is about something else, and one incidental word ("late" in "late for my
# dentist appointment") must not count as a match.
MIN_QUERY_COVERAGE = 0.5


def _stem(token: str) -> str:
    """Tiny suffix stripper so 'studying' ~ 'study' and 'dancing' ~ 'dance'."""
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    else:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


_WORK_OUT = re.compile(r"\bwork(?:ing|s|ed)? out\b")   # "work out" is not "work"


def tokenize(text: str, stopwords: frozenset = _STOPWORDS) -> List[str]:
    text = _WORK_OUT.sub("workout", text.lower())
    return [_stem(t) for t in _TOKEN_RE.findall(text) if t not in stopwords]


@dataclass
class Card:
    id: str
    title: str
    keywords: List[str]
    genres: List[str]
    moods: List[str]
    ranges: Dict[str, Tuple[float, float]]
    body: str
    tokens: List[str] = field(default_factory=list, repr=False)

    @property
    def guidance(self) -> str:
        """First sentence of the body, used to ground explanations."""
        first = re.split(r"(?<=[.!?])\s", self.body.strip(), maxsplit=1)[0]
        return first.strip()


def _split_list(value: str) -> List[str]:
    return [v.strip().lower() for v in re.split(r"[,\n]", value) if v.strip()]


def parse_card(path: Path) -> Card:
    """Parse one markdown context card. Raises ValueError on a malformed card."""
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("missing front matter")
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    for required in ("id", "title", "keywords"):
        if not meta.get(required):
            raise ValueError(f"missing '{required}'")

    ranges: Dict[str, Tuple[float, float]] = {}
    for attr in NUMERIC_ATTRS:
        if attr in meta:
            lo, hi = (float(x) for x in meta[attr].split("-"))
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"bad range for {attr}: {meta[attr]}")
            ranges[attr] = (lo, hi)

    card = Card(
        id=meta["id"],
        title=meta["title"],
        keywords=_split_list(meta["keywords"]),
        genres=_split_list(meta.get("genres", "")),
        moods=_split_list(meta.get("moods", "")),
        ranges=ranges,
        body=match.group(2).strip(),
    )
    # Field weighting: keywords count 3x, title 2x, body 1x.
    card.tokens = (
        tokenize(" ".join(card.keywords)) * 3 + tokenize(card.title) * 2 + tokenize(card.body)
    )
    return card


def load_cards(directory: Path = DEFAULT_KNOWLEDGE_DIR) -> List[Card]:
    """Load every card; a malformed card is logged and skipped instead of crashing the app."""
    cards: List[Card] = []
    if not Path(directory).exists():
        log.warning("Knowledge directory %s not found; running without retrieval", directory)
        return cards
    for path in sorted(Path(directory).glob("*.md")):
        try:
            cards.append(parse_card(path))
        except (ValueError, OSError) as exc:
            log.warning("Skipping malformed knowledge card %s: %s", path.name, exc)
    return cards


class Retriever:
    """TF-IDF cosine retriever over context cards (pure Python, deterministic)."""

    def __init__(self, cards: List[Card]):
        self.cards = cards
        n = len(cards)
        df: Counter = Counter()
        for card in cards:
            df.update(set(card.tokens))
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self._unseen_idf = math.log(1 + n) + 1.0   # a word no card uses is maximally rare
        self._vectors = [self._vector(card.tokens) for card in cards]

    def _vector(self, tokens: List[str]) -> Dict[str, float]:
        # Unseen words stay in the vector: they can't match a card, but they must still count
        # in the norm, or a request with one known word looks like a perfect match.
        tf = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, self._unseen_idf) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.08) -> List[Tuple[Card, float]]:
        """Top-k cards with cosine similarity >= min_score, best first."""
        qvec = self._vector(tokenize(query, _QUERY_STOPWORDS))
        known = sum(t in self.idf for t in qvec)
        if not qvec or known < MIN_QUERY_COVERAGE * len(qvec):
            return []
        scored = []
        for card, dvec in zip(self.cards, self._vectors):
            sim = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
            if sim >= min_score:
                scored.append((card, sim))
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:k]


TfidfRetriever = Retriever   # the original VibeFinder 2.0 retriever, kept as the offline baseline

# ---------------------------------------------------------------------------------------
# VibeFinder 3.0 retrievers.
#
# The agent's thresholds (a 0.12 floor, 0.30 = "full evidence") were tuned on TF-IDF
# cosine. Every other retriever maps its native score onto that same scale with a linear
# calibration (native floor -> 0.12, native "clearly relevant" -> 0.30), computed from the
# labelled queries in data/eval/retrieval_gold.json by `python -m src.index calibrate`.
# So switching retrieval mode never silently changes what the agent counts as evidence.
# ---------------------------------------------------------------------------------------
EVIDENCE_FLOOR = 0.12
EVIDENCE_FULL = 0.30
BODY_ONLY_DISCOUNT = 0.4   # a card matched only through its prose, never its keywords/title


@dataclass(frozen=True)
class Calibration:
    floor: float       # native score treated as the evidence floor
    full: float        # native score treated as full evidence
    signal: str = "native"   # the unit floor/full are in; must match what the retriever emits

    def to_evidence(self, native: float) -> float:
        span = max(1e-9, self.full - self.floor)
        return max(0.0, EVIDENCE_FLOOR + (native - self.floor) / span * (EVIDENCE_FULL - EVIDENCE_FLOOR))


class BM25Retriever:
    """Okapi BM25 over the same weighted card tokens, with the same off-topic guard."""

    name = "bm25"
    DEFAULT_CALIBRATION = Calibration(floor=0.10, full=0.30)

    def __init__(self, cards: List[Card], k1: float = 1.2, b: float = 0.75,
                 calibration: Optional[Calibration] = None):
        self.cards = cards
        self.k1, self.b = k1, b
        self.calibration = calibration or self.DEFAULT_CALIBRATION
        n = len(cards)
        self._tf = [Counter(c.tokens) for c in cards]
        self._len = [len(c.tokens) for c in cards]
        self._avg = (sum(self._len) / n) if n else 0.0
        df: Counter = Counter()
        for card in cards:
            df.update(set(card.tokens))
        # Lucene-style idf, which never goes negative for words most cards share.
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        # Words a card is *about* (its keywords and title), as opposed to words its prose
        # happens to use. "good songs" must not match the cooking card on "feel-good".
        self._topic = [set(tokenize(" ".join(c.keywords))) | set(tokenize(c.title)) for c in cards]
        self.last_info: Dict[str, str] = {}

    def native_scores(self, query: str) -> List[Tuple[Card, float]]:
        """(card, BM25 normalised to [0, 1]) for every card, unsorted. [] if off-topic."""
        q = Counter(tokenize(query, _QUERY_STOPWORDS))
        known = sum(1 for t in q if t in self.idf)
        if not q or known < MIN_QUERY_COVERAGE * len(q):
            return []
        # Divide by the best score any card could reach for this query (every term
        # saturated), so scores are comparable across short and long requests.
        ceiling = sum(self.idf.get(t, 0.0) * (self.k1 + 1) for t in q) or 1.0
        out = []
        for card, tf, length, topic in zip(self.cards, self._tf, self._len, self._topic):
            score = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * length / (self._avg or 1.0))
                    score += self.idf[t] * f * (self.k1 + 1) / denom
            if score and not topic.intersection(q):
                score *= BODY_ONLY_DISCOUNT
            out.append((card, score / ceiling))
        return out

    def retrieve(self, query: str, k: int = 3, min_score: float = EVIDENCE_FLOOR) -> List[Tuple[Card, float]]:
        self.last_info = {"mode": self.name}
        scored = [(c, self.calibration.to_evidence(s)) for c, s in self.native_scores(query) if s > 0]
        scored = [(c, s) for c, s in scored if s >= min_score]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:k]


def has_content(text: str) -> bool:
    """Is there anything here to retrieve on? False for '', whitespace and bare punctuation."""
    return any(ch.isalnum() for ch in text or "")


class DenseRetriever:
    """Centred cosine similarity between a Gemini query embedding and the prebuilt card index.

    `embed_query(text) -> vector` raises GeminiUnavailable when Gemini can't answer; the
    caller (HybridRetriever or the agent) decides how to fall back.

    Why the score is *centred* (a card's cosine minus the query's mean cosine to every card)
    rather than raw cosine: in a 768-d embedding space every text is fairly similar to every
    card, and short or generic text is uniformly so. Measured on the real index, the empty
    string has a raw cosine of 0.66 to its best card and "good songs" 0.72, both higher than
    real on-topic requests such as "debugging a nasty bug all day" (0.62). An absolute floor
    cannot separate those; it passed the calibration set by 0.006 and then let empty and
    vague queries through as evidence, which silently disabled the agent's refusal checks.
    What does separate them is whether one card *stands out* from the rest for this query.
    Centring is a per-query shift, so the *order* of the cards never changes. What changes
    is which cards clear the floor, and that can still move metrics measured at a cutoff:
    a top-3 list can now hold fewer than three cards, which is why dense R@3 went from 0.98
    to 0.97. Inside the hybrid it also changes how an exact RRF tie is broken, because ties
    are broken on calibrated evidence.

    A centred score has no ceiling that depends on the number of cards, unlike a z-score
    (whose maximum is about sqrt(n - 1)), so a floor fitted on the 30-card index is not
    silently invalid for a smaller knowledge base.
    """

    name = "dense"
    SIGNAL = "centered"
    # Fitted on data/eval/calibration_queries.json; `python -m src.index calibrate` refits it.
    DEFAULT_CALIBRATION = Calibration(floor=0.0635, full=0.1382, signal=SIGNAL)

    def __init__(self, cards: List[Card], vectors: Dict[str, List[float]],
                 embed_query, calibration: Optional[Calibration] = None):
        self.cards = [c for c in cards if c.id in vectors]
        self.vectors = vectors
        self.embed_query = embed_query
        if calibration is not None and calibration.signal != self.SIGNAL:
            # A calibration fitted on raw cosine is in the wrong units for this score: applying
            # it would put the floor near 0.6 on a scale where real matches sit near 0.1, and
            # nothing would ever clear it. Refuse it loudly rather than mis-scale silently.
            log.warning("Ignoring a dense calibration fitted on %r scores; this retriever emits %r. "
                        "Run: python -m src.index calibrate", calibration.signal, self.SIGNAL)
            calibration = None
        self.calibration = calibration or self.DEFAULT_CALIBRATION
        self.last_info: Dict[str, str] = {}

    def cosine_scores(self, query: str) -> List[Tuple[Card, float]]:
        qvec = self.embed_query(query)
        return [(c, sum(a * b for a, b in zip(qvec, self.vectors[c.id]))) for c in self.cards]

    def native_scores(self, query: str) -> List[Tuple[Card, float]]:
        """The score calibration is fitted on and retrieval thresholds: centred cosine."""
        raw = self.cosine_scores(query)
        if not raw:
            return []
        mean = sum(s for _, s in raw) / len(raw)
        return [(c, s - mean) for c, s in raw]

    def retrieve(self, query: str, k: int = 3, min_score: float = EVIDENCE_FLOOR) -> List[Tuple[Card, float]]:
        self.last_info = {"mode": self.name}
        if not has_content(query):
            # Lexical retrieval returns nothing for an empty query by construction. Dense
            # retrieval would happily embed "" and find it "similar" to every card, so the
            # cards' genres would be counted as the user's preferences. This is what stopped
            # "anything but jazz" (whose retrieval text is "" once the rejection is removed)
            # from ever being flagged as exclusion-only. Match the lexical contract instead.
            return []
        scored = [(c, self.calibration.to_evidence(s)) for c, s in self.native_scores(query)]
        scored = [(c, s) for c, s in scored if s >= min_score]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:k]


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """RRF: the sum of 1 / (k + rank) over every ranking a document appears in (rank from 1)."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def mmr_select(candidates: List[str], relevance: Dict[str, float], similarity,
               k: int, lam: float = 0.7) -> List[str]:
    """Maximal Marginal Relevance: trade relevance against similarity to what is already picked."""
    picked: List[str] = []
    pool = list(candidates)
    top = max(relevance.values(), default=1.0) or 1.0
    while pool and len(picked) < k:
        def gain(doc_id: str) -> float:
            redundancy = max((similarity(doc_id, p) for p in picked), default=0.0)
            return lam * relevance[doc_id] / top - (1 - lam) * redundancy
        best = max(pool, key=lambda d: (gain(d), -pool.index(d)))   # ties keep the fused order
        picked.append(best)
        pool.remove(best)
    return picked


class HybridRetriever:
    """Lexical (BM25) + dense, fused with Reciprocal Rank Fusion, diversified with MMR.

    * A card's reported score is the best calibrated evidence any arm gave it, so the
      agent's floor still means "at least one retriever is confident about this card".
    * If there is no dense arm or Gemini fails, this behaves exactly like the lexical
      retriever, and `last_info["fallback"]` says why (the agent puts it in its trace).
    """

    name = "hybrid"

    def __init__(self, lexical, dense: Optional[DenseRetriever] = None,
                 rrf_k: int = 60, pool: int = 8, mmr_lambda: float = 0.7):
        self.lexical = lexical
        self.dense = dense
        self.rrf_k, self.pool, self.mmr_lambda = rrf_k, pool, mmr_lambda
        self.cards = lexical.cards
        self.last_info: Dict[str, str] = {}

    def retrieve(self, query: str, k: int = 3, min_score: float = EVIDENCE_FLOOR) -> List[Tuple[Card, float]]:
        lexical_hits = self.lexical.retrieve(query, k=self.pool, min_score=min_score)
        if self.dense is None:
            self.last_info = {"mode": "lexical", "fallback": "no embedding index"}
            return lexical_hits[:k]
        try:
            dense_hits = self.dense.retrieve(query, k=self.pool, min_score=min_score)
        except Exception as exc:   # GeminiUnavailable or anything else: degrade, don't fail
            reason = getattr(exc, "reason", None) or type(exc).__name__
            self.last_info = {"mode": "lexical", "fallback": reason}
            return lexical_hits[:k]

        self.last_info = {"mode": self.name}
        evidence: Dict[str, float] = {}
        by_id: Dict[str, Card] = {}
        for card, score in lexical_hits + dense_hits:
            by_id[card.id] = card
            evidence[card.id] = max(evidence.get(card.id, 0.0), score)
        if not evidence:
            return []
        fused = reciprocal_rank_fusion([[c.id for c, _ in lexical_hits], [c.id for c, _ in dense_hits]],
                                       k=self.rrf_k)
        order = sorted(fused, key=lambda d: (-fused[d], -evidence[d], d))
        vectors = self.dense.vectors

        def similarity(a: str, b: str) -> float:
            va, vb = vectors.get(a), vectors.get(b)
            return sum(x * y for x, y in zip(va, vb)) if va and vb else 0.0

        chosen = mmr_select(order, fused, similarity, k, self.mmr_lambda)
        return [(by_id[d], evidence[d]) for d in chosen]
