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
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .recommender import NUMERIC_ATTRS, ROOT

log = logging.getLogger("vibefinder.retrieval")

DEFAULT_KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

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
