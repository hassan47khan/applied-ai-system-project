"""
The prebuilt card-embedding index and the retriever factory.

Card vectors are built once (`python -m src.index build`) and committed under data/index/,
so the app never re-embeds the knowledge base at runtime and dense retrieval only costs
one embedding call per new request (and none for a cached one).

    data/index/cards.json         {card_id: {"sha": ..., "vector": [...]}}
    data/index/manifest.json      model, dimensions, card count, content hash, build time
    data/index/calibration.json   per-retriever score calibration (see retrieval.Calibration)

A missing or stale index never breaks the app: the factory logs why and falls back to
lexical retrieval, and the reason ends up in the agent trace.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Settings, get_settings
from .gemini import GeminiClient, GeminiUnavailable, shared_client
from .retrieval import (
    BM25Retriever, Calibration, Card, DenseRetriever, HybridRetriever, TfidfRetriever,
)

log = logging.getLogger("vibefinder.embeddings")

CARDS_FILE = "cards.json"
MANIFEST_FILE = "manifest.json"
CALIBRATION_FILE = "calibration.json"


def card_text(card: Card) -> str:
    """What gets embedded for a card: its situation, its vocabulary and its guidance."""
    return (f"{card.title}. Keywords: {', '.join(card.keywords)}. "
            f"Genres: {', '.join(card.genres)}. Moods: {', '.join(card.moods)}. {card.body}")


def card_sha(card: Card) -> str:
    return hashlib.sha256(card_text(card).encode("utf-8")).hexdigest()


def content_sha(cards: List[Card]) -> str:
    blob = "\n".join(f"{c.id}:{card_sha(c)}" for c in sorted(cards, key=lambda c: c.id))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class CardIndex:
    model: str
    dim: int
    vectors: Dict[str, List[float]]
    stale: List[str]          # card ids whose text changed since the index was built
    missing: List[str]        # cards with no vector at all

    @property
    def usable(self) -> bool:
        return bool(self.vectors) and not self.stale and not self.missing


def load_index(cards: List[Card], directory: Optional[Path] = None) -> Optional[CardIndex]:
    directory = Path(directory or get_settings().index_dir)
    try:
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        stored = json.loads((directory / CARDS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    vectors, stale, missing = {}, [], []
    for card in cards:
        entry = stored.get(card.id)
        if entry is None:
            missing.append(card.id)
        elif entry.get("sha") != card_sha(card):
            stale.append(card.id)
        else:
            vectors[card.id] = entry["vector"]
    return CardIndex(manifest.get("model", ""), int(manifest.get("dim", 0)), vectors, stale, missing)


def build_index(cards: List[Card], client: GeminiClient, directory: Optional[Path] = None) -> CardIndex:
    """Embed every card and write the index. Raises GeminiUnavailable on failure."""
    directory = Path(directory or client.settings.index_dir)
    model = client.embedding_model()
    vectors = client.embed([card_text(c) for c in cards], kind="document", model=model,
                           titles=[c.title for c in cards], purpose="index")
    directory.mkdir(parents=True, exist_ok=True)
    stored = {c.id: {"sha": card_sha(c), "vector": [round(v, 6) for v in vec]}
              for c, vec in zip(cards, vectors)}
    (directory / CARDS_FILE).write_text(json.dumps(stored, sort_keys=True), encoding="utf-8")
    manifest = {
        "model": model, "dim": len(vectors[0]) if vectors else 0, "card_count": len(cards),
        "content_sha": content_sha(cards),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return CardIndex(model, manifest["dim"], {k: v["vector"] for k, v in stored.items()}, [], [])


def load_calibration(directory: Optional[Path] = None) -> Dict[str, Calibration]:
    directory = Path(directory or get_settings().index_dir)
    try:
        raw = json.loads((directory / CALIBRATION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for name, values in raw.items():
        try:
            out[name] = Calibration(float(values["floor"]), float(values["full"]),
                                    str(values.get("signal", "native")))
        except (KeyError, TypeError, ValueError):
            log.warning("Ignoring malformed calibration for %s", name)
    return out


def save_calibration(calibrations: Dict[str, Dict], directory: Optional[Path] = None) -> None:
    directory = Path(directory or get_settings().index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CALIBRATION_FILE).write_text(json.dumps(calibrations, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")


def make_retriever(cards: List[Card], mode: Optional[str] = None, settings: Optional[Settings] = None,
                   client: Optional[GeminiClient] = None, llm=None) -> Tuple[object, List[str]]:
    """Build the retriever for `mode` and return (retriever, notes about any downgrade).

    tfidf  the original VibeFinder 2.0 retriever (offline)
    bm25   Okapi BM25 (offline)
    dense  Gemini query embedding vs the prebuilt index; lexical if that's unavailable
    hybrid BM25 + dense with RRF and MMR; lexical if the dense arm is unavailable
    """
    settings = settings or get_settings()
    mode = mode or settings.retrieval_mode
    notes: List[str] = []
    calibration = load_calibration(settings.index_dir)
    bm25 = BM25Retriever(cards, calibration=calibration.get("bm25"))
    if mode == "tfidf":
        return TfidfRetriever(cards), notes
    if mode == "bm25":
        return bm25, notes

    client = client or shared_client()
    dense: Optional[DenseRetriever] = None
    index = load_index(cards, settings.index_dir)
    if index is None:
        notes.append("no embedding index (run: python -m src.index build)")
    elif not index.usable:
        notes.append(f"embedding index is stale for {len(index.stale) + len(index.missing)} card(s) "
                     "(run: python -m src.index build)")
    elif not client.available:
        notes.append(f"dense retrieval off ({client.unavailable_reason()})")
    else:
        def embed_query(text: str) -> List[float]:
            return client.embed([text], kind="query", model=index.model, purpose="query_embed")[0]
        dense = DenseRetriever(cards, index.vectors, embed_query, calibration=calibration.get("dense"))

    if mode == "dense":
        retriever = _DenseWithFallback(dense, bm25) if dense is not None else bm25
    else:
        retriever = HybridRetriever(bm25, dense)
    if settings.rerank:
        # Rerank is generation, so it follows VIBEFINDER_LLM; the dense arm above stays on the
        # Gemini client, because the committed index was built with its embedding model.
        if llm is None and settings.llm_provider == "gemini":
            llm = client
        elif llm is None:
            from .providers import shared_llm
            llm = shared_llm(settings.llm_provider)
        if llm.available:
            from .rerank import RerankingRetriever
            retriever = RerankingRetriever(retriever, llm)
    return retriever, notes


class _DenseWithFallback:
    """Dense-only mode that still answers (lexically) when Gemini is unavailable."""

    name = "dense"

    def __init__(self, dense: DenseRetriever, lexical):
        self.dense, self.lexical = dense, lexical
        self.cards = lexical.cards
        self.last_info: Dict[str, str] = {}

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.12):
        try:
            hits = self.dense.retrieve(query, k=k, min_score=min_score)
            self.last_info = {"mode": "dense"}
            return hits
        except GeminiUnavailable as exc:
            self.last_info = {"mode": "lexical", "fallback": exc.reason}
            return self.lexical.retrieve(query, k=k, min_score=min_score)
