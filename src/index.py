"""
Manage the retrieval index.

    python -m src.index build       embed every knowledge card (needs GEMINI_API_KEY)
    python -m src.index verify      check the committed index matches the cards on disk
    python -m src.index calibrate   fit each retriever's score scale on held-out queries
    python -m src.index info        print what is in the index

`build` and the dense half of `calibrate` call Gemini; `verify`, `info` and the BM25
half of `calibrate` run offline.
"""

import argparse
import json
import sys

from .config import get_settings
from .embeddings import (
    MANIFEST_FILE, build_index, load_calibration, load_index, save_calibration,
)
from .gemini import GeminiUnavailable, shared_client
from .log_setup import setup_logging
from .retrieval import BM25Retriever, DenseRetriever, load_cards
from .retrieval_eval import CALIBRATION_QUERIES_PATH, fit_calibration, load_queries


def cmd_build(settings) -> int:
    cards = load_cards(settings.knowledge_dir)
    client = shared_client()
    if not client.available:
        print(f"Cannot build the index: Gemini is {client.unavailable_reason()}. "
              "Set GEMINI_API_KEY in .env.", file=sys.stderr)
        return 2
    try:
        index = build_index(cards, client, settings.index_dir)
    except GeminiUnavailable as exc:
        print(f"Index build failed: {exc.reason}", file=sys.stderr)
        return 1
    print(f"Embedded {len(index.vectors)} cards with {index.model} ({index.dim}-d) -> {settings.index_dir}")
    print(f"Gemini calls: {client.stats.calls}, cache hits: {client.stats.cache_hits}")
    return 0


def cmd_verify(settings) -> int:
    cards = load_cards(settings.knowledge_dir)
    index = load_index(cards, settings.index_dir)
    if index is None:
        print("No index found. Run: python -m src.index build")
        return 1
    if index.usable:
        print(f"OK: {len(index.vectors)} cards, {index.model} ({index.dim}-d), up to date.")
        return 0
    if index.stale:
        print("Stale (card text changed): " + ", ".join(index.stale))
    if index.missing:
        print("Missing (no vector): " + ", ".join(index.missing))
    print("Run: python -m src.index build")
    return 1


def cmd_calibrate(settings) -> int:
    cards = load_cards(settings.knowledge_dir)
    queries = load_queries(CALIBRATION_QUERIES_PATH)
    calibrations = {name: {"floor": c.floor, "full": c.full, "signal": c.signal}
                    for name, c in load_calibration(settings.index_dir).items()}

    bm25 = BM25Retriever(cards)
    calibrations["bm25"] = fit_calibration(lambda q: {c.id: s for c, s in bm25.native_scores(q)}, queries)
    calibrations["bm25"]["signal"] = "native"
    print(f"bm25  : {calibrations['bm25']}")

    index = load_index(cards, settings.index_dir)
    client = shared_client()
    if index is not None and index.usable and client.available:
        try:
            texts = [item["q"] for item in queries]
            vectors = dict(zip(texts, client.embed(texts, kind="query", model=index.model,
                                                   purpose="calibrate")))
        except GeminiUnavailable as exc:
            print(f"dense : skipped ({exc.reason})")
        else:
            # Fit on DenseRetriever.native_scores itself, not on a re-implementation of it, so
            # the calibration is always in the units retrieval actually thresholds on. (It once
            # computed raw cosine here while retrieval used the same, and nothing checked that
            # they agreed; changing either one alone would have mis-scaled every threshold.)
            dense = DenseRetriever(cards, index.vectors, lambda q: vectors[q])
            calibrations["dense"] = fit_calibration(
                lambda q: {c.id: s for c, s in dense.native_scores(q)}, queries)
            calibrations["dense"]["signal"] = DenseRetriever.SIGNAL
            calibrations["dense"]["n_cards"] = len(dense.cards)
            print(f"dense : {calibrations['dense']}")
    else:
        print("dense : skipped (needs a usable index and GEMINI_API_KEY)")
    save_calibration(calibrations, settings.index_dir)
    print(f"Saved -> {settings.index_dir / 'calibration.json'}")
    return 0


def cmd_info(settings) -> int:
    try:
        print((settings.index_dir / MANIFEST_FILE).read_text(encoding="utf-8").strip())
    except OSError:
        print("No index manifest.")
    print(json.dumps({k: vars(v) for k, v in load_calibration(settings.index_dir).items()}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VibeFinder retrieval index")
    parser.add_argument("command", choices=["build", "verify", "calibrate", "info"])
    args = parser.parse_args(argv)
    setup_logging()
    settings = get_settings()
    return {"build": cmd_build, "verify": cmd_verify, "calibrate": cmd_calibrate,
            "info": cmd_info}[args.command](settings)


if __name__ == "__main__":
    raise SystemExit(main())
