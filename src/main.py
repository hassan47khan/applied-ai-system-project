"""
VibeFinder 3.0 command line.

    python -m src.main "something calm for studying late at night"
    python -m src.main --demo            # three built-in example requests
    python -m src.main --interactive     # keep asking until you type 'quit'
    python -m src.main --trace "..."     # also print the agent's step-by-step trace
    python -m src.main --offline "..."   # never call Gemini, even with GEMINI_API_KEY set
    python -m src.main --retrieval bm25 "..."   # tfidf | bm25 | dense | hybrid

With GEMINI_API_KEY set (in the environment or .env), Gemini reads the request, embeds it
for retrieval and writes the explanations; without it everything runs offline.
    python -m src.main --json "..."      # machine-readable output
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import List

from .agent import AgentResult, VibeFinderAgent
from .llm import make_explainer
from .log_setup import setup_logging

DEMO_QUERIES = [
    "I need something calm for studying late at night",
    "lofi at maximum energy, not acoustic",
    "asdfghjkl qwerty",
]

WIDTH = 68


def _wrap(prefix: str, text: str, indent: int) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width=WIDTH, initial_indent=prefix,
                         subsequent_indent=" " * indent) or [prefix]


def format_result(result: AgentResult, show_trace: bool = False) -> str:
    lines = ["=" * WIDTH, "  VibeFinder 3.0", f'  Request: "{result.cleaned_query or result.query}"', "=" * WIDTH]
    for w in result.warnings:
        lines.append(f"  [guardrail] {w}")
    if result.support:
        lines += _wrap("  [support] ", result.support, 12)
    if result.ok:
        ctx = ", ".join(f"{cid} ({score:.2f})" for cid, score in result.retrieved) or "none"
        lines.append(f"  Retrieved context : {ctx}")
        lines += _wrap("  Search profile    : ", result.profile, 22)
        lines.append(f"  Confidence        : {result.confidence:.2f}   "
                     f"Needs human review: {'YES' if result.needs_review else 'no'}")
        if result.parser != "rules" or (result.retrieval_mode and result.retrieval_mode != "tfidf"):
            lines.append(f"  Understood by     : {result.parser}   retrieval: {result.retrieval_mode or 'none'}")
    for rec in result.recommendations:
        song = rec.song
        lines.append("")
        lines.append(f"#{rec.rank}  {song['title']} - {song['artist']}  "
                     f"[{song['genre']} / {song['mood']}]  score {rec.score:.2f} (fit {rec.fit:.2f})")
        lines += _wrap("     ", rec.explanation, 5)
    scores = [rec.score for rec in result.recommendations]
    if scores != sorted(scores, reverse=True):
        lines.append("")
        lines += _wrap("  Note: ", "order includes a variety re-rank (repeat genres/artists are "
                       "down-weighted), so raw scores are not strictly descending.", 8)
    if result.message:
        lines.append("")
        lines += _wrap("  >> ", result.message, 6)
    if show_trace:
        lines += ["", "  Agent trace:"]
        for t in result.trace:
            lines += _wrap(f"   {t['step']:>2}. {t['action']:<9} ", t["detail"], 17)
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VibeFinder 3.0: retrieval-grounded music recommender")
    parser.add_argument("query", nargs="*", help="what you want to listen to")
    parser.add_argument("--demo", action="store_true", help="run built-in example requests")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--trace", action="store_true", help="print the agent's step-by-step trace")
    parser.add_argument("--llm", action="store_true",
                        help="kept for 2.0 compatibility; Gemini is now used whenever a key is set")
    parser.add_argument("--offline", action="store_true", help="never call any LLM")
    parser.add_argument("--provider", choices=["gemini", "nvidia"],
                        help="which LLM understands requests, reranks and explains "
                             "(default: VIBEFINDER_LLM or gemini)")
    parser.add_argument("--model", help="try this model first for the chosen provider, e.g. "
                                        "nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--retrieval", choices=["tfidf", "bm25", "dense", "hybrid"],
                        help="retrieval mode (default: VIBEFINDER_RETRIEVAL or hybrid)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--k", type=int, default=5, help="number of recommendations (default 5)")
    args = parser.parse_args(argv)

    # A request can hold characters the console encoding lacks (emoji on a cp1252 pipe).
    # Echoing it back must not crash the app after the agent has already succeeded.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    setup_logging()
    try:
        if args.offline:
            os.environ["VIBEFINDER_USE_GEMINI"] = "0"    # read when the shared client is created
        if args.provider:
            os.environ["VIBEFINDER_LLM"] = args.provider
        if args.model:
            os.environ["VIBEFINDER_LLM_MODEL"] = args.model
        agent = VibeFinderAgent(explainer=make_explainer(not args.offline), k=max(1, min(args.k, 10)),
                                retrieval_mode=args.retrieval)
    except (OSError, ValueError) as exc:
        print(f"Could not start VibeFinder: {exc}", file=sys.stderr)
        return 2

    from .session import Conversation
    conversation = Conversation(agent)

    def handle(text: str) -> None:
        # Interactive mode keeps context, so "more upbeat" or "not #3" refines the last answer.
        result = conversation.send(text) if args.interactive or not args.query else agent.run(text)
        if args.json:
            print(json.dumps(asdict(result), indent=2, default=str))
        else:
            print(format_result(result, show_trace=args.trace))

    if args.demo:
        for q in DEMO_QUERIES:
            handle(q)
            print()
    elif args.interactive or not args.query:
        print("VibeFinder 3.0 - describe what you want to hear ('quit' to exit, 'reset' to start over).")
        print("Follow-ups refine the last answer: 'more upbeat', 'less jazz', 'not #3', 'more like #2'.")
        while True:
            try:
                text = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.strip().lower() in {"quit", "exit", "q"}:
                break
            if text.strip().lower() == "reset":
                conversation.reset()
                continue
            handle(text)
    else:
        handle(" ".join(args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
