"""
VibeFinder 2.0 command line.

    python -m src.main "something calm for studying late at night"
    python -m src.main --demo            # three built-in example requests
    python -m src.main --interactive     # keep asking until you type 'quit'
    python -m src.main --trace "..."     # also print the agent's step-by-step trace
    python -m src.main --llm "..."       # Gemini-written explanations (needs GEMINI_API_KEY)
    python -m src.main --json "..."      # machine-readable output
"""

import argparse
import json
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
    lines = ["=" * WIDTH, "  VibeFinder 2.0", f'  Request: "{result.cleaned_query or result.query}"', "=" * WIDTH]
    for w in result.warnings:
        lines.append(f"  [guardrail] {w}")
    if result.ok:
        ctx = ", ".join(f"{cid} ({score:.2f})" for cid, score in result.retrieved) or "none"
        lines.append(f"  Retrieved context : {ctx}")
        lines += _wrap("  Search profile    : ", result.profile, 22)
        lines.append(f"  Confidence        : {result.confidence:.2f}   "
                     f"Needs human review: {'YES' if result.needs_review else 'no'}")
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
    parser = argparse.ArgumentParser(description="VibeFinder 2.0: retrieval-grounded music recommender")
    parser.add_argument("query", nargs="*", help="what you want to listen to")
    parser.add_argument("--demo", action="store_true", help="run built-in example requests")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--trace", action="store_true", help="print the agent's step-by-step trace")
    parser.add_argument("--llm", action="store_true", help="use Gemini for explanations if available")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--k", type=int, default=5, help="number of recommendations (default 5)")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        agent = VibeFinderAgent(explainer=make_explainer(args.llm), k=max(1, min(args.k, 10)))
    except (FileNotFoundError, OSError) as exc:
        print(f"Could not start VibeFinder: {exc}", file=sys.stderr)
        return 2

    def handle(text: str) -> None:
        result = agent.run(text)
        if args.json:
            print(json.dumps(asdict(result), indent=2, default=str))
        else:
            print(format_result(result, show_trace=args.trace))

    if args.demo:
        for q in DEMO_QUERIES:
            handle(q)
            print()
    elif args.interactive or not args.query:
        print("VibeFinder 2.0 - describe what you want to hear ('quit' to exit).")
        while True:
            try:
                text = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.strip().lower() in {"quit", "exit", "q"}:
                break
            handle(text)
    else:
        handle(" ".join(args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
