"""
Mutation check: does the test suite notice when the system is broken?

Each mutation plants one small, realistic bug (switch off the variety re-rank, hide the
review flag, make the harness's check() always pass, ...) in a throwaway copy of the
project, then runs pytest. A mutation the suite fails on is "killed". One that passes
"survived", meaning no test actually checks that behaviour.

    python tools/mutate.py            # all mutations, about a minute
    python tools/mutate.py diversity  # only mutations whose name contains "diversity"

Exits 1 if any mutation survives. The project itself is never modified.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
IGNORE = shutil.ignore_patterns(".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
                                "logs", "tools", ".cache", ".taste-skill")


class Mutation(NamedTuple):
    name: str
    path: str
    old: str
    new: str


MUTATIONS: List[Mutation] = [
    # agent: the loop, the verifier and the decision
    Mutation("agent: rejected songs not filtered", "src/agent.py",
             "candidates = [s for s in self.songs if profile.allows(s)]", "candidates = list(self.songs)"),
    Mutation("agent: diversity re-rank disabled", "src/agent.py",
             "return item[1] - DIVERSITY_PENALTY * repeats", "return item[1]"),
    Mutation("agent: diversity ignores artist", "src/agent.py",
             '(p[0]["genre"] == song["genre"]) + (p[0]["artist"] == song["artist"])',
             '(p[0]["genre"] == song["genre"])'),
    Mutation("agent: no relaxation penalty", "src/agent.py",
             "RELAXATION_PENALTY = 0.15", "RELAXATION_PENALTY = 0.0"),
    Mutation("agent: evidence always 1.0", "src/agent.py",
             "evidence = 1.0 if profile.has_explicit_signal else min(1.0, top_card_score / EVIDENCE_REF)",
             "evidence = 1.0"),
    Mutation("agent: never revise on conflict", "src/agent.py",
             "if conflict and iteration < MAX_ITERATIONS:", "if False:"),
    Mutation("agent: relaxation not flagged for review", "src/agent.py",
             "result.needs_review = bool(result.relaxations) or bool(attempt.issues)",
             "result.needs_review = bool(attempt.issues)"),
    Mutation("agent: card choice ignores song", "src/agent.py",
             'if song["genre"] in card.genres or song["mood"] in card.moods:', "if False:"),
    Mutation("agent: CONFIDENCE_MIN 0.60 -> 0.30", "src/agent.py",
             "CONFIDENCE_MIN = 0.60", "CONFIDENCE_MIN = 0.30"),
    Mutation("agent: retrieval floor removed", "src/agent.py",
             "RETRIEVAL_MIN_SCORE = 0.12", "RETRIEVAL_MIN_SCORE = 0.0"),
    Mutation("agent: unsatisfiable never raised", "src/agent.py",
             "if best_all > UNSAT_TOLERANCE:", "if False:"),
    Mutation("agent: conflict tolerance 0.25 -> 0.9", "src/agent.py",
             "CONFLICT_TOLERANCE = 0.25", "CONFLICT_TOLERANCE = 0.9"),
    Mutation("agent: unsatisfiable note dropped from message", "src/agent.py",
             'for code in ("mixed_context", "exclusion_only", "unsatisfiable")',
             'for code in ("mixed_context", "exclusion_only")'),
    Mutation("agent: recommendation fit not clamped", "src/agent.py",
             "fit=round(max(0.0, min(1.0, score / attempt.ceiling)), 2)",
             "fit=round(score / attempt.ceiling, 2)"),
    Mutation("agent: confidence rounded across the threshold", "src/agent.py",
             "    if confidence < CONFIDENCE_MIN <= shown:\n", "    if False:\n"),
    Mutation("agent: retrieval sees rejected phrases", "src/agent.py",
             "strip_negated(text, self.genre_vocab, self.mood_vocab)", "text"),
    # scorer
    Mutation("recommender: no title tie-break", "src/recommender.py",
             'scored.sort(key=lambda item: (-item[1], item[0]["title"]))',
             "scored.sort(key=lambda item: -item[1])"),
    Mutation("recommender: context mood weight 0", "src/recommender.py",
             "CONTEXT_MOOD_WEIGHT = 0.5 ", "CONTEXT_MOOD_WEIGHT = 0.0 "),
    Mutation("recommender: explicit mood ignored", "src/recommender.py",
             "        score += MOOD_WEIGHT\n", "        score += 0\n"),
    Mutation("recommender: avoid penalty hits related genres", "src/recommender.py",
             'if any(g.lower() == song["genre"].lower() for g in avoid_genres):',
             'if _best_affinity(avoid_genres, song["genre"]):'),
    Mutation("recommender: Recommender ignores acoustic preference", "src/recommender.py",
             '"acousticness": 0.8 if user.likes_acoustic else 0.2,', '"acousticness": 0.5,'),
    Mutation("recommender: explanation is junk", "src/recommender.py",
             'return "; ".join(reasons) if reasons else "no matching attributes"\n', 'return "x"\n'),
    # retrieval
    Mutation("retrieval: no field weighting", "src/retrieval.py",
             'tokenize(" ".join(card.keywords)) * 3 + tokenize(card.title) * 2',
             'tokenize(" ".join(card.keywords)) + tokenize(card.title)'),
    Mutation("retrieval: no id tie-break", "src/retrieval.py",
             "scored.append((card, sim))\n        scored.sort(key=lambda item: (-item[1], item[0].id))",
             "scored.append((card, sim))\n        scored.sort(key=lambda item: -item[1])"),
    Mutation("retrieval: query coverage check off", "src/retrieval.py",
             "MIN_QUERY_COVERAGE = 0.5", "MIN_QUERY_COVERAGE = 0.0"),
    # guardrails
    Mutation("guardrails: whitespace not collapsed", "src/guardrails.py",
             r'text = re.sub(r"\s+", " ", text).strip()', "text = text.strip()"),
    Mutation("guardrails: 'you are now' not detected", "src/guardrails.py",
             'r"you are now ",', ""),
    Mutation("guardrails: truncation cuts mid-word", "src/guardrails.py",
             "            cut = cut[:space]\n", "            pass\n"),
    # explainer
    Mutation("llm: trade-offs not separated", "src/llm.py",
             'positives = [r for r in reasons if "misses" not in r]', "positives = list(reasons)"),
    Mutation("llm: template drops the card guidance", "src/llm.py",
             'text += f" Context ({card.title}): {card.guidance}"', "pass"),
    Mutation("llm: prompt drops the grounding instruction", "src/llm.py",
             "do not invent details about the song or artist.", "feel free to embellish."),
    Mutation("llm: --llm without a key uses an unusable client", "src/llm.py",
             "    if not client.available:\n", "    if False:\n"),
    # profile builder and negation
    Mutation("profile: second card always blended", "src/profile.py",
             "SECONDARY_CARD_RATIO = 0.6", "SECONDARY_CARD_RATIO = 0.0"),
    Mutation("profile: second card never blended", "src/profile.py",
             "MAX_CONTEXT_CARDS = 2", "MAX_CONTEXT_CARDS = 1"),
    Mutation("profile: card clash gap 0.15 -> 0.9", "src/profile.py",
             "BLEND_CONFLICT_GAP = 0.15", "BLEND_CONFLICT_GAP = 0.9"),
    Mutation("profile: copy() drops context_clash", "src/profile.py",
             "            context_clash=self.context_clash,\n", ""),
    Mutation("profile: any comma ends a rejected list", "src/profile.py",
             "    continuation = _list_continuation(list_terms)\n", "    continuation = None\n"),
    Mutation("profile: 'not less X' not flipped", "src/profile.py",
             'leaning.append((attr, flipped if found.group("neg") else lean))', "leaning.append((attr, lean))"),
    Mutation("profile: 'less X' read as a rejection", "src/profile.py",
             "        found = pattern.search(normalised)\n", "        found = None\n"),
    # CLI, harness, logging
    Mutation("main: review flag always 'no'", "src/main.py",
             "{'YES' if result.needs_review else 'no'}", "no"),
    Mutation("main: --json prints repr", "src/main.py",
             "print(json.dumps(asdict(result), indent=2, default=str))", "print(result)"),
    Mutation("main: --k not clamped", "src/main.py", "k=max(1, min(args.k, 10))", "k=args.k"),
    Mutation("main: guardrail warnings not shown", "src/main.py",
             '        lines.append(f"  [guardrail] {w}")\n', "        pass\n"),
    Mutation("main: message not shown", "src/main.py",
             '        lines += _wrap("  >> ", result.message, 6)\n', "        pass\n"),
    Mutation("evaluate: check() always passes", "src/evaluate.py",
             "missing\")\n    return failures\n", "missing\")\n    return []\n"),
    Mutation("evaluate: --strict ignored", "src/evaluate.py",
             "return 1 if (args.strict and (passed != total or injection_failures)) else 0", "return 0"),
    Mutation("log_setup: no log file", "src/log_setup.py", "        root.addHandler(file_handler)\n", "        pass\n"),

    # VibeFinder 3.0: retrieval
    Mutation("bm25: body-only matches not discounted", "src/retrieval.py",
             "                score *= BODY_ONLY_DISCOUNT\n", "                pass\n"),
    Mutation("hybrid: fallback reason dropped", "src/retrieval.py",
             'self.last_info = {"mode": "lexical", "fallback": reason}', 'self.last_info = {"mode": "lexical"}'),
    Mutation("rrf: rank fusion ignores the k constant", "src/retrieval.py", "1.0 / (k + rank)", "1.0 / rank"),
    Mutation("mmr: redundancy ignored", "src/retrieval.py",
             "return lam * relevance[doc_id] / top - (1 - lam) * redundancy", "return relevance[doc_id]"),
    Mutation("index: edited cards not detected", "src/embeddings.py",
             'elif entry.get("sha") != card_sha(card):', "elif False:"),
    # Gemini client
    Mutation("gemini: no retry on rate limit", "src/gemini.py",
             "if _is_transient(exc) and attempt < self.max_attempts:", "if False:"),
    Mutation("gemini: cache never read", "src/gemini.py",
             "        cached = self.cache.get(key)\n", "        cached = None\n"),
    # understanding and injection defences
    Mutation("understanding: ranges not clamped", "src/understanding.py",
             "clamped = (min(1.0, max(0.0, lo)), min(1.0, max(0.0, hi)))", "clamped = (lo, hi)"),
    Mutation("understanding: instruction-like paraphrase kept", "src/understanding.py",
             "if any(p.search(cleaned) for p in _INJECTION_PATTERNS):", "if False:"),
    Mutation("agent: flagged input sent to Gemini", "src/agent.py",
             'if any("instruction-override" in w for w in result.warnings):', "if False:"),
    Mutation("guardrails: invisible characters kept", "src/guardrails.py",
             'text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", raw))', "text = raw"),
    Mutation("agent: distress not flagged for review", "src/agent.py",
             " or bool(result.support)", ""),
    # explanations and conversation
    Mutation("explain: invented numbers accepted", "src/llm.py", "        if invented:\n", "        if False:\n"),
    Mutation("explain: foreign card citations accepted", "src/llm.py",
             'if cited not in {"none", card.id if card else "none"}:', "if False:"),
    Mutation("session: dismissed songs forgotten", "src/session.py",
             "self.dismissed |= {shown_ids[r - 1] for r in ref.drop_ranks}", "pass"),
    Mutation("session: 'more upbeat' moves the wrong way", "src/session.py",
             "center = _clamp(center + direction * STEP)", "center = _clamp(center - direction * STEP)"),
    # dense abstention: each of these re-introduces one cause of the live refusal regression
    Mutation("dense: empty query embedded", "src/retrieval.py",
             "if not has_content(query):", "if False:"),
    Mutation("dense: raw cosine used as evidence", "src/retrieval.py",
             "        return [(c, s - mean) for c, s in raw]", "        return raw"),
    Mutation("dense: calibration unit not checked", "src/retrieval.py",
             "if calibration is not None and calibration.signal != self.SIGNAL:", "if False:"),
    Mutation("calibration: unit dropped when loading", "src/embeddings.py",
             'str(values.get("signal", "native"))', '"centered"'),
]


def run_one(mutation: Mutation) -> bool:
    """Plant one mutation in a temporary copy; True if the test suite failed (killed)."""
    with tempfile.TemporaryDirectory(prefix="vibefinder-mutant-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp) / "project"
        shutil.copytree(ROOT, work, ignore=IGNORE)
        target = work / mutation.path
        source = target.read_text(encoding="utf-8")
        if source.count(mutation.old) != 1:
            raise SystemExit(f"mutation {mutation.name!r}: pattern found {source.count(mutation.old)}x "
                             f"in {mutation.path}; update tools/mutate.py to match the code")
        target.write_text(source.replace(mutation.old, mutation.new), encoding="utf-8")
        proc = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                              cwd=work, capture_output=True, text=True)
        return proc.returncode != 0


def main(argv: List[str]) -> int:
    wanted = [m for m in MUTATIONS if not argv or any(a in m.name for a in argv)]
    survivors = []
    for mutation in wanted:
        killed = run_one(mutation)
        print(f"{'killed  ' if killed else 'SURVIVED'}  {mutation.name}", flush=True)
        if not killed:
            survivors.append(mutation.name)
    print(f"\n{len(wanted) - len(survivors)}/{len(wanted)} mutations killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
