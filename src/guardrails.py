"""
Input guardrails. The goal is to never crash on odd input and never let user text
steer anything beyond "what kind of music do I want?".

Note: user text is only ever used for retrieval/keyword matching. It is never placed
into an LLM prompt (see llm.py), so prompt-injection text has nothing to hijack.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

log = logging.getLogger("vibefinder.guardrails")

MAX_QUERY_CHARS = 300
_WORD_BACKOFF = 40           # how far back truncation will look for a word boundary
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above) (instructions|prompts)",
        r"disregard (the |your )?(system|previous) ",
        r"reveal (your |the )?(system )?prompt",
        r"you are now ",
    )
]


@dataclass
class CleanedQuery:
    text: str
    warnings: List[str]
    ok: bool


def _truncate(text: str) -> str:
    """Cut to MAX_QUERY_CHARS without leaving half a word ("stu" from "study") at the end.
    Backs off to the last space only if it is close; one huge 'word' is simply cut."""
    cut = text[:MAX_QUERY_CHARS]
    if text[MAX_QUERY_CHARS] != " ":
        space = cut.rfind(" ")
        if space >= MAX_QUERY_CHARS - _WORD_BACKOFF:
            cut = cut[:space]
    return cut.rstrip()


def sanitize_query(raw) -> CleanedQuery:
    """Validate and normalise a free-text request. Never raises."""
    warnings: List[str] = []
    if not isinstance(raw, str):
        return CleanedQuery("", ["input must be text"], ok=False)

    text = _CONTROL_CHARS.sub(" ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return CleanedQuery("", ["input was empty"], ok=False)

    if len(text) > MAX_QUERY_CHARS:
        text = _truncate(text)
        warnings.append(f"input truncated to {MAX_QUERY_CHARS} characters")

    if any(p.search(text) for p in _INJECTION_PATTERNS):
        warnings.append("input looked like an instruction-override attempt; treated as plain text")

    for w in warnings:
        log.warning("guardrail: %s", w)
    return CleanedQuery(text, warnings, ok=True)
