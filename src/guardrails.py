"""
Input guardrails. The goal is to never crash on odd input and never let user text
steer anything beyond "what kind of music do I want?".

Since VibeFinder 3.0 the cleaned text DOES reach Gemini (query understanding and rerank).
These checks are the first line of defence; the schema-constrained, validated output in
understanding.py is the second. Detected override attempts are kept as plain text and
flagged, never executed: the model only ever classifies the request.

Distress statements ("I want to hurt myself") are not refused. The user still gets an
answer, plus a short pointer to real help, and the result is flagged for a human.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger("vibefinder.guardrails")

MAX_QUERY_CHARS = 300
_WORD_BACKOFF = 40           # how far back truncation will look for a word boundary
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Zero-width and bidirectional-override characters: invisible, and used to split words so
# "i​gnore previous instructions" slips past a pattern match.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿­]")
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any |every |the |your )?(previous |prior |above |earlier |system )?"
        r"(instructions|prompts|rules)",
        r"disregard (the |your |all )?(system|previous|prior) ",
        r"(reveal|print|output|show|repeat|leak|translate) (me )?(your |the |all |any )?"
        r"(full |hidden |system )*(prompt|instructions)",
        r"you are now ",
        r"\bsystem\s*:",
        r"new (instruction|policy|rule)s?\s*:",
        r"\b(?:api key|environment variables?|env vars?)\b",
    )
]
_DISTRESS = re.compile(
    r"\b(?:kill(?:ing)? myself|hurt(?:ing)? myself|end(?:ing)? (?:my|it) (?:life|all)|suicid\w*|"
    r"self[- ]harm|don'?t want to (?:live|be alive|be here)|want to die)\b", re.IGNORECASE)
SUPPORT_MESSAGE = ("It sounds like you might be going through something really hard. You don't have "
                   "to handle it alone: in the US you can call or text 988 (Suicide & Crisis Lifeline), "
                   "or find a local line at findahelpline.com.")


@dataclass
class CleanedQuery:
    text: str
    warnings: List[str]
    ok: bool
    support: Optional[str] = None   # shown to the user when distress is detected


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

    # NFKC folds full-width and other look-alike letters ("ｉｇｎｏｒｅ" -> "ignore").
    text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", raw))
    text = _CONTROL_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return CleanedQuery("", ["input was empty"], ok=False)

    if len(text) > MAX_QUERY_CHARS:
        text = _truncate(text)
        warnings.append(f"input truncated to {MAX_QUERY_CHARS} characters")

    if any(p.search(text) for p in _INJECTION_PATTERNS):
        warnings.append("input looked like an instruction-override attempt; treated as plain text")

    support = None
    if _DISTRESS.search(text):
        support = SUPPORT_MESSAGE
        warnings.append("possible distress detected; support information shown")

    for w in warnings:
        log.warning("guardrail: %s", w)
    return CleanedQuery(text, warnings, ok=True, support=support)
