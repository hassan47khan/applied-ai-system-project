"""
Explanation layer.

TemplateExplainer is the default: deterministic, offline, and grounded in the score
reasons plus the retrieved context card.

GroundedBatchExplainer (VibeFinder 3.0, used when GEMINI_API_KEY is set) explains all picks
in one call with per-pick validation; see its docstring.

GeminiExplainer is the original per-song 2.0 explainer, kept for compatibility. It only
rewrites the facts it is handed into friendlier prose. Guardrails:
  * the prompt contains ONLY song fields, score reasons and the card's guidance text;
    the user's raw request is never included, so injected instructions can't reach it
  * the reply must be short and mention the song title, otherwise we fall back to the template
  * any API error falls back to the template, so the app never fails because of the LLM
"""

import logging
import os
import re
from typing import Dict, List, Optional, Protocol, Tuple

from .retrieval import Card

log = logging.getLogger("vibefinder.llm")

GEMINI_MODEL_NAME = "gemini-flash-lite-latest"  # same model the Module 4 DocuBot used
MAX_EXPLANATION_WORDS = 70


class Explainer(Protocol):
    def explain(self, song: Dict, reasons: List[str], card: Optional[Card]) -> str: ...


def _split_reasons(reasons: List[str]):
    positives = [r for r in reasons if "misses" not in r]
    tradeoffs = [r for r in reasons if "misses" in r]
    return positives, tradeoffs


class TemplateExplainer:
    def explain(self, song: Dict, reasons: List[str], card: Optional[Card]) -> str:
        positives, tradeoffs = _split_reasons(reasons)
        text = f"{song['title']} fits because " + ("; ".join(positives[:3]) if positives else
                                                   "it was the closest available option")
        text += "."
        if tradeoffs:
            text += " Trade-off: " + "; ".join(tradeoffs[:2]) + "."
        if card is not None:
            text += f" Context ({card.title}): {card.guidance}"
        return text


class GeminiExplainer:
    def __init__(self, client=None, fallback: Optional[Explainer] = None):
        self.fallback = fallback or TemplateExplainer()
        if client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            from google import genai  # imported lazily so the app runs without the package
            client = genai.Client(api_key=api_key)
        self.client = client

    def explain(self, song: Dict, reasons: List[str], card: Optional[Card]) -> str:
        fallback_text = self.fallback.explain(song, reasons, card)
        prompt = (
            "Write a 1-2 sentence explanation (max 45 words) of why this song was recommended. "
            "Use ONLY the facts below; do not invent details about the song or artist.\n"
            f"Song: {song['title']} by {song['artist']} ({song['genre']}, {song['mood']})\n"
            f"Scoring reasons: {'; '.join(reasons) or 'closest available option'}\n"
            f"Listening guidance: {card.guidance if card else 'none'}\n"
        )
        try:
            response = self.client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
            text = (getattr(response, "text", "") or "").strip()
        except Exception as exc:  # network, quota, auth... never break the app
            log.warning("Gemini explanation failed (%s); using template", type(exc).__name__)
            return fallback_text
        if not text or len(text.split()) > MAX_EXPLANATION_WORDS or song["title"].lower() not in text.lower():
            log.warning("Gemini explanation rejected by validation; using template")
            return fallback_text
        return text


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_BATCH_SYSTEM = (
    "You write short, friendly explanations of why a music recommender picked each song. "
    "Use ONLY the facts given for that song: its title, artist, genre, mood, the scoring reasons and "
    "the listening context. Do not invent anything about the artist, lyrics, era, instruments or "
    "popularity, and do not quote any number that is not in the facts. Each explanation: one or two "
    "sentences, at most 40 words, and it must contain the song title exactly as given."
)


def _facts_numbers(song: Dict, reasons: List[str], card: Optional[Card]) -> set:
    text = " ".join([str(song.get(k, "")) for k in ("title", "artist", "genre", "mood")] + reasons
                    + ([card.title, card.guidance] if card else []))
    found = set(_NUMBER.findall(text))
    for key in ("energy", "danceability", "acousticness", "valence", "tempo_bpm"):
        if key in song:
            found |= {f"{song[key]:g}", f"{song[key]:.2f}", f"{song[key]:.0f}"}
    return found


class GroundedBatchExplainer:
    """One Gemini call explains every pick, each grounded in its own facts and context card.

    Per explanation the validator checks: the title is present, it is short, it cites only
    the card it was given (or none), and it contains no number that isn't in its facts
    (an invented "released in 1998" or "120 BPM" is rejected). Any explanation that fails
    falls back to the template on its own; the rest are kept. Like the 2.0 explainer, the
    prompt holds only catalog facts and card text, never the user's raw request.
    """

    def __init__(self, client, fallback: Optional[Explainer] = None):
        self.client = client
        self.fallback = fallback or TemplateExplainer()
        self.last_sources: List[str] = []

    def explain(self, song: Dict, reasons: List[str], card: Optional[Card]) -> str:
        return self.explain_batch([(song, reasons, card)])[0][0]

    def explain_batch(self, items: List[Tuple[Dict, List[str], Optional[Card]]]) -> List[Tuple[str, str, Optional[str]]]:
        """[(text, source, cited_card_id)] in the same order; source is the provider
        ("gemini" or "nvidia") or "template"."""
        fallbacks = [self.fallback.explain(s, r, c) for s, r, c in items]
        card_ids = sorted({c.id for _, _, c in items if c is not None}) + ["none"]
        blocks = []
        for i, (song, reasons, card) in enumerate(items):
            blocks.append(
                f"[{i}] {song['title']} by {song['artist']} ({song['genre']}, {song['mood']})\n"
                f"    reasons: {'; '.join(reasons) or 'closest available option'}\n"
                f"    context card: {card.id + ' - ' + card.title + ': ' + card.guidance if card else 'none'}")
        schema = {"type": "object", "properties": {"explanations": {"type": "array", "items": {
            "type": "object", "properties": {
                "index": {"type": "integer", "minimum": 0, "maximum": max(0, len(items) - 1)},
                "text": {"type": "string"},
                "cited_card": {"type": "string", "enum": card_ids}},
            "required": ["index", "text", "cited_card"]}}}, "required": ["explanations"]}
        try:
            data = self.client.generate_json("\n".join(blocks), schema, purpose="explain",
                                             system=_BATCH_SYSTEM, temperature=0.3)
        except Exception as exc:   # GeminiUnavailable and anything unexpected
            log.warning("Batched explanations unavailable (%s); using templates", getattr(exc, "reason", exc))
            self.last_sources = ["template"] * len(items)
            return [(t, "template", None) for t in fallbacks]

        out: List[Tuple[str, str, Optional[str]]] = [(t, "template", None) for t in fallbacks]
        entries = data.get("explanations", []) if isinstance(data, dict) else []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            i, text, cited = entry.get("index"), entry.get("text"), entry.get("cited_card")
            if not isinstance(i, int) or not 0 <= i < len(items) or not isinstance(text, str):
                continue
            song, reasons, card = items[i]
            problem = self._check(text.strip(), cited, song, reasons, card)
            if problem:
                log.warning("Explanation for %r rejected: %s", song["title"], problem)
                continue
            source = getattr(self.client, "provider", "gemini")
            out[i] = (text.strip(), source, card.id if card and cited == card.id else None)
        self.last_sources = [src for _, src, _ in out]
        return out

    @staticmethod
    def _check(text: str, cited, song: Dict, reasons: List[str], card: Optional[Card]) -> Optional[str]:
        if not text:
            return "empty"
        if song["title"].lower() not in text.lower():
            return "title missing"
        if len(text.split()) > MAX_EXPLANATION_WORDS:
            return "too long"
        if cited not in {"none", card.id if card else "none"}:
            return f"cites a card it was not given ({cited})"
        invented = set(_NUMBER.findall(text)) - _facts_numbers(song, reasons, card)
        if invented:
            return f"invented numbers {sorted(invented)}"
        return None


def make_explainer(use_llm: bool) -> Explainer:
    """The configured LLM (Gemini or NVIDIA) if requested and available, else the template."""
    if not use_llm:
        return TemplateExplainer()
    from .providers import shared_llm
    client = shared_llm()
    if not client.available:
        log.warning("LLM explanations unavailable (%s); using offline templates", client.unavailable_reason())
        return TemplateExplainer()
    return GroundedBatchExplainer(client)
