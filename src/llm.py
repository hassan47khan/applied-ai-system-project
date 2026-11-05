"""
Explanation layer.

TemplateExplainer is the default: deterministic, offline, and grounded in the score
reasons plus the retrieved context card.

GeminiExplainer is optional (set GEMINI_API_KEY and pass --llm). It only rewrites the
facts it is handed into friendlier prose. Guardrails:
  * the prompt contains ONLY song fields, score reasons and the card's guidance text;
    the user's raw request is never included, so injected instructions can't reach it
  * the reply must be short and mention the song title, otherwise we fall back to the template
  * any API error falls back to the template, so the app never fails because of the LLM
"""

import logging
import os
from typing import Dict, List, Optional, Protocol

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


def make_explainer(use_llm: bool) -> Explainer:
    """Gemini if requested and available, otherwise the offline template."""
    if not use_llm:
        return TemplateExplainer()
    try:
        return GeminiExplainer()
    except Exception as exc:
        log.warning("LLM explanations unavailable (%s); using offline templates", exc)
        return TemplateExplainer()
