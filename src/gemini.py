"""
One door to Gemini. Every model call in VibeFinder goes through `GeminiClient`, which adds:

  * lazy client construction (the app imports and runs without `google-genai` or a key)
  * retry with exponential backoff on rate limits (429) and server errors (5xx)
  * a model fallback chain: if Google renames or retires a model (404), try the next one
  * a disk cache keyed by sha256(model + request), so repeated requests cost nothing and
    the eval harness is reproducible even on a tight free-tier quota
  * call accounting (calls, cache hits, failures, tokens, latency) for the trace and the UI

Anything that can't be recovered raises `GeminiUnavailable`. Callers catch it and fall back
to the offline path, so the LLM is never a single point of failure.
"""

import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm_base import (  # noqa: F401  (re-exported: older code and tests import these from here)
    MODEL_GONE_CODES, TRANSIENT_CODES, CallStats, DiskCache, LLMClientBase, LLMUnavailable,
    _describe, _error_code, _is_transient,
)

log = logging.getLogger("vibefinder.gemini")

# Kept as the name every caller already catches. It is the provider-neutral LLMUnavailable,
# so `except GeminiUnavailable` also catches a failure from any other provider.
GeminiUnavailable = LLMUnavailable


def l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else list(vec)


class GeminiClient(LLMClientBase):
    provider = "gemini"
    label = "Gemini"

    # ------------------------------------------------------------------ setup
    @property
    def available(self) -> bool:
        """True if a call could be attempted (a client was injected, or a key is set)."""
        return self._client is not None or self.settings.gemini_enabled

    def unavailable_reason(self) -> str:
        if self._client is not None or self.settings.gemini_enabled:
            return ""
        if not self.settings.use_gemini:
            return "disabled"
        return "no key"

    def _sdk(self):
        if self._client is None:
            if not self.settings.gemini_enabled:
                raise GeminiUnavailable(self.unavailable_reason() or "no key")
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise GeminiUnavailable("google-genai not installed") from exc
            self._client = genai.Client(
                api_key=self.settings.api_key,
                http_options=types.HttpOptions(timeout=int(self.settings.request_timeout_s * 1000)),
            )
        return self._client

    def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        *,
        purpose: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> Any:
        """Ask for JSON matching `schema`; return the parsed JSON. Raises GeminiUnavailable."""
        key = DiskCache.key("generate_json", self.settings.generation_models, system, prompt, schema, temperature)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if not self.available:
            raise GeminiUnavailable(self.unavailable_reason())

        def config():
            from google.genai import types
            return types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=schema,
            )

        try:
            response = self._generate(prompt, config, purpose)
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                raise GeminiUnavailable("empty response")
            parsed = json.loads(text)
        except GeminiUnavailable as exc:
            self.stats.failures += 1
            self.stats.last_error = exc.reason
            raise
        except ValueError as exc:
            raise self._record_failure(purpose, exc) from exc
        except Exception as exc:
            raise self._record_failure(purpose, exc) from exc
        self.cache.put(key, parsed)
        return parsed

    def _generate(self, prompt: str, config_factory: Callable[[], Any], purpose: str):
        sdk = self._sdk()
        models = [self._gen_model] if self._gen_model else list(self.settings.generation_models)
        last_exc: Optional[BaseException] = None
        for model in models:
            self._count(purpose)
            try:
                response = self._call_with_retry(purpose, lambda: sdk.models.generate_content(
                    model=model, contents=prompt, config=config_factory()))
            except Exception as exc:
                if _error_code(exc) in MODEL_GONE_CODES:
                    log.warning("Gemini model %s not available; trying the next one", model)
                    last_exc = exc
                    continue
                raise
            self._gen_model = model
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                self.stats.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
                self.stats.output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
            return response
        raise last_exc or GeminiUnavailable("no generation model available")

    # -------------------------------------------------------------- embeddings
    @staticmethod
    def format_for_embedding(text: str, kind: str, model: str, title: str = "") -> str:
        """gemini-embedding-2 has no task_type, so the task goes in the text itself."""
        if model.startswith("gemini-embedding-2"):
            if kind == "query":
                return f"task: search result | query: {text}"
            return f"title: {title or 'none'} | text: {text}"
        return text

    def embed(self, texts: Sequence[str], *, kind: str, model: Optional[str] = None,
              titles: Optional[Sequence[str]] = None, purpose: str = "embed") -> List[List[float]]:
        """L2-normalised embeddings, one per text. kind is "query" or "document"."""
        if kind not in {"query", "document"}:
            raise ValueError("kind must be 'query' or 'document'")
        models = [model] if model else list(self.settings.embedding_models)
        titles = list(titles) if titles is not None else [""] * len(texts)
        last_exc: Optional[BaseException] = None
        for name in models:
            try:
                return self._embed_with(name, list(texts), kind, titles, purpose)
            except GeminiUnavailable:
                raise
            except Exception as exc:
                if _error_code(exc) in MODEL_GONE_CODES and len(models) > 1:
                    log.warning("Embedding model %s not available; trying the next one", name)
                    last_exc = exc
                    continue
                raise self._record_failure(purpose, exc) from exc
        raise self._record_failure(purpose, last_exc or RuntimeError("no embedding model"))

    def embedding_model(self) -> str:
        return self.settings.embedding_models[0]

    def _embed_with(self, model: str, texts: List[str], kind: str, titles: List[str],
                    purpose: str) -> List[List[float]]:
        dim = self.settings.embedding_dim
        keys = [DiskCache.key("embed", model, dim, kind, t, title) for t, title in zip(texts, titles)]
        out: List[Optional[List[float]]] = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        self.stats.cache_hits += len(texts) - len(missing)
        if missing:
            if not self.available:
                raise GeminiUnavailable(self.unavailable_reason())
            sdk = self._sdk()
            from google.genai import types
            formatted = [self.format_for_embedding(texts[i], kind, model, titles[i]) for i in missing]
            if model.startswith("gemini-embedding-2"):
                # Each text in its own Content, or the API returns one aggregated vector.
                contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in formatted]
                config = types.EmbedContentConfig(output_dimensionality=dim)
            else:
                contents = formatted
                config = types.EmbedContentConfig(
                    output_dimensionality=dim,
                    task_type="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT")
            self._count(purpose)
            response = self._call_with_retry(purpose, lambda: sdk.models.embed_content(
                model=model, contents=contents, config=config))
            vectors = [l2_normalize(e.values) for e in response.embeddings]
            if len(vectors) != len(missing):
                raise GeminiUnavailable(f"expected {len(missing)} embeddings, got {len(vectors)}")
            for i, vec in zip(missing, vectors):
                out[i] = vec
                self.cache.put(keys[i], vec)
        return [v for v in out if v is not None]


_shared: Optional[GeminiClient] = None


def shared_client() -> GeminiClient:
    """Process-wide client so the CLI, agent and UI share one cache and one set of stats."""
    global _shared
    if _shared is None:
        _shared = GeminiClient()
    return _shared


def reset_shared_client(client: Optional[GeminiClient] = None) -> None:
    global _shared
    _shared = client
