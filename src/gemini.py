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

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import Settings, get_settings

log = logging.getLogger("vibefinder.gemini")

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
MODEL_GONE_CODES = {404}


class GeminiUnavailable(RuntimeError):
    """Gemini can't answer right now (no key, quota, network, bad output...)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class CallStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    last_error: Optional[str] = None
    by_purpose: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _error_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    code = _error_code(exc)
    if code in TRANSIENT_CODES:
        return True
    # httpx/httpx2 timeouts and connection errors don't carry a code.
    return type(exc).__name__ in {"ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError"}


def _describe(exc: BaseException) -> str:
    code = _error_code(exc)
    return f"{type(exc).__name__}{f' {code}' if code else ''}"


def l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else list(vec)


class DiskCache:
    """sha256-keyed JSON files. Corrupt or unreadable entries are treated as misses."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # a read-only disk must not break recommendations
            log.warning("Could not write Gemini cache entry: %s", exc)


class GeminiClient:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Any = None,
        cache: Optional[DiskCache] = None,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings or get_settings()
        self._client = client
        self.cache = cache if cache is not None else DiskCache(self.settings.cache_dir)
        self.max_attempts = max(1, max_attempts)
        self._sleep = sleep
        self.stats = CallStats()
        self._gen_model: Optional[str] = None     # first model in the chain that answered

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

    # ---------------------------------------------------------------- calling
    def _call_with_retry(self, purpose: str, fn: Callable[[], Any]) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_attempts + 1):
            start = time.perf_counter()
            try:
                result = fn()
                self.stats.latency_ms += (time.perf_counter() - start) * 1000
                return result
            except GeminiUnavailable:
                raise
            except Exception as exc:  # the SDK raises many types; classify them
                self.stats.latency_ms += (time.perf_counter() - start) * 1000
                if _is_transient(exc) and attempt < self.max_attempts:
                    log.warning("Gemini %s failed (%s); retry %d in %.0fs",
                                purpose, _describe(exc), attempt, delay)
                    self._sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _record_failure(self, purpose: str, exc: BaseException) -> GeminiUnavailable:
        self.stats.failures += 1
        self.stats.last_error = _describe(exc)
        log.warning("Gemini %s unavailable (%s); falling back", purpose, self.stats.last_error)
        return GeminiUnavailable(self.stats.last_error)

    def _count(self, purpose: str) -> None:
        self.stats.calls += 1
        self.stats.by_purpose[purpose] = self.stats.by_purpose.get(purpose, 0) + 1

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
