"""
What every LLM provider client shares: failure type, call accounting, the disk cache, and
retry with backoff.

`GeminiClient` (src/gemini.py) and `NvidiaClient` (src/nvidia.py) both build on this, so
switching provider changes which model answers, never how failures, retries, caching or the
offline fallback behave. Callers only ever see `generate_json(...)` and `LLMUnavailable`.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import Settings, get_settings

log = logging.getLogger("vibefinder.llm")

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
MODEL_GONE_CODES = {404}
# Timeouts and dropped connections carry no HTTP status. Names cover httpx / httpx2 (used by
# google-genai) and the OpenAI-compatible SDK used for NVIDIA.
TRANSIENT_NAMES = {"ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
                   "APITimeoutError", "APIConnectionError"}


class LLMUnavailable(RuntimeError):
    """The model can't answer right now (no key, quota, network, bad output...)."""

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
    """The HTTP status of an SDK error. google-genai puts it on `.code`; the OpenAI-compatible
    SDK puts it on `.status_code` and leaves `.code` as None. Reading only `.code` would mean an
    NVIDIA 429 is never retried and a retired-model 404 never falls through to the next model."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if _error_code(exc) in TRANSIENT_CODES:
        return True
    return type(exc).__name__ in TRANSIENT_NAMES


def _describe(exc: BaseException) -> str:
    code = _error_code(exc)
    return f"{type(exc).__name__}{f' {code}' if code else ''}"


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
            log.warning("Could not write LLM cache entry: %s", exc)


class LLMClientBase:
    """Retry, failure recording and accounting shared by every provider."""

    provider = "llm"        # short id used in traces and labels: "gemini", "nvidia"
    label = "LLM"           # display name used in log lines and the UI

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

    @property
    def model(self) -> Optional[str]:
        """The model that answered last, once one has; useful in traces and the UI."""
        return self._gen_model

    def _call_with_retry(self, purpose: str, fn: Callable[[], Any]) -> Any:
        delay = 1.0
        for attempt in range(1, self.max_attempts + 1):
            start = time.perf_counter()
            try:
                result = fn()
                self.stats.latency_ms += (time.perf_counter() - start) * 1000
                return result
            except LLMUnavailable:
                raise
            except Exception as exc:  # SDKs raise many types; classify them
                self.stats.latency_ms += (time.perf_counter() - start) * 1000
                if _is_transient(exc) and attempt < self.max_attempts:
                    log.warning("%s %s failed (%s); retry %d in %.0fs",
                                self.label, purpose, _describe(exc), attempt, delay)
                    self._sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _record_failure(self, purpose: str, exc: BaseException) -> LLMUnavailable:
        self.stats.failures += 1
        self.stats.last_error = _describe(exc)
        log.warning("%s %s unavailable (%s); falling back", self.label, purpose, self.stats.last_error)
        return LLMUnavailable(self.stats.last_error)

    def _count(self, purpose: str) -> None:
        self.stats.calls += 1
        self.stats.by_purpose[purpose] = self.stats.by_purpose.get(purpose, 0) + 1
