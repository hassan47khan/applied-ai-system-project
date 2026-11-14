"""
Runtime settings: where the data lives, which Gemini models to use, which retrieval mode.

Everything comes from environment variables (optionally loaded from a `.env` file in the
project root), so the app, the tests and the UI can each point at different data without
code changes. Scoring and verification thresholds stay next to the code that uses them
(agent.py, recommender.py) because they are tuned together with it.

    GEMINI_API_KEY            enables every Gemini feature; without it the app runs offline
    VIBEFINDER_USE_GEMINI     "0" forces the offline path even when a key is set (every
                              provider: the name predates NVIDIA support)
    VIBEFINDER_LLM            gemini | nvidia: which model understands requests, reranks and
                              writes explanations (default: gemini)
    VIBEFINDER_LLM_MODEL      try this model first for the chosen provider
    NVIDIA_API_KEY            enables the NVIDIA (Nemotron) provider; from build.nvidia.com
    NVIDIA_BASE_URL           OpenAI-compatible endpoint (default: NVIDIA's hosted API)
    NVIDIA_MODELS             comma-separated fallback chain for NVIDIA
    VIBEFINDER_RETRIEVAL      tfidf | bm25 | dense | hybrid   (default: hybrid)
    VIBEFINDER_RERANK         "1" adds a Gemini rerank of the top cards (default: off)
    VIBEFINDER_SONGS          path to the song catalog CSV
    VIBEFINDER_KNOWLEDGE      directory of knowledge cards
    VIBEFINDER_INDEX          directory of the prebuilt embedding index
    VIBEFINDER_CACHE_DIR      where Gemini responses are cached on disk
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

RETRIEVAL_MODES = ("tfidf", "bm25", "dense", "hybrid")

# First available model wins; later entries are fallbacks if Google renames or retires one.
# Measured 2026-11-14 against a free AI Studio key: the "-latest" aliases 404 on API version
# v1, and gemini-2.5-flash(-lite) answer "no longer available to new users". The 3.5 pair is
# what a key issued now can actually call, so it leads the chain; the older names stay as
# fallbacks for keys that still have access to them.
GENERATION_MODELS: Tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash",
)
EMBEDDING_MODELS: Tuple[str, ...] = ("gemini-embedding-2", "gemini-embedding-001")

LLM_PROVIDERS = ("gemini", "nvidia")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Nemotron 3, served on NVIDIA's hosted OpenAI-compatible API (build.nvidia.com). Ultra is the
# chosen model for this project; Super stays as the fallback if Ultra is retired or overloaded.
# Model ids get retired (every Gemini id in the first chain did), so this is a chain, and
# NVIDIA_MODELS overrides it without a code change.
NVIDIA_MODELS: Tuple[str, ...] = (
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
)
EMBEDDING_DIM = 768


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Real environment variables always win."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value) if value else default


_PLACEHOLDER_KEYS = {"your_api_key_here", "your-api-key", "changeme", "xxx", "<your key>"}


def _real_key(value: Optional[str]) -> Optional[str]:
    """None for a missing key or the .env.example placeholder (a doomed API call per request)."""
    value = (value or "").strip()
    return None if not value or value.lower() in _PLACEHOLDER_KEYS else value


def _chain(env: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """A comma-separated model list from the environment, or the default chain."""
    names = tuple(n.strip() for n in (os.environ.get(env) or "").split(",") if n.strip())
    return names or default


def _first(model: Optional[str], chain: Tuple[str, ...]) -> Tuple[str, ...]:
    """Put `model` at the head of `chain`, keeping the rest as fallbacks, without duplicates."""
    if not model:
        return chain
    return (model,) + tuple(m for m in chain if m != model)


def _flag(env: str, default: bool) -> bool:
    value = os.environ.get(env)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class Settings:
    songs_path: Path = field(default_factory=lambda: _path("VIBEFINDER_SONGS", ROOT / "data" / "songs.csv"))
    knowledge_dir: Path = field(default_factory=lambda: _path("VIBEFINDER_KNOWLEDGE", ROOT / "data" / "knowledge"))
    index_dir: Path = field(default_factory=lambda: _path("VIBEFINDER_INDEX", ROOT / "data" / "index"))
    cache_dir: Path = field(default_factory=lambda: _path("VIBEFINDER_CACHE_DIR", ROOT / ".cache" / "gemini"))
    retrieval_mode: str = field(default_factory=lambda: os.environ.get("VIBEFINDER_RETRIEVAL", "hybrid"))
    rerank: bool = field(default_factory=lambda: _flag("VIBEFINDER_RERANK", False))
    use_gemini: bool = field(default_factory=lambda: _flag("VIBEFINDER_USE_GEMINI", True))
    api_key: Optional[str] = field(default_factory=lambda: _real_key(os.environ.get("GEMINI_API_KEY")))
    generation_models: Tuple[str, ...] = GENERATION_MODELS
    embedding_models: Tuple[str, ...] = EMBEDDING_MODELS
    embedding_dim: int = EMBEDDING_DIM
    request_timeout_s: float = 20.0
    nvidia_timeout_s: float = 90.0      # the hosted 550B Ultra answers slower than Gemini Flash
    llm_provider: str = field(default_factory=lambda: (os.environ.get("VIBEFINDER_LLM") or "gemini")
                              .strip().lower())
    llm_model: Optional[str] = None     # None: VIBEFINDER_LLM_MODEL, if it was set for this provider
    nvidia_api_key: Optional[str] = field(default_factory=lambda: _real_key(os.environ.get("NVIDIA_API_KEY")))
    nvidia_base_url: str = field(default_factory=lambda: os.environ.get("NVIDIA_BASE_URL") or NVIDIA_BASE_URL)
    nvidia_models: Tuple[str, ...] = field(default_factory=lambda: _chain("NVIDIA_MODELS", NVIDIA_MODELS))

    def __post_init__(self) -> None:
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"VIBEFINDER_RETRIEVAL must be one of {RETRIEVAL_MODES}, "
                             f"got {self.retrieval_mode!r}")
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"VIBEFINDER_LLM must be one of {LLM_PROVIDERS}, got {self.llm_provider!r}")
        # VIBEFINDER_LLM_MODEL names a model of VIBEFINDER_LLM's provider. Settings built for a
        # different provider (the CLI's --provider, the UI's selector) must not inherit it, or
        # an NVIDIA model name would be sent to Gemini first. An explicit llm_model always wins.
        if self.llm_model is None:
            env_provider = (os.environ.get("VIBEFINDER_LLM") or "gemini").strip().lower()
            if env_provider == self.llm_provider:
                self.llm_model = (os.environ.get("VIBEFINDER_LLM_MODEL") or "").strip() or None
        # The override goes ahead of that provider's own fallbacks, never into the other's chain.
        if self.llm_provider == "gemini":
            self.generation_models = _first(self.llm_model, self.generation_models)
        else:
            self.nvidia_models = _first(self.llm_model, self.nvidia_models)

    @property
    def gemini_enabled(self) -> bool:
        return self.use_gemini and bool(self.api_key)

    @property
    def nvidia_enabled(self) -> bool:
        # The offline switch is provider-neutral: --offline means no model is called at all.
        return self.use_gemini and bool(self.nvidia_api_key)

    @property
    def llm_enabled(self) -> bool:
        """Can the chosen generation provider be called?"""
        return self.nvidia_enabled if self.llm_provider == "nvidia" else self.gemini_enabled


def get_settings(**overrides) -> Settings:
    """Fresh settings from the current environment, with optional overrides."""
    return Settings(**overrides)
