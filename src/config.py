"""
Runtime settings: where the data lives, which Gemini models to use, which retrieval mode.

Everything comes from environment variables (optionally loaded from a `.env` file in the
project root), so the app, the tests and the UI can each point at different data without
code changes. Scoring and verification thresholds stay next to the code that uses them
(agent.py, recommender.py) because they are tuned together with it.

    GEMINI_API_KEY            enables every Gemini feature; without it the app runs offline
    VIBEFINDER_USE_GEMINI     "0" forces the offline path even when a key is set
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
GENERATION_MODELS: Tuple[str, ...] = ("gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash")
EMBEDDING_MODELS: Tuple[str, ...] = ("gemini-embedding-2", "gemini-embedding-001")
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
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY") or None)
    generation_models: Tuple[str, ...] = GENERATION_MODELS
    embedding_models: Tuple[str, ...] = EMBEDDING_MODELS
    embedding_dim: int = EMBEDDING_DIM
    request_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"VIBEFINDER_RETRIEVAL must be one of {RETRIEVAL_MODES}, "
                             f"got {self.retrieval_mode!r}")

    @property
    def gemini_enabled(self) -> bool:
        return self.use_gemini and bool(self.api_key)


def get_settings(**overrides) -> Settings:
    """Fresh settings from the current environment, with optional overrides."""
    return Settings(**overrides)
