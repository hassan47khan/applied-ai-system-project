"""
Which model generates: the one switch for understanding, rerank and explanations.

    shared_llm()             the generation client for VIBEFINDER_LLM (gemini | nvidia)
    shared_llm("nvidia")     a specific provider, e.g. from the CLI or the Streamlit selector

Embeddings never go through here. Dense retrieval always uses `gemini.shared_client()`,
because the committed card index and its calibration belong to the Gemini embedding model.
"""

from typing import Dict, Optional

from .config import LLM_PROVIDERS, get_settings
from .llm_base import LLMClientBase

_shared: Dict[str, LLMClientBase] = {}

PROVIDER_NAMES = {"gemini": "Gemini", "nvidia": "NVIDIA Nemotron"}


def make_llm(provider: Optional[str] = None, settings=None) -> LLMClientBase:
    """A fresh generation client for `provider` (default: the configured one)."""
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider).lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; choose from {LLM_PROVIDERS}")
    if provider == "nvidia":
        from .nvidia import NvidiaClient
        return NvidiaClient(settings=settings)
    from .gemini import GeminiClient
    return GeminiClient(settings=settings)


def shared_llm(provider: Optional[str] = None) -> LLMClientBase:
    """Process-wide generation client per provider, so one cache and one set of stats each.

    For Gemini this is the same object as `gemini.shared_client()`, so the generation and
    embedding calls of a Gemini run are counted together, as they were before NVIDIA existed.
    """
    provider = (provider or get_settings().llm_provider).lower()
    if provider == "gemini":
        from .gemini import shared_client
        return shared_client()
    if provider not in _shared:
        _shared[provider] = make_llm(provider)
    return _shared[provider]


def reset_shared_llm(provider: Optional[str] = None, client: Optional[LLMClientBase] = None) -> None:
    """Drop (or replace) the cached client; tests use this to inject a fake."""
    if provider is None:
        _shared.clear()
        return
    if client is None:
        _shared.pop(provider, None)
    else:
        _shared[provider] = client


def describe(client: LLMClientBase) -> str:
    """'NVIDIA Nemotron (nvidia/nemotron-3-super-120b-a12b)' once a model has answered."""
    name = PROVIDER_NAMES.get(client.provider, client.provider)
    return f"{name} ({client.model})" if client.model else name
