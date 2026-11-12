"""A stand-in for the google-genai SDK client, so Gemini code paths run with no network.

    FakeSDK(json_replies=[...])        generate_content returns these JSON payloads in order
                                       (or a callable(prompt) -> payload)
    FakeSDK(errors=[exc, ...])         raise these first, then answer normally
    embed_content                      deterministic "concept" vectors: words in the same
                                       CONCEPTS group land on the same axis, so synonyms
                                       ("bummed", "sad") embed close together and the
                                       hybrid/dense tests can check semantic recall.
"""

import json
import math
import re
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Union

CONCEPTS = [
    {"sad", "bummed", "down", "blue", "melancholy", "gloomy", "cry", "rain", "rainy", "lonely", "dumped",
     "heartbreak", "triste"},
    {"gym", "workout", "lifting", "weights", "pump", "training", "treadmill", "run", "running", "jog",
     "joggen", "cardio"},
    {"study", "studying", "homework", "exam", "focus", "concentrate", "estudiar", "library"},
    {"sleep", "bedtime", "asleep", "dormir", "tired", "insomnia", "nap"},
    {"party", "dance", "dancing", "club", "fiesta", "celebrate"},
    {"angry", "mad", "livid", "furious", "rage", "vent"},
    {"beach", "summer", "sun", "playa", "ocean"},
]


class APIError(Exception):
    def __init__(self, code: int, message: str = "fake"):
        super().__init__(f"{code} {message}")
        self.code = code


def concept_vector(text: str) -> List[float]:
    words = set(re.findall(r"[a-záéíóúüñ]+", text.lower()))
    vec = [float(len(words & group)) for group in CONCEPTS] + [0.3]   # small shared bias axis
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class _Models:
    def __init__(self, sdk: "FakeSDK"):
        self.sdk = sdk

    def _maybe_raise(self, model: str):
        if model in self.sdk.missing_models:
            raise APIError(404, f"model {model} not found")
        if self.sdk.errors:
            raise self.sdk.errors.pop(0)

    def generate_content(self, model: str, contents: Any, config: Any = None):
        self.sdk.generate_calls.append({"model": model, "contents": contents, "config": config})
        self._maybe_raise(model)
        reply = self.sdk.json_replies
        if callable(reply):
            payload = reply(contents)
        elif isinstance(reply, list):
            payload = reply.pop(0) if reply else {}
        else:
            payload = reply
        text = payload if isinstance(payload, str) else json.dumps(payload)
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
        return SimpleNamespace(text=text, usage_metadata=usage)

    def embed_content(self, model: str, contents: Any, config: Any = None):
        self.sdk.embed_calls.append({"model": model, "contents": contents, "config": config})
        self._maybe_raise(model)
        texts = []
        for item in contents if isinstance(contents, list) else [contents]:
            parts = getattr(item, "parts", None)
            texts.append(parts[0].text if parts else str(item))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=concept_vector(t)) for t in texts])


class FakeSDK:
    def __init__(self, json_replies: Union[list, dict, str, Callable, None] = None,
                 errors: Optional[list] = None, missing_models: Optional[set] = None):
        self.json_replies = json_replies if json_replies is not None else {}
        self.errors = list(errors or [])
        self.missing_models = set(missing_models or ())
        self.generate_calls: list = []
        self.embed_calls: list = []
        self.models = _Models(self)


def fake_client(tmp_path, sdk: Optional[FakeSDK] = None, **settings):
    """A GeminiClient wired to a FakeSDK and a private cache dir, never sleeping on retry."""
    from src.config import Settings
    from src.gemini import DiskCache, GeminiClient
    s = Settings(cache_dir=tmp_path / "cache", **settings)
    return GeminiClient(settings=s, client=sdk or FakeSDK(), cache=DiskCache(s.cache_dir),
                        sleep=lambda _: None)
