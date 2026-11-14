"""
NVIDIA Nemotron as a generation provider, through NVIDIA's OpenAI-compatible API.

Drop-in for the generation half of `GeminiClient`: same `generate_json(...)`, same retry,
model fallback, disk cache, accounting and `LLMUnavailable` (all from `llm_base`). Query
understanding, the optional rerank and explanations can therefore run on either provider,
and every safety check that treats model output as untrusted applies unchanged: the
schema validator in understanding.py, the rerank enum, and the per-explanation checks in
llm.py all see the same parsed JSON whichever model wrote it.

What it deliberately does *not* do is embeddings. Dense retrieval stays on the committed
Gemini embedding index, because that index and its calibration are tied to the embedding
model that built them; swapping the embedder silently would mis-scale every evidence
threshold, which is exactly the failure fixed in d0bd3fb.

Nemotron specifics, from NVIDIA's model pages:
  * Nemotron 3 is a reasoning model. Thinking is switched off per request with
    `chat_template_kwargs: {"enable_thinking": False}`: these are short JSON tasks, and
    reasoning would add latency and tokens without improving a schema-constrained answer.
  * Schema-constrained output is requested with `nvext: {"guided_json": schema}`. If an
    endpoint rejects that extension, the request is retried once without it. The schema is
    also stated in the prompt, and the reply is parsed and validated locally either way, so
    guided decoding is a convenience here, never the defence.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .llm_base import MODEL_GONE_CODES, DiskCache, LLMClientBase, LLMUnavailable, _error_code

log = logging.getLogger("vibefinder.nvidia")

# A 400/422 that names one of these is the endpoint refusing the request's extensions, not
# the request itself.
_EXTENSION_ERRORS = ("nvext", "guided_json", "guided", "extra")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse the JSON in a chat reply. Raises ValueError if there is none.

    Tolerates what OpenAI-compatible chat models wrap around an answer: a <think> block
    (when reasoning leaks into the content), a ```json fence, or prose before the object.
    """
    text = _THINK.sub("", text or "").strip()
    if text.lower().startswith("<think>"):          # an unterminated reasoning block
        raise ValueError("reply contains only reasoning, no answer")
    text = _FENCE.sub("", text).strip()
    if not text:
        raise ValueError("empty reply")
    try:
        return json.loads(text)
    except ValueError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in reply")
    value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return value


def schema_instruction(schema: Dict[str, Any]) -> str:
    return ("\n\nReply with only a JSON value that matches this JSON Schema, with no prose and "
            "no code fences:\n" + json.dumps(schema, separators=(",", ":"), ensure_ascii=False))


class NvidiaClient(LLMClientBase):
    provider = "nvidia"
    label = "NVIDIA"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guided = True     # send nvext.guided_json until an endpoint refuses it

    # ------------------------------------------------------------------ setup
    @property
    def available(self) -> bool:
        return self._client is not None or self.settings.nvidia_enabled

    def unavailable_reason(self) -> str:
        if self._client is not None or self.settings.nvidia_enabled:
            return ""
        if not self.settings.use_gemini:      # the provider-neutral offline switch
            return "disabled"
        return "no NVIDIA key"

    def _sdk(self):
        if self._client is None:
            if not self.settings.nvidia_enabled:
                raise LLMUnavailable(self.unavailable_reason() or "no NVIDIA key")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise LLMUnavailable("openai package not installed") from exc
            self._client = OpenAI(
                base_url=self.settings.nvidia_base_url,
                api_key=self.settings.nvidia_api_key,
                timeout=self.settings.nvidia_timeout_s,
                max_retries=0,     # retries are ours (llm_base), with backoff and accounting
            )
        return self._client

    # ---------------------------------------------------------------- calling
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
        """Ask for JSON matching `schema`; return the parsed JSON. Raises LLMUnavailable."""
        # "nvidia" and the endpoint are in the key, so the two providers never share answers.
        key = DiskCache.key("generate_json", "nvidia", self.settings.nvidia_base_url,
                            self.settings.nvidia_models, system, prompt, schema, temperature)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if not self.available:
            raise LLMUnavailable(self.unavailable_reason())

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt + schema_instruction(schema)})

        try:
            response = self._generate(messages, schema, temperature, max_output_tokens, purpose)
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise LLMUnavailable("reply truncated at the token limit")
            parsed = extract_json(getattr(choice.message, "content", "") or "")
        except LLMUnavailable as exc:
            self.stats.failures += 1
            self.stats.last_error = exc.reason
            raise
        except Exception as exc:     # ValueError from parsing, or anything from the SDK
            raise self._record_failure(purpose, exc) from exc
        self.cache.put(key, parsed)
        return parsed

    def _request(self, model: str, messages, schema, temperature: float, max_tokens: int, purpose: str):
        sdk = self._sdk()

        def send(guided: bool):
            extra: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
            if guided:
                extra["nvext"] = {"guided_json": schema}
            return self._call_with_retry(purpose, lambda: sdk.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                extra_body=extra))

        try:
            return send(self._guided)
        except Exception as exc:
            refused = (self._guided and _error_code(exc) in {400, 422}
                       and any(word in str(exc).lower() for word in _EXTENSION_ERRORS))
            if not refused:
                raise
            log.warning("NVIDIA endpoint refused guided_json (%s); retrying without it. "
                        "The schema is still in the prompt and the reply is still validated.",
                        _error_code(exc))
            self._guided = False
            return send(False)

    def _generate(self, messages, schema, temperature: float, max_tokens: int, purpose: str):
        models = [self._gen_model] if self._gen_model else list(self.settings.nvidia_models)
        last_exc: Optional[BaseException] = None
        for model in models:
            self._count(purpose)
            try:
                response = self._request(model, messages, schema, temperature, max_tokens, purpose)
            except Exception as exc:
                if _error_code(exc) in MODEL_GONE_CODES:
                    log.warning("NVIDIA model %s not available; trying the next one", model)
                    last_exc = exc
                    continue
                raise
            self._gen_model = model
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.stats.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.stats.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            return response
        raise last_exc or LLMUnavailable("no NVIDIA model available")
