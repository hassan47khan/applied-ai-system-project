"""The Gemini client wrapper: availability, retries, model fallback, caching, accounting."""

import pytest

from src.config import Settings
from src.gemini import DiskCache, GeminiClient, GeminiUnavailable, l2_normalize
from tests.fakes import APIError, FakeSDK, fake_client

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}}


def test_no_key_means_unavailable_with_a_reason(tmp_path):
    client = GeminiClient(settings=Settings(api_key=None, use_gemini=True, cache_dir=tmp_path))
    assert not client.available and client.unavailable_reason() == "no key"
    with pytest.raises(GeminiUnavailable):
        client.generate_json("hi", SCHEMA, purpose="t")


def test_disabled_flag_wins_over_a_key(tmp_path):
    client = GeminiClient(settings=Settings(api_key="k", use_gemini=False, cache_dir=tmp_path))
    assert not client.available and client.unavailable_reason() == "disabled"


def test_json_reply_is_parsed_and_cached(tmp_path):
    sdk = FakeSDK(json_replies=[{"x": 1}, {"x": 2}])
    client = fake_client(tmp_path, sdk)
    assert client.generate_json("q", SCHEMA, purpose="t") == {"x": 1}
    assert client.generate_json("q", SCHEMA, purpose="t") == {"x": 1}   # cache, not the 2nd reply
    assert len(sdk.generate_calls) == 1 and client.stats.cache_hits == 1


def test_rate_limit_is_retried_then_succeeds(tmp_path):
    sdk = FakeSDK(json_replies={"x": 3}, errors=[APIError(429), APIError(503)])
    sleeps = []
    client = GeminiClient(settings=Settings(cache_dir=tmp_path), client=sdk,
                          cache=DiskCache(tmp_path), sleep=sleeps.append)
    assert client.generate_json("q", SCHEMA, purpose="t") == {"x": 3}
    assert sleeps == [1.0, 2.0]


def test_persistent_rate_limit_becomes_unavailable(tmp_path):
    sdk = FakeSDK(errors=[APIError(429)] * 5)
    client = fake_client(tmp_path, sdk)
    with pytest.raises(GeminiUnavailable) as exc:
        client.generate_json("q", SCHEMA, purpose="t")
    assert "429" in exc.value.reason and client.stats.failures == 1


def test_client_errors_are_not_retried(tmp_path):
    sdk = FakeSDK(errors=[APIError(400)])
    with pytest.raises(GeminiUnavailable):
        fake_client(tmp_path, sdk).generate_json("q", SCHEMA, purpose="t")
    assert len(sdk.generate_calls) == 1


def test_a_retired_model_falls_through_to_the_next(tmp_path):
    sdk = FakeSDK(json_replies={"x": 4}, missing_models={"gemini-flash-latest"})
    client = fake_client(tmp_path, sdk)
    assert client.generate_json("q", SCHEMA, purpose="t") == {"x": 4}
    assert [c["model"] for c in sdk.generate_calls] == ["gemini-flash-latest", "gemini-flash-lite-latest"]
    client.generate_json("q2", SCHEMA, purpose="t")
    assert sdk.generate_calls[-1]["model"] == "gemini-flash-lite-latest"   # remembered


def test_non_json_reply_is_a_failure_not_a_crash(tmp_path):
    with pytest.raises(GeminiUnavailable):
        fake_client(tmp_path, FakeSDK(json_replies="not json")).generate_json("q", SCHEMA, purpose="t")


def test_tokens_and_purposes_are_counted(tmp_path):
    client = fake_client(tmp_path, FakeSDK(json_replies={"x": 1}))
    client.generate_json("q", SCHEMA, purpose="understand")
    assert client.stats.by_purpose == {"understand": 1}
    assert client.stats.prompt_tokens == 10 and client.stats.output_tokens == 5


def test_embedding_model_fallback(tmp_path):
    sdk = FakeSDK(missing_models={"gemini-embedding-2"})
    vecs = fake_client(tmp_path, sdk).embed(["sad"], kind="query")
    assert len(vecs) == 1 and sdk.embed_calls[-1]["model"] == "gemini-embedding-001"
    assert sdk.embed_calls[-1]["config"].task_type == "RETRIEVAL_QUERY"


def test_embed_rejects_an_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        fake_client(tmp_path).embed(["x"], kind="banana")


def test_corrupt_cache_entries_are_misses(tmp_path):
    cache = DiskCache(tmp_path)
    key = DiskCache.key("a")
    cache.put(key, {"ok": True})
    (tmp_path / key[:2] / f"{key}.json").write_text("{broken", encoding="utf-8")
    assert cache.get(key) is None


def test_l2_normalize():
    assert l2_normalize([3, 4]) == [0.6, 0.8]
    assert l2_normalize([0, 0]) == [0, 0]


def test_dotenv_never_overrides_the_real_environment(tmp_path, monkeypatch):
    from src.config import load_dotenv
    env = tmp_path / ".env"
    env.write_text("VF_TEST_A=from_file\nVF_TEST_B='quoted'\n# comment\nnot a pair\n", encoding="utf-8")
    monkeypatch.setenv("VF_TEST_A", "from_env")
    monkeypatch.delenv("VF_TEST_B", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["VF_TEST_A"] == "from_env" and os.environ["VF_TEST_B"] == "quoted"
    monkeypatch.delenv("VF_TEST_B")


def test_invalid_retrieval_mode_is_rejected():
    with pytest.raises(ValueError):
        Settings(retrieval_mode="magic")
