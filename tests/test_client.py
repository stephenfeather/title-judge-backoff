import json
import threading
import time
from dataclasses import fields

import httpx
import pytest

from judge.client import (
    MAX_BACKOFF_S,
    OPERATIONAL_FIELDS,
    RETRY_AFTER_DEFAULT_S,
    VERDICT_AFFECTING_FIELDS,
    Backend,
    JudgeClient,
    LimiterRegistry,
    RateLimiter,
    config_digest,
    limiter_host,
    load_backends,
    rate_limit_penalty,
)
from judge.prompts import PROMPT_VERSION
from judge.schema import Pair, ReasonCode

BACKENDS_TOML = """\
[[backends]]
name = "nvidia-llama31-8b"
base_url = "https://integrate.api.nvidia.com/v1"
model_id = "meta/llama-3.1-8b-instruct"
rpm = 40
eval_only = true
api_key_env = "NVIDIA_API_KEY"

[[backends]]
name = "deepinfra-qwen"
base_url = "https://api.deepinfra.com/v1/openai"
model_id = "Qwen/Qwen2.5-72B-Instruct"
rpm = 120
eval_only = false

[[backends]]
name = "anthropic-haiku"
base_url = "https://api.anthropic.com/v1"
model_id = "claude-haiku-4-5"
rpm = 50
eval_only = false
api = "anthropic"
role = "contender"
api_key_env = "ANTHROPIC_API_KEY"
"""

PAIR = Pair(
    id="p1",
    original="acme widget 3000 blk",
    enriched="Acme Widget 3000, Black",
    brand="Acme",
    mpn="W3000-BLK",
    ground_truth="approve",
    reason=ReasonCode.OK,
)


def test_load_backends_parses_all_fields(tmp_path):
    path = tmp_path / "backends.toml"
    path.write_text(BACKENDS_TOML)
    backends = load_backends(path)
    assert backends[0] == Backend(
        name="nvidia-llama31-8b",
        base_url="https://integrate.api.nvidia.com/v1",
        model_id="meta/llama-3.1-8b-instruct",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
        api="openai",
        role="contender",
    )


def test_backend_timeout_defaults_above_observed_slow_model_latency():
    # thinkingmachines/inkling took 79s to answer; a 60s timeout reported that
    # as "unreachable", which is a wrong diagnosis rather than a slow one.
    backend = Backend(
        name="slow",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="K",
    )
    assert backend.timeout_s >= 120.0


def test_load_backends_reads_per_backend_timeout(tmp_path):
    path = tmp_path / "backends.toml"
    path.write_text(BACKENDS_TOML + '\ntimeout_s = 300.0\n')
    assert load_backends(path)[-1].timeout_s == 300.0


def test_load_backends_reads_api_and_role(tmp_path):
    path = tmp_path / "backends.toml"
    path.write_text(BACKENDS_TOML)
    anthropic = load_backends(path)[2]
    assert anthropic.api == "anthropic"
    assert anthropic.role == "contender"


def test_load_backends_rejects_unknown_api(tmp_path):
    path = tmp_path / "backends.toml"
    path.write_text(BACKENDS_TOML.replace('api = "anthropic"', 'api = "telepathy"'))
    with pytest.raises(ValueError, match="api"):
        load_backends(path)


def test_load_backends_defaults_api_key_env_from_name(tmp_path):
    path = tmp_path / "backends.toml"
    path.write_text(BACKENDS_TOML)
    backends = load_backends(path)
    assert backends[1].api_key_env == "DEEPINFRA_QWEN_API_KEY"


def test_rate_limiter_spaces_calls_by_rpm():
    clock = {"now": 100.0}
    sleeps = []
    limiter = RateLimiter(rpm=30, now=lambda: clock["now"], sleep=sleeps.append)
    limiter.wait()          # first call: no wait
    limiter.wait()          # second call at same instant: full 2s interval
    clock["now"] += 0.5
    limiter.wait()          # 0.5s later: wait the remaining 1.5s
    assert sleeps == [2.0, 1.5]


def limiter_backend(name, base_url, rpm):
    return Backend(
        name=name,
        base_url=base_url,
        model_id=f"model/{name}",
        rpm=rpm,
        eval_only=False,
        api_key_env="NVIDIA_API_KEY",
    )


def test_rate_limiter_does_not_release_concurrent_callers_together():
    # The serial limiter reads _last, sleeps, then writes _last. N threads can
    # all read the SAME _last, all sleep one interval, and all fire at once —
    # which is how two NVIDIA backends asked one host for 80 rpm and got 503s.
    # Real clock on purpose: a virtual clock advanced by a fake sleep cannot
    # observe simultaneity, and simultaneity is the whole bug.
    interval = 0.02
    limiter = RateLimiter(rpm=int(60 / interval))
    limiter.wait()  # prime _last so every thread races the same stale value
    gate = threading.Barrier(6)
    released = []
    lock = threading.Lock()

    def worker():
        gate.wait()
        limiter.wait()
        with lock:
            released.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Six calls one interval apart span five intervals. Unlocked they all land
    # within roughly one, so the span is the discriminator, not the count.
    span = max(released) - min(released)
    assert span >= 4 * interval, f"6 calls released within {span:.3f}s of each other"


def test_limiter_registry_shares_one_limiter_across_backends_on_the_same_host():
    # Six nvidia-* backends share integrate.api.nvidia.com and its quota. Keyed
    # per backend they would ask for 6x40 rpm; keyed per host they ask for 40.
    registry = LimiterRegistry(sleep=lambda _: None)
    deepseek = limiter_backend("nvidia-deepseek", "https://integrate.api.nvidia.com/v1", 40)
    nemotron = limiter_backend("nvidia-nemotron", "https://integrate.api.nvidia.com/v1", 40)
    assert registry.for_backend(deepseek) is registry.for_backend(nemotron)


def test_limiter_registry_separates_unrelated_hosts():
    registry = LimiterRegistry(sleep=lambda _: None)
    nvidia = limiter_backend("nvidia-deepseek", "https://integrate.api.nvidia.com/v1", 40)
    xai = limiter_backend("xai-grok", "https://api.x.ai/v1", 60)
    assert registry.for_backend(nvidia) is not registry.for_backend(xai)


def test_limiter_registry_ignores_path_when_keying_on_host():
    # Two base_urls on one host with different paths are still one quota.
    registry = LimiterRegistry(sleep=lambda _: None)
    a = limiter_backend("a", "https://generativelanguage.googleapis.com/v1beta/openai", 60)
    b = limiter_backend("b", "https://generativelanguage.googleapis.com/v1", 60)
    assert registry.for_backend(a) is registry.for_backend(b)


def test_limiter_registry_tightens_to_the_lowest_rpm_on_a_shared_host():
    # The shared limiter must never exceed the most conservative declaration on
    # the host, and must not depend on which backend registered first.
    sleeps = []
    clock = {"now": 0.0}
    registry = LimiterRegistry(now=lambda: clock["now"], sleep=sleeps.append)
    fast = limiter_backend("fast", "https://integrate.api.nvidia.com/v1", 60)
    slow = limiter_backend("slow", "https://integrate.api.nvidia.com/v1", 20)
    limiter = registry.for_backend(fast)
    registry.for_backend(slow)  # tightens the already-created limiter to 20 rpm
    limiter.wait()
    limiter.wait()
    assert sleeps == [3.0]  # 60/20, not 60/60


def test_limiter_registry_keeps_the_lowest_rpm_regardless_of_registration_order():
    sleeps = []
    clock = {"now": 0.0}
    registry = LimiterRegistry(now=lambda: clock["now"], sleep=sleeps.append)
    slow = limiter_backend("slow", "https://integrate.api.nvidia.com/v1", 20)
    fast = limiter_backend("fast", "https://integrate.api.nvidia.com/v1", 60)
    registry.for_backend(slow)
    limiter = registry.for_backend(fast)  # must NOT loosen back to 60
    limiter.wait()
    limiter.wait()
    assert sleeps == [3.0]


def http_429(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://x.test"))
    return httpx.HTTPStatusError("429", request=response.request, response=response)


def test_rate_limit_penalty_ignores_errors_that_are_not_429():
    other = httpx.HTTPStatusError(
        "500",
        request=httpx.Request("POST", "https://x.test"),
        response=httpx.Response(500, request=httpx.Request("POST", "https://x.test")),
    )
    assert rate_limit_penalty(other) is None
    assert rate_limit_penalty(RuntimeError("boom")) is None


def test_rate_limit_penalty_defaults_when_the_host_sends_no_header():
    # DeepInfra — the successor host for deepseek once NVIDIA's free tier ends
    # — sends no x-ratelimit-* and no Retry-After at all (verified live
    # 2026-08-07). Headroom is not discoverable in-band, so a 429 there carries
    # no number and the default has to apply.
    assert rate_limit_penalty(http_429()) == RETRY_AFTER_DEFAULT_S


def test_rate_limit_penalty_honors_retry_after_when_the_host_sends_one():
    assert rate_limit_penalty(http_429("12")) == 12.0


def test_rate_limit_penalty_falls_back_on_an_unparseable_retry_after():
    # Retry-After may legally be an HTTP-date. Rather than parse dates, fall
    # back — a wrong small number is worse than a conservative default.
    assert rate_limit_penalty(http_429("Wed, 21 Oct 2026 07:28:00 GMT")) == RETRY_AFTER_DEFAULT_S


def test_rate_limit_penalty_rejects_a_non_finite_retry_after():
    # `float("inf")` parses. Feeding it to penalize() poisons the host deadline:
    # time.sleep(inf) raises OverflowError on this platform, so the leg aborts
    # and every other backend on the host aborts behind it. `nan` is worse and
    # fails OPEN — nan > 0 is False, so no sleep happens, _last becomes nan, and
    # every later comparison is nan, silently disabling backoff for the run.
    assert rate_limit_penalty(http_429("inf")) == RETRY_AFTER_DEFAULT_S
    assert rate_limit_penalty(http_429("nan")) == RETRY_AFTER_DEFAULT_S


def test_rate_limit_penalty_rejects_a_negative_retry_after():
    assert rate_limit_penalty(http_429("-5")) == RETRY_AFTER_DEFAULT_S


def test_rate_limit_penalty_is_bounded():
    # Even a well-formed enormous value would park the host under the limiter
    # lock with nothing observable. A bake-off leg should fail and be resumed,
    # not silently sit still for eleven days.
    assert rate_limit_penalty(http_429("999999")) == MAX_BACKOFF_S


def test_limiter_host_does_not_collapse_schemeless_urls_into_one_bucket():
    # urlparse puts everything in `path` when there is no scheme, so netloc is
    # "" and EVERY schemeless backend keys to the same empty-string bucket —
    # silently sharing one rate limit across unrelated providers. Latent today
    # because every configured base_url carries a scheme, and invisible if it
    # ever stops being true.
    nvidia = limiter_host("integrate.api.nvidia.com/v1")
    openai = limiter_host("api.openai.com/v1")
    assert nvidia == "integrate.api.nvidia.com"
    assert openai == "api.openai.com"
    assert nvidia != openai


def test_limiter_host_agrees_with_itself_across_scheme_presence():
    assert limiter_host("https://api.x.ai/v1") == limiter_host("api.x.ai/v1")


def test_penalize_delays_every_caller_on_the_host_not_just_the_one_that_429ed():
    # The concurrency-specific bug: back off only the worker that got the 429
    # and the other seven keep hammering the same host. The bucket is the
    # enforcement point, so the penalty has to live there.
    sleeps = []
    clock = {"now": 0.0}
    limiter = RateLimiter(rpm=6000, now=lambda: clock["now"], sleep=sleeps.append)
    limiter.wait()  # a call departs at t=0
    limiter.penalize(30.0)  # ...and comes back 429
    limiter.wait()  # the NEXT caller — a different worker — must wait it out
    assert sleeps == [30.01]  # 30s penalty + the 0.01s rate interval


def test_penalize_does_not_shorten_an_existing_longer_backoff():
    # Eight workers in flight can each catch a 429 for the same overload. The
    # penalties must not stack into a multi-minute stall, nor let a late small
    # one undercut a longer wait already in force.
    sleeps = []
    clock = {"now": 0.0}
    limiter = RateLimiter(rpm=6000, now=lambda: clock["now"], sleep=sleeps.append)
    limiter.wait()
    limiter.penalize(30.0)
    limiter.penalize(5.0)  # a second worker's 429, shorter
    limiter.wait()
    assert sleeps == [30.01]


def test_penalize_is_shared_by_every_backend_on_the_host():
    # Two NVIDIA backends, one endpoint: a 429 earned by one must slow both.
    sleeps = []
    clock = {"now": 0.0}
    registry = LimiterRegistry(now=lambda: clock["now"], sleep=sleeps.append)
    deepseek = limiter_backend("nvidia-deepseek", "https://integrate.api.nvidia.com/v1", 6000)
    nemotron = limiter_backend("nvidia-nemotron", "https://integrate.api.nvidia.com/v1", 6000)
    registry.for_backend(deepseek).wait()
    registry.for_backend(deepseek).penalize(30.0)
    registry.for_backend(nemotron).wait()
    assert sleeps == [30.01]


def test_judge_client_parses_chat_completion(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="meta/llama-3.1-8b-instruct",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
        temperature=0.0,  # this endpoint accepts it; policy is covered separately
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"verdict": "reject", "reason": "casing_error"}'}}
                ]
            },
        )

    client = JudgeClient(backend, transport=httpx.MockTransport(handler))
    verdict = client.judge(PAIR)
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert json.loads(captured["body"])["temperature"] == 0.0
    assert verdict.pair_id == "p1"
    assert verdict.verdict == "reject"
    assert verdict.reason == ReasonCode.CASING_ERROR
    assert verdict.model_id == "meta/llama-3.1-8b-instruct"
    assert verdict.prompt_version == "v1"
    assert verdict.temperature == 0.0


def test_judge_records_the_usage_the_host_reported(monkeypatch):
    # Issue #11. The data is already in the response object — this is a
    # capture, not an extra call — and without it cost has to be estimated.
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"verdict": "approve", "reason": "ok"}'}}],
                "usage": {
                    "prompt_tokens": 412,
                    "completion_tokens": 138,
                    "total_tokens": 550,
                    "prompt_tokens_details": {"cached_tokens": 384},
                    "completion_tokens_details": {"reasoning_tokens": 120},
                },
            },
        )

    verdict = JudgeClient(backend, transport=httpx.MockTransport(handler)).judge(PAIR)
    assert verdict.usage.prompt_tokens == 412
    assert verdict.usage.total_tokens == 550
    assert verdict.usage.cached_tokens == 384
    assert verdict.usage.reasoning_tokens == 120


def test_judge_leaves_usage_none_when_the_host_reports_none(monkeypatch):
    # A host that sends no usage block must not produce a row of zeros — that
    # would read as a measured free call and understate the bill.
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"verdict": "approve", "reason": "ok"}'}}]},
        )

    verdict = JudgeClient(backend, transport=httpx.MockTransport(handler)).judge(PAIR)
    assert verdict.usage is None


def test_judge_records_usage_from_the_anthropic_dialect(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = Backend(
        name="anthropic-haiku",
        base_url="https://example.test/v1",
        model_id="claude-haiku-4-5",
        rpm=50,
        eval_only=False,
        api_key_env="ANTHROPIC_API_KEY",
        api="anthropic",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"verdict": "approve", "reason": "ok"}'}],
                "usage": {"input_tokens": 300, "output_tokens": 20},
            },
        )

    verdict = JudgeClient(backend, transport=httpx.MockTransport(handler)).judge(PAIR)
    assert verdict.usage.prompt_tokens == 300
    assert verdict.usage.completion_tokens == 20
    assert verdict.usage.total_tokens == 320  # derived; Anthropic sends no total


def test_judge_client_speaks_anthropic_messages_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    backend = Backend(
        name="anthropic-haiku",
        base_url="https://example.test/v1",
        model_id="claude-haiku-4-5",
        rpm=50,
        eval_only=False,
        api_key_env="ANTHROPIC_API_KEY",
        api="anthropic",
        temperature=0.0,  # this endpoint accepts it; policy is covered separately
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"verdict": "approve", "reason": "ok"}'}]},
        )

    client = JudgeClient(backend, transport=httpx.MockTransport(handler))
    verdict = client.judge(PAIR)

    assert captured["url"] == "https://example.test/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert "authorization" not in captured["headers"]
    body = captured["body"]
    assert body["temperature"] == 0.0
    # Anthropic takes the system prompt as a top-level field, not a message.
    assert body["system"].startswith("You are a strict")
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert verdict.verdict == "approve"
    assert verdict.reason == ReasonCode.OK
    assert verdict.model_id == "claude-haiku-4-5"


def openai_backend(**overrides):
    base = dict(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    base.update(overrides)
    return Backend(**base)


def capture_body(backend, response_json=None):
    """Run one judge call against a mock transport; return the request body."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json=response_json
            or {
                "model": "m-2026-01-01",
                "choices": [
                    {"message": {"content": '{"verdict": "approve", "reason": "ok"}'}}
                ],
            },
        )

    client = JudgeClient(backend, transport=httpx.MockTransport(handler))
    verdict = client.judge(PAIR)
    return captured, verdict, client


def test_judge_stamps_the_host_that_served_the_call(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    _, verdict, _ = capture_body(openai_backend(base_url="https://host-a.test/v1"))
    assert verdict.base_url == "https://host-a.test/v1"


def test_judge_stamps_the_config_digest(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    backend = openai_backend()
    _, verdict, _ = capture_body(backend)
    assert verdict.config_digest == config_digest(backend, prompt_version=PROMPT_VERSION)


def test_judge_stamps_the_code_version_it_was_given(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"verdict": "approve", "reason": "ok"}'}}]},
        )

    client = JudgeClient(
        openai_backend(),
        transport=httpx.MockTransport(handler),
        code_version="9b0d01a1c2d3",
    )
    assert client.judge(PAIR).code_version == "9b0d01a1c2d3"


def test_openai_request_omits_temperature_by_default(monkeypatch):
    # R2: gpt-5.6 rejects temperature=0 at any reasoning effort above `none`,
    # and sending 1.0 to claim "the default" is a lie the results would carry.
    # Omit the key entirely unless a backend has proven it is accepted.
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    body, verdict, _ = capture_body(openai_backend())
    assert "temperature" not in body
    assert verdict.temperature is None


def test_openai_request_sends_temperature_when_backend_declares_it(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    body, verdict, _ = capture_body(openai_backend(temperature=0.0))
    assert body["temperature"] == 0.0
    assert verdict.temperature == 0.0


def test_anthropic_request_omits_temperature_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    backend = openai_backend(api="anthropic", api_key_env="ANTHROPIC_API_KEY")
    body, _, _ = capture_body(
        backend,
        response_json={
            "model": "m-2026-01-01",
            "content": [{"type": "text", "text": '{"verdict": "approve", "reason": "ok"}'}],
        },
    )
    assert "temperature" not in body
    assert body["max_tokens"] == 256  # Anthropic still requires this one


def test_openai_request_sends_reasoning_effort_when_declared(monkeypatch):
    # R7: effort changes the answer (effort=none picked a different reason code
    # than effort=medium on a borderline pair), so it is pinned and recorded,
    # never left to the provider default.
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    body, verdict, _ = capture_body(openai_backend(reasoning_effort="medium"))
    assert body["reasoning_effort"] == "medium"
    assert verdict.reasoning_effort == "medium"


def test_openai_request_omits_reasoning_effort_when_not_declared(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    body, verdict, _ = capture_body(openai_backend())
    assert "reasoning_effort" not in body
    assert verdict.reasoning_effort is None


def test_backend_rejects_unknown_reasoning_effort():
    with pytest.raises(ValueError, match="reasoning_effort"):
        openai_backend(reasoning_effort="telepathic")


def test_structured_output_sends_strict_json_schema_enum(monkeypatch):
    # R6: strict mode masks logits to schema-legal tokens, removing format-level
    # variance and parse failures that would otherwise read as disagreement.
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    body, _, _ = capture_body(openai_backend(structured_output=True))
    schema = body["response_format"]["json_schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert schema["strict"] is True
    props = schema["schema"]["properties"]
    assert props["verdict"]["enum"] == ["approve", "reject"]
    assert set(props["reason"]["enum"]) == {
        "overcorrection",
        "meaning_change",
        "casing_error",
        "truncation_worse",
        "ok",
    }
    assert schema["schema"]["additionalProperties"] is False


def test_structured_output_is_off_by_default(monkeypatch):
    # Not every OpenAI-compatible endpoint implements strict mode; opting in
    # per backend keeps an unsupported endpoint from 400ing the whole run.
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    body, _, _ = capture_body(openai_backend())
    assert "response_format" not in body


ANTHROPIC_REPLY = {
    "model": "claude-haiku-4-5-20251001",
    "content": [{"type": "text", "text": '{"verdict": "approve", "reason": "ok"}'}],
}


def anthropic_backend(**overrides):
    base = dict(api="anthropic", api_key_env="ANTHROPIC_API_KEY", model_id="claude-haiku-4-5")
    base.update(overrides)
    return openai_backend(**base)


def test_anthropic_request_sends_structured_output_in_anthropic_shape(monkeypatch):
    # The Anthropic Messages API DOES support strict JSON-schema output, but the
    # shape differs from OpenAI's: output_config.format, and the schema is bare
    # (no name/strict wrapper). Dropping the field silently — while the verdict
    # and manifest still recorded structured_output=true — made the metadata lie.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body, _, _ = capture_body(
        anthropic_backend(structured_output=True), response_json=ANTHROPIC_REPLY
    )
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["verdict"]["enum"] == ["approve", "reject"]
    assert fmt["schema"]["additionalProperties"] is False
    assert "response_format" not in body  # that is the OpenAI spelling
    assert "json_schema" not in fmt  # OpenAI's name/strict wrapper does not apply


def test_anthropic_request_maps_effort_onto_output_config(monkeypatch):
    # Anthropic's equivalent of reasoning_effort is output_config.effort.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body, verdict, _ = capture_body(
        anthropic_backend(reasoning_effort="high"), response_json=ANTHROPIC_REPLY
    )
    assert body["output_config"]["effort"] == "high"
    assert "reasoning_effort" not in body  # that is the OpenAI spelling
    assert verdict.reasoning_effort == "high"


def test_anthropic_effort_none_maps_to_disabled_thinking(monkeypatch):
    # Anthropic has no effort level "none" — the documented equivalent of
    # "don't reason" is thinking: {type: disabled}. Passing "none" through as an
    # effort value would 400.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body, _, _ = capture_body(
        anthropic_backend(reasoning_effort="none"), response_json=ANTHROPIC_REPLY
    )
    assert body["thinking"] == {"type": "disabled"}
    assert "effort" not in body.get("output_config", {})


def test_anthropic_request_combines_effort_and_structured_output(monkeypatch):
    # Both live under output_config — setting one must not clobber the other.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body, _, _ = capture_body(
        anthropic_backend(reasoning_effort="medium", structured_output=True),
        response_json=ANTHROPIC_REPLY,
    )
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_request_omits_output_config_when_neither_is_declared(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body, _, _ = capture_body(anthropic_backend(), response_json=ANTHROPIC_REPLY)
    assert "output_config" not in body
    assert "thinking" not in body


def test_judge_records_run_index(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "m-2026-01-01",
                "choices": [{"message": {"content": '{"verdict": "approve", "reason": "ok"}'}}],
            },
        )

    client = JudgeClient(openai_backend(), transport=httpx.MockTransport(handler))
    assert client.judge(PAIR, run_index=2).run_index == 2
    assert client.judge(PAIR).run_index == 0


def test_client_records_observed_model_snapshots(monkeypatch):
    # R7: `system_fingerprint` is unusable on Responses, so the resolved model
    # string is our only drift signal. Collecting the set (not just the first)
    # catches a snapshot that changes mid-run.
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    snapshots = iter(["m-2026-01-01", "m-2026-06-30"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": next(snapshots),
                "choices": [{"message": {"content": '{"verdict": "approve", "reason": "ok"}'}}],
            },
        )

    client = JudgeClient(openai_backend(), transport=httpx.MockTransport(handler))
    client.judge(PAIR)
    client.judge(PAIR)
    assert client.observed_models == {"m-2026-01-01", "m-2026-06-30"}


def test_load_backends_reads_temperature_effort_and_structured_output(tmp_path):
    path = tmp_path / "backends.toml"
    path.write_text(
        BACKENDS_TOML
        + '\ntemperature = 0.0\nreasoning_effort = "high"\nstructured_output = true\n'
    )
    backend = load_backends(path)[-1]
    assert backend.temperature == 0.0
    assert backend.reasoning_effort == "high"
    assert backend.structured_output is True


def test_load_backends_defaults_temperature_to_omitted(tmp_path):
    path = tmp_path / "backends.toml"
    path.write_text(BACKENDS_TOML)
    assert load_backends(path)[0].temperature is None


def test_every_backend_field_is_classified():
    # The guard against issue #13 failure 3: a new Backend field that shapes the
    # request must not slip outside the resume identity by default. This fails
    # the moment someone adds a field without deciding which set it belongs to.
    classified = VERDICT_AFFECTING_FIELDS | OPERATIONAL_FIELDS
    actual = {f.name for f in fields(Backend)}
    assert classified == actual


def test_verdict_affecting_and_operational_fields_are_disjoint():
    assert not (VERDICT_AFFECTING_FIELDS & OPERATIONAL_FIELDS)


def test_request_shaping_fields_are_verdict_affecting():
    # api selects the wire protocol and the response-extraction path;
    # structured_output toggles strict json_schema. Both change the judgment.
    assert {"base_url", "model_id", "api", "temperature", "reasoning_effort", "structured_output"} <= (
        VERDICT_AFFECTING_FIELDS
    )


def test_scheduling_fields_are_operational():
    # These change how calls are scheduled or labelled, never what is asked.
    assert {"name", "rpm", "eval_only", "api_key_env", "role", "timeout_s"} <= OPERATIONAL_FIELDS


def test_config_digest_is_stable_for_the_same_config():
    a = openai_backend()
    b = openai_backend()
    assert config_digest(a, prompt_version="v1") == config_digest(b, prompt_version="v1")


def test_config_digest_is_twelve_hex_chars():
    digest = config_digest(openai_backend(), prompt_version="v1")
    assert len(digest) == 12
    assert all(c in "0123456789abcdef" for c in digest)


def test_config_digest_changes_when_api_dialect_changes():
    # The hole rev 1 left open: same model, same host, same sha, different wire
    # protocol and response-extraction path.
    openai = config_digest(openai_backend(api="openai"), prompt_version="v1")
    anthropic = config_digest(openai_backend(api="anthropic"), prompt_version="v1")
    assert openai != anthropic


def test_config_digest_changes_when_structured_output_changes():
    off = config_digest(openai_backend(structured_output=False), prompt_version="v1")
    on = config_digest(openai_backend(structured_output=True), prompt_version="v1")
    assert off != on


def test_config_digest_changes_when_prompt_version_changes():
    backend = openai_backend()
    assert config_digest(backend, prompt_version="v1") != config_digest(backend, prompt_version="v2")


def test_config_digest_ignores_operational_fields():
    # Re-tuning rpm or renaming a backend must NOT invalidate a results file:
    # neither changes what the model was asked.
    base = config_digest(openai_backend(), prompt_version="v1")
    retuned = config_digest(openai_backend(rpm=999, name="renamed", timeout_s=1.0), prompt_version="v1")
    assert base == retuned


def test_judge_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        JudgeClient(backend)
