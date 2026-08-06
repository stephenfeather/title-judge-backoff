import json

import httpx
import pytest

from judge.client import Backend, JudgeClient, RateLimiter, load_backends
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
