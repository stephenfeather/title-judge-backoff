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
