"""OpenAI-compatible chat client for judge backends.

Backends are declared in backends.toml; API keys come from the environment
only (never from config or code).
"""

from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from judge.prompts import PROMPT_VERSION, build_messages, parse_judge_response
from judge.schema import VALID_VERDICTS, Pair, ReasonCode, Verdict

MAX_TOKENS = 256  # a verdict object is tiny; Anthropic requires the field
ANTHROPIC_VERSION = "2023-06-01"

# Sampling temperature is OPT-IN per backend, never a harness-wide constant.
#
# The GPT-5.x family gates sampling params on reasoning effort: probing
# gpt-5.6-luna on 2026-08-06 returned 400 for temperature=0 at effort=medium on
# both chat/completions and responses, and 200 at effort="none". Anthropic has
# deprecated the knob outright on recent Opus models. Sending a value "just to
# be explicit" either 400s the run or silently pins a model to a temperature it
# would not otherwise use, so a backend that has not proven it accepts a value
# gets no temperature key at all — and its verdicts record None, not 0.0.
TEMPERATURE_OMITTED = None

# Effort is not a free knob either: on a borderline pair, effort="none" spent 0
# reasoning tokens and returned reason=truncation_worse where effort="medium"
# spent ~120 and returned casing_error. Since score.py scores per-reason
# confusion, effort changes the metric — so it is pinned per backend and
# recorded on every verdict rather than left to the provider default.
VALID_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

# Large reasoning models are genuinely slow: thinkingmachines/inkling answered
# in 79s. A timeout below that reports a slow model as unreachable, which is a
# wrong diagnosis rather than a slow one. Override per backend via timeout_s.
REQUEST_TIMEOUT_S = 180.0


VALID_APIS = ("openai", "anthropic")
VALID_ROLES = ("contender", "floor")


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    model_id: str
    rpm: int
    eval_only: bool
    api_key_env: str
    api: str = "openai"  # wire protocol: OpenAI-compatible chat, or Anthropic messages
    role: str = "contender"  # "floor" backends are baselines, not contenders
    timeout_s: float = REQUEST_TIMEOUT_S
    # None = omit the key entirely. Set only where a value is PROVEN accepted.
    temperature: float | None = TEMPERATURE_OMITTED
    reasoning_effort: str | None = None  # None = omit; provider default applies
    structured_output: bool = False  # strict json_schema; not every endpoint implements it

    def __post_init__(self) -> None:
        if self.api not in VALID_APIS:
            raise ValueError(f"backend {self.name!r}: api must be one of {VALID_APIS}, got {self.api!r}")
        if self.role not in VALID_ROLES:
            raise ValueError(f"backend {self.name!r}: role must be one of {VALID_ROLES}, got {self.role!r}")
        if self.reasoning_effort is not None and self.reasoning_effort not in VALID_EFFORTS:
            raise ValueError(
                f"backend {self.name!r}: reasoning_effort must be one of {VALID_EFFORTS}, "
                f"got {self.reasoning_effort!r}"
            )


def load_backends(path: str | Path) -> list[Backend]:
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    backends = []
    for entry in config["backends"]:
        default_env = entry["name"].upper().replace("-", "_") + "_API_KEY"
        backends.append(
            Backend(
                name=entry["name"],
                base_url=entry["base_url"].rstrip("/"),
                model_id=entry["model_id"],
                rpm=entry["rpm"],
                eval_only=entry["eval_only"],
                api_key_env=entry.get("api_key_env", default_env),
                api=entry.get("api", "openai"),
                role=entry.get("role", "contender"),
                timeout_s=float(entry.get("timeout_s", REQUEST_TIMEOUT_S)),
                temperature=entry.get("temperature", TEMPERATURE_OMITTED),
                reasoning_effort=entry.get("reasoning_effort"),
                structured_output=entry.get("structured_output", False),
            )
        )
    return backends


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart; clock injectable for tests."""

    def __init__(
        self,
        rpm: int,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / rpm
        self._now = now
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._interval - (self._now() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._now()


VERDICT_JSON_SCHEMA = {
    "name": "judge_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VALID_VERDICTS)},
            "reason": {"type": "string", "enum": [rc.value for rc in ReasonCode]},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    },
}


def _openai_request(backend: Backend, pair: Pair) -> tuple[str, dict]:
    """Path and body for an OpenAI-compatible chat completion.

    `temperature` and `reasoning_effort` appear only when the backend declares
    them — see TEMPERATURE_OMITTED. An absent key is not the same as a default
    value here: it is the difference between a 200 and a 400 on GPT-5.x.
    """
    body = {
        "model": backend.model_id,
        "messages": build_messages(pair),
    }
    if backend.temperature is not None:
        body["temperature"] = backend.temperature
    if backend.reasoning_effort is not None:
        body["reasoning_effort"] = backend.reasoning_effort
    if backend.structured_output:
        body["response_format"] = {"type": "json_schema", "json_schema": VERDICT_JSON_SCHEMA}
    return "/chat/completions", body


def _openai_content(payload: dict) -> str:
    return payload["choices"][0]["message"]["content"]


def _anthropic_request(backend: Backend, pair: Pair) -> tuple[str, dict]:
    """Path and body for the Anthropic messages API.

    Anthropic takes the system prompt as a top-level field rather than a
    message, so the shared prompt is split rather than rewritten.
    """
    system, user = build_messages(pair)
    body = {
        "model": backend.model_id,
        "system": system["content"],
        "messages": [user],
        "max_tokens": MAX_TOKENS,
    }
    if backend.temperature is not None:
        body["temperature"] = backend.temperature
    return "/messages", body


def _anthropic_content(payload: dict) -> str:
    return "".join(block["text"] for block in payload["content"] if block["type"] == "text")


def _auth_headers(api: str, api_key: str) -> dict[str, str]:
    if api == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    return {"Authorization": f"Bearer {api_key}"}


class JudgeClient:
    def __init__(self, backend: Backend, transport: httpx.BaseTransport | None = None) -> None:
        api_key = os.environ.get(backend.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"backend {backend.name!r} needs API key in ${backend.api_key_env} (env only, never commit keys)"
            )
        self.backend = backend
        # Every distinct snapshot string the provider resolved us to. Responses
        # has no system_fingerprint, so this set is the only drift signal we
        # get — and a set, not a single value, catches a mid-run swap.
        self.observed_models: set[str] = set()
        self._http = httpx.Client(
            base_url=backend.base_url,
            headers=_auth_headers(backend.api, api_key),
            timeout=backend.timeout_s,
            transport=transport,
        )

    def request_body(self, pair: Pair) -> dict:
        """The exact body that judging `pair` would POST — for the run manifest."""
        build = _anthropic_request if self.backend.api == "anthropic" else _openai_request
        return build(self.backend, pair)[1]

    def judge(self, pair: Pair, run_index: int = 0) -> Verdict:
        is_anthropic = self.backend.api == "anthropic"
        build = _anthropic_request if is_anthropic else _openai_request
        extract = _anthropic_content if is_anthropic else _openai_content

        path, body = build(self.backend, pair)
        response = self._http.post(path, json=body)
        response.raise_for_status()
        payload = response.json()
        if resolved := payload.get("model"):
            self.observed_models.add(resolved)
        verdict, reason = parse_judge_response(extract(payload))
        return Verdict(
            pair_id=pair.id,
            verdict=verdict,
            reason=reason,
            model_id=self.backend.model_id,
            prompt_version=PROMPT_VERSION,
            temperature=self.backend.temperature,
            run_index=run_index,
            reasoning_effort=self.backend.reasoning_effort,
        )

    def close(self) -> None:
        self._http.close()
