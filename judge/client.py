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
from judge.schema import Pair, Verdict

TEMPERATURE = 0.0
MAX_TOKENS = 256  # a verdict object is tiny; Anthropic requires the field
ANTHROPIC_VERSION = "2023-06-01"


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

    def __post_init__(self) -> None:
        if self.api not in VALID_APIS:
            raise ValueError(f"backend {self.name!r}: api must be one of {VALID_APIS}, got {self.api!r}")
        if self.role not in VALID_ROLES:
            raise ValueError(f"backend {self.name!r}: role must be one of {VALID_ROLES}, got {self.role!r}")


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


def _openai_request(backend: Backend, pair: Pair) -> tuple[str, dict]:
    """Path and body for an OpenAI-compatible chat completion."""
    return "/chat/completions", {
        "model": backend.model_id,
        "messages": build_messages(pair),
        "temperature": TEMPERATURE,
    }


def _openai_content(payload: dict) -> str:
    return payload["choices"][0]["message"]["content"]


def _anthropic_request(backend: Backend, pair: Pair) -> tuple[str, dict]:
    """Path and body for the Anthropic messages API.

    Anthropic takes the system prompt as a top-level field rather than a
    message, so the shared prompt is split rather than rewritten.
    """
    system, user = build_messages(pair)
    return "/messages", {
        "model": backend.model_id,
        "system": system["content"],
        "messages": [user],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }


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
        self._http = httpx.Client(
            base_url=backend.base_url,
            headers=_auth_headers(backend.api, api_key),
            timeout=60.0,
            transport=transport,
        )

    def judge(self, pair: Pair) -> Verdict:
        is_anthropic = self.backend.api == "anthropic"
        build = _anthropic_request if is_anthropic else _openai_request
        extract = _anthropic_content if is_anthropic else _openai_content

        path, body = build(self.backend, pair)
        response = self._http.post(path, json=body)
        response.raise_for_status()
        verdict, reason = parse_judge_response(extract(response.json()))
        return Verdict(
            pair_id=pair.id,
            verdict=verdict,
            reason=reason,
            model_id=self.backend.model_id,
            prompt_version=PROMPT_VERSION,
            temperature=TEMPERATURE,
        )

    def close(self) -> None:
        self._http.close()
