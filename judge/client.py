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


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    model_id: str
    rpm: int
    eval_only: bool
    api_key_env: str


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
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
            transport=transport,
        )

    def judge(self, pair: Pair) -> Verdict:
        response = self._http.post(
            "/chat/completions",
            json={
                "model": self.backend.model_id,
                "messages": build_messages(pair),
                "temperature": TEMPERATURE,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        verdict, reason = parse_judge_response(content)
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
