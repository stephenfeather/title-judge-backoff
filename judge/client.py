"""OpenAI-compatible chat client for judge backends.

Backends are declared in backends.toml; API keys come from the environment
only (never from config or code).
"""

from __future__ import annotations

import math
import os
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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

# Backends can take a long time to come back even when they never answer:
# thinkingmachines/inkling took 79s to return an HTTP 500 (it did NOT answer in
# 79s). A timeout below that turns "slow" and "slow then failed" into the same
# unreachable verdict, hiding which one we actually hit — and the response body
# is where the difference shows up. Override per backend via timeout_s.
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
    """Spaces calls at least 60/rpm seconds apart; clock injectable for tests.

    Safe to share between threads. The sleep happens while the lock is HELD,
    which is the point: the decision "when may the next call leave" is
    inherently serial, so admitting one caller per interval is exactly the
    intended behaviour. Only admission is serialized — the HTTP call itself
    runs outside the lock, so N workers stay in flight concurrently while calls
    still depart at the declared rate.

    Releasing the lock before sleeping would defeat it: every waiting thread
    would read the same `_last`, sleep the same interval, and fire together.
    """

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
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            if self._last is not None:
                remaining = self._interval - (self._now() - self._last)
                if remaining > 0:
                    self._sleep(remaining)
            self._last = self._now()

    def tighten_to(self, rpm: int) -> None:
        """Lower the rate to `rpm` if that is stricter than the current one.

        Only ever tightens, so the result does not depend on the order backends
        register — see LimiterRegistry.
        """
        with self._lock:
            self._interval = max(self._interval, 60.0 / rpm)

    def penalize(self, seconds: float) -> None:
        """Hold back EVERY caller on this host for `seconds` after a 429.

        Backing off only the worker that caught the 429 is the concurrency
        version of the bug this whole registry exists to prevent: the other N-1
        workers keep hammering the same overloaded host while one of them
        sleeps. The bucket is the enforcement point, so the penalty lives here.

        Never shortens a longer penalty already in force. Eight workers in
        flight can each catch a 429 for the same overload; taking the max keeps
        that from either stacking into a multi-minute stall or letting a late
        small penalty undercut a longer wait.
        """
        with self._lock:
            # Measured from NOW, not from any deadline already set: a 429 means
            # "do not send for `seconds` from here". Taking the max of the two
            # deadlines is what keeps concurrent penalties from stacking.
            deadline = self._now() + seconds
            self._last = deadline if self._last is None else max(self._last, deadline)


# How long to hold a host back after a 429 that carries no Retry-After.
#
# Not every provider tells you. DeepInfra — the successor host for deepseek
# once NVIDIA's free tier ends after 2026-08-07 — sends no `x-ratelimit-*` and
# no `Retry-After` at all (verified live 2026-08-07), so on that host headroom
# is not discoverable in band and a declared rpm is a request rather than a
# guarantee. Nothing here infers headroom from response headers; a 429 is the
# only signal, and it means the bucket was too generous.
RETRY_AFTER_DEFAULT_S = 30.0

# Upper bound on any single backoff, however large a value the host asks for.
# A leg that stalls longer than this should fail and be resumed rather than sit
# under the limiter lock with nothing observable — an unbounded Retry-After
# would park every backend on the host for as long as the header says.
MAX_BACKOFF_S = 300.0


def rate_limit_penalty(exc: BaseException) -> float | None:
    """Seconds to hold the host back for, or None if `exc` is not a 429.

    `Retry-After` is honored only when it is a finite, non-negative number of
    seconds, and is clamped to MAX_BACKOFF_S. Everything else falls back to the
    default. The rejected cases are not hypothetical pedantry:

      * "inf" parses through float(). Feeding it to penalize() sets the host
        deadline to infinity, and time.sleep(inf) then raises OverflowError —
        aborting the leg, and every other backend on that host behind it.
      * "nan" fails OPEN, which is worse. `nan > 0` is False so no sleep
        happens, `_last` becomes nan, and every later comparison is nan too:
        backoff is silently disabled on that host for the rest of the run.
      * An HTTP-date is legal in this header. Rather than parse dates, fall
        back — guessing a small number is worse than waiting a safe one.
    """
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code != 429:
        return None
    raw = exc.response.headers.get("retry-after")
    if raw is None:
        return RETRY_AFTER_DEFAULT_S
    try:
        seconds = float(raw)
    except ValueError:
        return RETRY_AFTER_DEFAULT_S
    if not math.isfinite(seconds) or seconds < 0:
        return RETRY_AFTER_DEFAULT_S
    return min(seconds, MAX_BACKOFF_S)


def limiter_host(base_url: str) -> str:
    """The rate-limit key for a base_url: its host, ignoring path and scheme.

    Falls back to the first path segment when there is no scheme. urlparse puts
    the whole string in `path` in that case, leaving netloc empty — and an empty
    key would collapse EVERY schemeless backend into one shared limiter,
    throttling unrelated providers against each other with no error to notice.
    """
    parsed = urlparse(base_url)
    if parsed.netloc:
        return parsed.netloc
    return parsed.path.split("/", 1)[0]


class LimiterRegistry:
    """One RateLimiter per HOST, shared by every backend on that host.

    Quotas are enforced by providers per endpoint, not per model. Six nvidia-*
    backends share integrate.api.nvidia.com; limiting each of them to its own
    40 rpm asks that one host for 240. Measured on 2026-08-06: running nemotron
    alongside deepseek drove deepseek to p50 9.77s with 81 failures where it had
    been p50 2.48s with none. Same backend, same host, same hour.

    Because the limiter is shared, the host runs at the LOWEST rpm any of its
    backends declared — a conservative declaration is never loosened by a
    later, more permissive one.
    """

    def __init__(
        self,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._now = now
        self._sleep = sleep
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def for_backend(self, backend: Backend) -> RateLimiter:
        host = limiter_host(backend.base_url)
        with self._lock:
            limiter = self._limiters.get(host)
            if limiter is None:
                limiter = RateLimiter(rpm=backend.rpm, now=self._now, sleep=self._sleep)
                self._limiters[host] = limiter
            else:
                limiter.tighten_to(backend.rpm)
            return limiter


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


# Anthropic supports the same two capabilities as the OpenAI path, under
# different names — so `reasoning_effort` and `structured_output` are honored
# here rather than silently dropped while the verdict and manifest still claim
# them. The spellings differ in three ways:
#   * effort lives at output_config.effort, not top-level reasoning_effort;
#   * the JSON schema lives at output_config.format and is bare — no
#     {name, strict, schema} wrapper (that shape is OpenAI's);
#   * there is no effort level "none"; the documented equivalent of "do not
#     reason" is thinking: {"type": "disabled"} (see _anthropic_request).
# Note claude-haiku-4-5 — the only Anthropic backend on the current slate —
# does NOT accept the effort parameter at all. Declaring effort on it will 400,
# which is the correct loud failure rather than a silently ignored setting.
ANTHROPIC_VERDICT_SCHEMA = VERDICT_JSON_SCHEMA["schema"]
ANTHROPIC_EFFORT_NONE = {"type": "disabled"}


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

    output_config: dict = {}
    if backend.reasoning_effort == "none":
        body["thinking"] = ANTHROPIC_EFFORT_NONE
    elif backend.reasoning_effort is not None:
        output_config["effort"] = backend.reasoning_effort
    if backend.structured_output:
        output_config["format"] = {"type": "json_schema", "schema": ANTHROPIC_VERDICT_SCHEMA}
    if output_config:
        body["output_config"] = output_config
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
