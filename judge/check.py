"""Dry-run validation of a backend slate.

Key presence is checked from the environment only; no network call is made
unless the caller explicitly asks for a reachability ping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from judge.client import Backend, JudgeClient
from judge.schema import Pair, ReasonCode

PING_PAIR = Pair(
    id="ping",
    original="acme widget 3000 blk",
    enriched="Acme Widget 3000, Black",
    brand="Acme",
    mpn="W3000-BLK",
    ground_truth="approve",
    reason=ReasonCode.OK,
)


@dataclass(frozen=True)
class BackendCheck:
    name: str
    status: str  # "ready" | "skipped"
    detail: str
    reachable: bool | None  # None when no ping was attempted


def check_key_presence(backend: Backend) -> BackendCheck:
    """Ready if the backend's API key env var holds a non-blank value."""
    value = os.environ.get(backend.api_key_env, "")
    if value.strip():
        return BackendCheck(
            name=backend.name, status="ready", detail=f"{backend.api_key_env} is set", reachable=None
        )
    return BackendCheck(
        name=backend.name,
        status="skipped",
        detail=f"{backend.api_key_env} is not set",
        reachable=None,
    )


def _failure_detail(exc: Exception) -> str:
    """Describe a failed ping, including the provider's own explanation.

    httpx's HTTPStatusError stringifies to just the status line, which is how a
    400 saying "this model does not accept temperature=0" spent a while looking
    like an invalid API key. The response body is the entire point of a ping, so
    it is appended whenever there is one.
    """
    detail = f"ping failed: {exc}"
    response = getattr(exc, "response", None)
    if response is not None and (body := response.text.strip()):
        detail = f"{detail} | body: {body[:500]}"
    return detail


def ping_backend(backend: Backend, transport=None) -> BackendCheck:
    """Send ONE judge request to confirm the endpoint and key actually work.

    Only called behind an explicit flag — this spends real tokens.
    """
    check = check_key_presence(backend)
    if check.status == "skipped":
        return check
    try:
        JudgeClient(backend, transport=transport).judge(PING_PAIR)
    except Exception as exc:  # noqa: BLE001 - any failure is a failed ping
        return BackendCheck(
            name=backend.name, status="ready", detail=_failure_detail(exc), reachable=False
        )
    return BackendCheck(name=backend.name, status="ready", detail=check.detail, reachable=True)


def render_check_report(checks: list[BackendCheck]) -> str:
    ready = [c for c in checks if c.status == "ready"]
    skipped = [c for c in checks if c.status == "skipped"]
    lines = [f"{len(ready)} ready, {len(skipped)} skipped", ""]
    for c in ready:
        mark = {True: " [reachable]", False: " [UNREACHABLE]", None: ""}[c.reachable]
        lines.append(f"  ready   {c.name}: {c.detail}{mark}")
    for c in skipped:
        lines.append(f"  skipped {c.name}: {c.detail}")
    return "\n".join(lines)
