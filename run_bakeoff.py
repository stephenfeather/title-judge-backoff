# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27"]
# ///
"""Run the judge bake-off: every backend judges every calibration pair.

Usage:
    uv run run_bakeoff.py --backends backends.toml --data pairs.jsonl --out results/2026-08-05/

Resumable: verdicts are appended to results/<out>/<backend-name>.jsonl as they
arrive; re-running skips pairs that backend has already judged. Calls are
rate-limited per backend's rpm. API keys are read from the environment only
(see backends.toml api_key_env).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from judge.check import check_key_presence, ping_backend, render_check_report
from judge.client import Backend, JudgeClient, RateLimiter, load_backends
from judge.prompts import PROMPT_VERSION
from judge.schema import Pair, pair_from_dict, verdict_from_json_line, verdict_to_json_line

# Majority-of-3 is the efficient point: it cuts effective flip rate from ~13% to
# ~5% and judge-noise sd(kappa) from ~0.048 to ~0.030 — below the +/-0.057
# sampling floor of a 200-pair calibration set — for 3x cost. N=5 buys a further
# ~0.01 for 5x. Repetition is the lever because temperature is not available:
# see judge/client.py TEMPERATURE_OMITTED.
DEFAULT_VOTES = 3


def already_judged_ids(
    results_path: Path,
    *,
    model_id: str,
    prompt_version: str,
    temperature: float | None,
    reasoning_effort: str | None,
) -> set[tuple[str, int]]:
    """(pair_id, run_index) pairs already judged under the SAME run config.

    The identity is the pair AND the vote index: under majority-of-N the same
    pair is judged N times deliberately, so keying on pair_id alone would treat
    one completed vote as "this pair is done" and silently collapse the run to
    N=1.

    Raises ValueError if the file holds verdicts from a different model_id,
    prompt_version, temperature, or reasoning effort — silently resuming over
    those would mix incomparable verdicts. Note that temperature=None (field
    omitted) and temperature=0.0 (field sent) are different configs, and that
    effort changes the reason code the model returns. Use a fresh --out
    directory for a new run config.
    """
    if not results_path.exists():
        return set()
    expected = (model_id, prompt_version, temperature, reasoning_effort)
    ids = set()
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        v = verdict_from_json_line(line)
        found = (v.model_id, v.prompt_version, v.temperature, v.reasoning_effort)
        if found != expected:
            raise ValueError(
                f"{results_path} contains verdicts from a different run config: "
                f"found (model_id={v.model_id!r}, prompt_version={v.prompt_version!r}, "
                f"temperature={v.temperature!r}, reasoning_effort={v.reasoning_effort!r}), "
                f"expected (model_id={model_id!r}, prompt_version={prompt_version!r}, "
                f"temperature={temperature!r}, reasoning_effort={reasoning_effort!r}). "
                f"Use a fresh --out directory for a new run config."
            )
        ids.add((v.pair_id, v.run_index))
    return ids


def pending_votes(
    pairs: list[Pair], judged: set[tuple[str, int]], *, votes: int
) -> list[tuple[Pair, int]]:
    """Every (pair, vote index) still owed, in pair-then-vote order.

    A run interrupted partway through a pair's votes resumes at the missing
    vote rather than re-judging the pair or skipping it.
    """
    return [
        (pair, run_index)
        for pair in pairs
        for run_index in range(votes)
        if (pair.id, run_index) not in judged
    ]


# Keys are fetched from the macOS Keychain by get_secret(), which is defined in
# 002_functions and consumed by 042_env_ai_tokens. Sourcing the tokens file
# ALONE sets every key to empty, which is worse than not sourcing it — so both
# must be sourced, in this order, in the same shell as the run.
ENV_PREFIX = (
    "zsh -c 'source ~/.zsh/zshrc.d/002_functions && "
    "source ~/.zsh/zshrc.d/042_env_ai_tokens && <command>'"
)


def skipped_backend_names(backends: list[Backend]) -> list[str]:
    """Backends whose API key is absent, in slate order."""
    return [b.name for b in backends if check_key_presence(b).status == "skipped"]


def render_skip_warning(names: list[str]) -> str:
    """Loud, actionable warning about backends that would silently vanish.

    A missing key is not an error anywhere in this harness — the backend is
    simply skipped. On a multi-hour unattended sweep that is only discovered
    when the report comes back short, so this says what to do about it.
    """
    return "\n".join(
        [
            "",
            "=" * 72,
            f"REFUSING TO START: {len(names)} backend(s) would be SILENTLY SKIPPED",
            "=" * 72,
            *(f"  - {name}" for name in names),
            "",
            "A missing key does not fail the run — the backend just vanishes from",
            "the results, which on a long sweep is only noticed at the report.",
            "",
            "Most likely cause: keys come from the macOS Keychain and were not",
            "loaded into this shell. Launch with BOTH files sourced, in order:",
            "",
            f"  {ENV_PREFIX}",
            "",
            "Sourcing 042_env_ai_tokens alone sets every key to EMPTY, because",
            "get_secret() is defined in 002_functions.",
            "",
            "If the omission is deliberate, re-run with --allow-skipped.",
            "=" * 72,
            "",
        ]
    )


def _health_summary(
    latencies: list[float], errors: list[str], failed_latencies: list[float] | None = None
) -> dict:
    """Per-backend operational health for the scenario report.

    A ping gives one latency sample; a full run gives a distribution, and that
    is what decides whether a backend is usable at pack scale. Median rather
    than mean because one 180s timeout would otherwise swamp 599 fast calls.

    FAILED calls are timed too, and reported separately. They are typically the
    slowest — a 180s timeout is the whole reason latency matters — so recording
    only successes made a backend that timed out on every call look instant.
    Kept in their own fields so a slow failure is never read as a slow success.
    """
    ordered = sorted(latencies)
    failed = sorted(failed_latencies or [])
    return {
        "calls_ok": len(latencies),
        "calls_failed": len(errors),
        "latency_min": ordered[0] if ordered else None,
        "latency_median": statistics.median(ordered) if ordered else None,
        "latency_max": ordered[-1] if ordered else None,
        "failed_latency_median": statistics.median(failed) if failed else None,
        "failed_latency_max": failed[-1] if failed else None,
        "error_kinds": dict(Counter(errors)),
    }


def run_manifest(
    backend: Backend,
    *,
    votes: int,
    prompt_version: str,
    n_pairs: int,
    sample_payload: dict,
    observed_models: set[str],
    latencies: list[float] | None = None,
    errors: list[str] | None = None,
    failed_latencies: list[float] | None = None,
) -> dict:
    """Everything needed to reconstruct what this run actually sent.

    The Responses API exposes no system_fingerprint, so a recorded payload plus
    the set of resolved snapshot strings is the only drift-detection mechanism
    available. Written next to the verdicts for every run.
    """
    return {
        "backend": backend.name,
        "model_id": backend.model_id,
        "base_url": backend.base_url,
        "api": backend.api,
        "temperature": backend.temperature,
        "reasoning_effort": backend.reasoning_effort,
        "structured_output": backend.structured_output,
        "votes": votes,
        "prompt_version": prompt_version,
        "n_pairs": n_pairs,
        "request_payload": sample_payload,
        "observed_models": sorted(observed_models),
        "health": _health_summary(latencies or [], errors or [], failed_latencies or []),
    }


def load_pairs(path: Path) -> list[Pair]:
    return [pair_from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def run_backend(backend: Backend, pairs: list[Pair], out_dir: Path, *, votes: int) -> None:
    results_path = out_dir / f"{backend.name}.jsonl"
    judged = already_judged_ids(
        results_path,
        model_id=backend.model_id,
        prompt_version=PROMPT_VERSION,
        temperature=backend.temperature,
        reasoning_effort=backend.reasoning_effort,
    )
    todo = pending_votes(pairs, judged, votes=votes)
    label = " (eval-only backend)" if backend.eval_only else ""
    print(
        f"[{backend.name}]{label} {len(todo)} pending of {len(pairs)} pairs x {votes} votes"
    )
    if not todo:
        return

    client = JudgeClient(backend)
    limiter = RateLimiter(rpm=backend.rpm)
    latencies: list[float] = []
    failed_latencies: list[float] = []
    errors: list[str] = []
    try:
        with open(results_path, "a") as fh:
            for i, (pair, run_index) in enumerate(todo, 1):
                limiter.wait()
                started = time.monotonic()
                try:
                    verdict = client.judge(pair, run_index=run_index)
                except Exception as exc:  # noqa: BLE001 - keep going, resume covers gaps
                    # Time failures too — a timeout is the slowest call a
                    # backend makes, and the one worth knowing about.
                    failed_latencies.append(time.monotonic() - started)
                    errors.append(type(exc).__name__)
                    print(
                        f"[{backend.name}] {pair.id} vote {run_index}: ERROR {exc}",
                        file=sys.stderr,
                    )
                    continue
                latencies.append(time.monotonic() - started)
                fh.write(verdict_to_json_line(verdict) + "\n")
                fh.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"[{backend.name}] {i}/{len(todo)}")
        # Written after the run so observed_models reflects every snapshot that
        # actually answered, including a mid-run swap.
        manifest_path = out_dir / f"{backend.name}.manifest.json"
        manifest_path.write_text(
            json.dumps(
                run_manifest(
                    backend,
                    votes=votes,
                    prompt_version=PROMPT_VERSION,
                    n_pairs=len(pairs),
                    sample_payload=client.request_body(pairs[0]),
                    observed_models=client.observed_models,
                    latencies=latencies,
                    errors=errors,
                    failed_latencies=failed_latencies,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", type=Path, default=Path("backends.toml"))
    parser.add_argument("--data", type=Path, help="calibration pairs.jsonl (never committed)")
    parser.add_argument("--out", type=Path, help="output directory, e.g. results/2026-08-05/")
    parser.add_argument(
        "--check-backends",
        action="store_true",
        help="validate the slate (key presence per backend) and exit without judging",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="with --check-backends, also send ONE real request per backend (spends tokens)",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=DEFAULT_VOTES,
        help=(
            f"judge each pair N times and take the majority verdict "
            f"(default {DEFAULT_VOTES}; N=1 disables voting and costs 1x)"
        ),
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="proceed even when some backends have no API key (default: refuse)",
    )
    args = parser.parse_args()
    if args.votes < 1:
        parser.error("--votes must be at least 1")

    backends = load_backends(args.backends)

    if args.check_backends:
        check = ping_backend if args.ping else check_key_presence
        print(render_check_report([check(b) for b in backends]))
        return

    if args.data is None or args.out is None:
        parser.error("--data and --out are required unless --check-backends is given")

    # Fail fast, before hours of work: a backend with no key is skipped, not
    # failed, so an unnoticed env problem quietly produces a short report.
    skipped = skipped_backend_names(backends)
    if skipped and not args.allow_skipped:
        print(render_skip_warning(skipped), file=sys.stderr)
        raise SystemExit(2)

    pairs = load_pairs(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    for backend in backends:
        if check_key_presence(backend).status == "skipped":
            print(f"[{backend.name}] SKIPPED: ${backend.api_key_env} is not set", file=sys.stderr)
            continue
        run_backend(backend, pairs, args.out, votes=args.votes)


if __name__ == "__main__":
    main()
