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
import queue
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Self

from judge.check import check_key_presence, ping_backend, render_check_report
from judge.client import (
    Backend,
    JudgeClient,
    LimiterRegistry,
    RateLimiter,
    limiter_host,
    load_backends,
    rate_limit_penalty,
)
from judge.prompts import PROMPT_VERSION
from judge.schema import (
    Pair,
    Usage,
    pair_from_dict,
    verdict_from_json_line,
    verdict_to_json_line,
)

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
        # Sorted, not insertion-ordered: dict(Counter(...)) follows the order
        # errors ARRIVED, which concurrency makes arbitrary. The counts were
        # always stable; without this the manifest's key order shuffles between
        # otherwise identical runs and diffs of two manifests become noise.
        "error_kinds": dict(sorted(Counter(errors).items())),
    }


def _usage_summary(usages: list[Usage | None]) -> dict:
    """What this run actually spent, summed from the per-call captures.

    Measured, not estimated — before these were captured, a cost model had to
    derive a chars/token ratio from an unrelated metering call.

    Two counts guard the totals rather than decorate them:

    * `calls_unmeasured` — a total summed over 2 of 600 calls is worse than no
      total, because it looks like one. Sums stay None until something reports.
    * `calls_with_cache_hit` — the three votes send byte-identical requests, so
      a host serving them from cache collapses majority-of-3 to n=1 and drives
      the flip rate to a spurious 0.0. That failure IMPROVES every metric while
      measuring nothing, and `cached_tokens` is the only in-band tell.
    """
    measured = [u for u in usages if u is not None]

    def total(field: str) -> int | None:
        values = [getattr(u, field) for u in measured if getattr(u, field) is not None]
        return sum(values) if values else None

    return {
        "calls_measured": len(measured),
        "calls_unmeasured": len(usages) - len(measured),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        # Kept out of completion_tokens on purpose: these bill as output and
        # roughly double a reasoning backend's real cost, and a single summed
        # number would hide which backend carries that weight.
        "reasoning_tokens": total("reasoning_tokens"),
        "cached_tokens": total("cached_tokens"),
        "calls_with_cache_hit": sum(1 for u in measured if (u.cached_tokens or 0) > 0),
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
    usages: list[Usage | None] | None = None,
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
        "usage": _usage_summary(usages or []),
    }


def load_pairs(path: Path) -> list[Pair]:
    return [pair_from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


class WriterGone(RuntimeError):
    """Raised in a worker when the results writer has died.

    Stops the pool so the run cannot go on spending paid API calls whose
    verdicts have nowhere to land. The writer's own exception is the root cause
    and is what ultimately surfaces, out of ResultWriter.__exit__.
    """


class ResultWriter:
    """The single writer for one results file, fed by a queue.

    The append-and-flush-per-line contract is what makes a killed run
    resumable, and it only holds while exactly one thing owns the handle. N
    worker threads writing directly would interleave partial lines and leave a
    torn file that `already_judged_ids` cannot parse. Workers hand finished
    lines here instead; this thread does every write, in queue order.
    """

    _STOP = object()
    _ACK_POLL_S = 0.25  # how often a blocked write() re-checks that the writer lives

    def __init__(self, path: Path) -> None:
        self._path = path
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._drain, name=f"writer-{path.name}")
        self._failure: BaseException | None = None

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._queue.put(self._STOP)
        self._thread.join()
        # A writer that died — full disk, revoked permissions — would otherwise
        # drop every verdict after it silently, and a multi-hour run would look
        # like it merely judged fewer pairs. Fail loudly instead.
        if self._failure is not None:
            raise self._failure

    @property
    def failure(self) -> BaseException | None:
        """The error that killed the writer, or None while it is healthy.

        Workers must consult this BEFORE spending another API call. Persistence
        being gone does not stop the run on its own: `write()` still accepts,
        the drain loop still runs, and every remaining worker keeps making PAID
        requests whose verdicts are then discarded. On a multi-hour sweep that
        is the worst failure this runner has — the money is gone by the time
        __exit__ reports it.
        """
        return self._failure

    def _open(self):
        """Seam: tests substitute a handle that starts failing mid-run."""
        return open(self._path, "a")

    def write(self, line: str) -> None:
        """Hand a line to the writer and WAIT until it is on disk.

        Returning before the flush would mean a kill could lose verdicts that
        were already paid for. The old serial loop flushed before issuing the
        next request, so at most the call in flight was at risk; an unwaited
        queue widens that to everything still queued. Small in practice, but
        this PR's central claim is that --concurrency 1 reproduces prior runs
        exactly, and a durability regression there undercuts the claim.

        The wait costs a local flush — sub-millisecond — against calls that
        take seconds, and workers still never touch the file themselves, so the
        single-writer guarantee is unchanged.
        """
        flushed = threading.Event()
        self._queue.put((line, flushed))
        # Poll rather than wait forever: a caller that reached write() after
        # the writer thread stopped would otherwise hang the run outright,
        # which is a worse failure than the lost line it is guarding.
        while not flushed.wait(timeout=self._ACK_POLL_S):
            if not self._thread.is_alive():
                raise WriterGone(f"results writer is gone: {self._failure!r}")
        if self._failure is not None:
            raise WriterGone(f"results writer died: {self._failure!r}")

    def _drain(self) -> None:
        try:
            with self._open() as fh:
                while (item := self._queue.get()) is not self._STOP:
                    line, flushed = item
                    try:
                        fh.write(line + "\n")
                        fh.flush()
                    finally:
                        # Release the waiting worker even on failure, or it
                        # blocks forever on a consumer that is about to die.
                        flushed.set()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            # Deliberately BaseException, not Exception. Anything at all that
            # kills this thread must be recorded, because a writer that dies
            # unnoticed turns a corrupted run into one that merely looks short.
            # It IS re-raised — on the caller's thread, out of __exit__.
            self._failure = exc
            # Discard whatever is still queued, releasing each waiter as we go.
            # write() blocks on its flush event, so a producer left unreleased
            # here would hang forever on a consumer that is no longer writing.
            while True:
                item = self._queue.get()
                if item is self._STOP:
                    break
                item[1].set()


class RunHealth:
    """Per-call timings and errors, collected across worker threads.

    Latency is the evidence base for issue #9 — an unguarded list.append can
    drop samples under concurrency, and a lost sample silently biases the p50
    that decides whether a backend is usable at pack scale.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latencies: list[float] = []
        self.failed_latencies: list[float] = []
        self.errors: list[str] = []
        self.usages: list[Usage | None] = []
        self._attempts = 0

    def claim_attempt(self) -> int:
        """1-based index of this attempt, counting failures — as enumerate did."""
        with self._lock:
            self._attempts += 1
            return self._attempts

    def record_ok(self, seconds: float, usage: Usage | None = None) -> None:
        with self._lock:
            self.latencies.append(seconds)
            # Appended even when None, so calls_unmeasured counts the calls a
            # host stayed silent about rather than losing them entirely.
            self.usages.append(usage)

    def record_failure(self, seconds: float, exc: Exception) -> None:
        with self._lock:
            # Time failures too — a timeout is the slowest call a backend
            # makes, and the one worth knowing about.
            self.failed_latencies.append(seconds)
            self.errors.append(type(exc).__name__)


def _handle_call_failure(
    backend: Backend,
    limiter: RateLimiter,
    exc: Exception,
    *,
    pair_id: str,
    run_index: int,
) -> None:
    """Report one failed vote, and slow the host if it was a 429.

    A 429 is the host saying the bucket is too generous, so the penalty lands
    on the HOST — every worker on it, and every other backend sharing it — not
    only the worker that happened to catch it. The call itself is not retried
    in-run; resume picks it up next launch, like any other failed vote.
    """
    penalty = rate_limit_penalty(exc)
    if penalty is not None:
        limiter.penalize(penalty)
        print(
            f"[{backend.name}] 429 from {limiter_host(backend.base_url)}: "
            f"holding that host back {penalty:g}s",
            file=sys.stderr,
        )
    print(f"[{backend.name}] {pair_id} vote {run_index}: ERROR {exc}", file=sys.stderr)


def _write_manifest(
    backend: Backend,
    pairs: list[Pair],
    out_dir: Path,
    *,
    votes: int,
    client: JudgeClient,
    health: RunHealth,
) -> None:
    """Record what this run actually sent, once the run is over.

    Written AFTER the calls so observed_models reflects every snapshot that
    answered, including a provider swapping the model mid-run.
    """
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
                latencies=health.latencies,
                errors=health.errors,
                failed_latencies=health.failed_latencies,
                usages=health.usages,
            ),
            indent=2,
        )
        + "\n"
    )


def run_backend(
    backend: Backend,
    pairs: list[Pair],
    out_dir: Path,
    *,
    votes: int,
    concurrency: int = 1,
    limiters: LimiterRegistry | None = None,
    client_factory: Callable[[Backend], JudgeClient] = JudgeClient,
    writer_factory: Callable[[Path], ResultWriter] = ResultWriter,
) -> None:
    """Judge every owed vote for one backend, `concurrency` calls in flight.

    At concurrency=1 this is the original serial loop: one worker consuming a
    FIFO queue in submission order, so the results file is written in exactly
    the pair-then-vote order prior runs produced.
    """
    results_path = out_dir / f"{backend.name}.jsonl"
    judged = already_judged_ids(
        results_path,
        model_id=backend.model_id,
        prompt_version=PROMPT_VERSION,
        temperature=backend.temperature,
        reasoning_effort=backend.reasoning_effort,
    )
    # Partitioned ONCE, before any worker starts. A worker that re-read the
    # results file could hand the same (pair_id, run_index) to two threads.
    todo = pending_votes(pairs, judged, votes=votes)
    label = " (eval-only backend)" if backend.eval_only else ""
    print(
        f"[{backend.name}]{label} {len(todo)} pending of {len(pairs)} pairs x {votes} votes"
    )
    if not todo:
        return

    client = client_factory(backend)
    limiter = (limiters or LimiterRegistry()).for_backend(backend)
    health = RunHealth()

    def judge_one(item: tuple[Pair, int], writer: ResultWriter) -> None:
        pair, run_index = item
        # Never spend a paid call we cannot persist. Checked before the limiter
        # so a dead writer stops the run in the time it takes the in-flight
        # calls to land, rather than at the end of a multi-hour sweep.
        if writer.failure is not None:
            raise WriterGone(f"results writer died: {writer.failure!r}")
        i = health.claim_attempt()
        limiter.wait()
        started = time.monotonic()
        try:
            verdict = client.judge(pair, run_index=run_index)
        except Exception as exc:  # noqa: BLE001 - keep going, resume covers gaps
            health.record_failure(time.monotonic() - started, exc)
            _handle_call_failure(backend, limiter, exc, pair_id=pair.id, run_index=run_index)
            return
        health.record_ok(time.monotonic() - started, verdict.usage)
        writer.write(verdict_to_json_line(verdict))
        if i % 20 == 0 or i == len(todo):
            print(f"[{backend.name}] {i}/{len(todo)}")

    try:
        with writer_factory(results_path) as writer:
            pool = ThreadPoolExecutor(
                max_workers=concurrency, thread_name_prefix=f"judge-{backend.name}"
            )
            try:
                futures = [pool.submit(judge_one, item, writer) for item in todo]
                for future in futures:
                    future.result()  # surface anything judge_one did not catch
            except BaseException:
                # Every vote is submitted up front, so a plain
                # shutdown(wait=True) would keep sending the REMAINING paid
                # requests — possibly for hours — before a Ctrl-C propagated.
                # An operator killing a run expects the spend to stop.
                # BaseException, not Exception: KeyboardInterrupt is the case
                # that matters most and is not an Exception.
                #
                # cancel_futures drops everything not yet started, which is
                # where the unspent money is. wait=True still lets the handful
                # already in flight finish: they are bounded by `concurrency`,
                # and abandoning them would leave a worker blocked in
                # writer.write() after ResultWriter.__exit__ had already joined
                # the writer thread — a hang instead of a clean stop.
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        _write_manifest(backend, pairs, out_dir, votes=votes, client=client, health=health)
    finally:
        client.close()


def run_slate(
    backends: list[Backend],
    pairs: list[Pair],
    out_dir: Path,
    *,
    votes: int,
    concurrency: int = 1,
    limiters: LimiterRegistry | None = None,
    client_factory: Callable[[Backend], JudgeClient] = JudgeClient,
    on_backend_done: Callable[[str], None] | None = None,
) -> None:
    """Run every backend on the slate.

    Backends run concurrently whenever concurrency > 1. Five of the eight
    slate backends sit on unrelated hosts and share nothing, so serializing
    them made the slate cost sum() where it could cost max() — and worse, a
    single melting-down host stalled every backend queued behind it.

    concurrency == 1 keeps them strictly serial, in slate order, so a run is
    reproducible byte for byte against the runs that came before this change.
    That includes failure behaviour: a raising backend aborts the slate, as it
    always did. Only the parallel path isolates failures, because there the
    other backends have already done work worth keeping.
    """
    if not backends:
        return
    limiters = limiters or LimiterRegistry()

    # Register the WHOLE slate before any backend sends anything. The registry
    # only ever tightens, but tightening after the calls have left is useless:
    # for_backend() used to be reached on each backend's own thread, so a
    # loose-rpm backend could run at its own rate until a stricter sibling on
    # the same host registered — and at concurrency 1 it could finish its
    # entire leg first. That is a quota burst on a shared host, which is the
    # one thing host-keying exists to prevent.
    for backend in backends:
        limiters.for_backend(backend)

    def run_one(backend: Backend) -> None:
        run_backend(
            backend,
            pairs,
            out_dir,
            votes=votes,
            concurrency=concurrency,
            limiters=limiters,
            client_factory=client_factory,
        )
        if on_backend_done is not None:
            on_backend_done(backend.name)

    if concurrency == 1:
        for backend in backends:
            run_one(backend)
        return

    with ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="backend") as pool:
        futures = {pool.submit(run_one, backend): backend for backend in backends}
        failures = []
        for future, backend in futures.items():
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report all, not just the first
                print(f"[{backend.name}] FAILED: {exc}", file=sys.stderr)
                failures.append(f"{backend.name}: {exc}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} backend(s) failed: " + "; ".join(failures)
        )


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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "calls in flight per backend; N>1 also runs backends in parallel "
            "(default 1 = fully serial, reproduces pre-concurrency runs exactly). "
            "Rate limits are enforced per HOST, so raising this cannot exceed a "
            "provider's quota no matter how many backends share an endpoint"
        ),
    )
    args = parser.parse_args()
    if args.votes < 1:
        parser.error("--votes must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

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

    runnable = []
    for backend in backends:
        if check_key_presence(backend).status == "skipped":
            print(f"[{backend.name}] SKIPPED: ${backend.api_key_env} is not set", file=sys.stderr)
            continue
        runnable.append(backend)

    run_slate(runnable, pairs, args.out, votes=args.votes, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
