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
import itertools
import json
import queue
import statistics
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Self

from judge.check import check_key_presence, ping_backend, render_check_report
from judge.client import (
    Backend,
    JudgeClient,
    LimiterRegistry,
    RateLimiter,
    config_digest,
    limiter_host,
    load_backends,
    rate_limit_penalty,
)
from judge.provenance import DirtyTree, GitUnavailable, resolve_code_version
from judge.prompts import PROMPT_VERSION, prompt_variant_counts
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


def _manifest_beside(results_path: Path) -> dict | None:
    """The sidecar manifest for a results file, or None if unusable.

    Rows written before issue #13 carry no provenance, but `_write_manifest` has
    always recorded base_url next to them. For a legacy file that sidecar is the
    only evidence of which host produced it, and ignoring it would turn every
    pre-existing results directory into a free pass.
    """
    path = results_path.with_name(results_path.stem + ".manifest.json")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def already_judged_ids(
    results_path: Path,
    *,
    model_id: str,
    prompt_version: str,
    temperature: float | None,
    reasoning_effort: str | None,
    base_url: str | None = None,
    config_digest: str | None = None,
    code_version: str | None = None,
    allow_unknown_provenance: bool = False,
) -> set[tuple[str, int]]:
    """(pair_id, run_index) pairs already judged under the SAME run config.

    The identity is the pair AND the vote index: under majority-of-N the same
    pair is judged N times deliberately, so keying on pair_id alone would treat
    one completed vote as "this pair is done" and silently collapse the run to
    N=1.

    Raises ValueError if the file holds verdicts from a different model_id,
    prompt_version, temperature, reasoning effort, host, request configuration,
    or code version — silently resuming over any of those mixes incomparable
    verdicts into a file that claims to be one run. Note that temperature=None
    (field omitted) and temperature=0.0 (field sent) are different configs, and
    that effort changes the reason code the model returns. Use a fresh --out
    directory for a new run config.

    Provenance (issue #13) is compared per field rather than as a tuple, because
    absent must mean "unknown, do not compare" and not "mismatch": every row
    written before #13 lacks all three, and reading those as a mismatch would
    invalidate results/ wholesale. Unknown is then resolved against the sidecar
    manifest, and a file that can prove nothing is refused unless the caller
    explicitly accepts it.

    The provenance arguments default to None for callers that do not track it;
    passing None disables that field's check entirely. run_backend always passes
    all three.
    """
    if not results_path.exists():
        return set()
    expected = (model_id, prompt_version, temperature, reasoning_effort)
    ids = set()
    unproven = False
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
        _check_provenance(
            results_path,
            found_base_url=v.base_url,
            found_config_digest=v.config_digest,
            found_code_version=v.code_version,
            base_url=base_url,
            config_digest=config_digest,
            code_version=code_version,
        )
        # ANY expected field the row cannot answer leaves it unproven. Requiring
        # every field to be absent let a partially stamped row (host recorded,
        # code version not) satisfy the guard while _check_provenance skipped
        # the null — resuming it under different code, silently.
        if (
            (base_url is not None and v.base_url is None)
            or (config_digest is not None and v.config_digest is None)
            or (code_version is not None and v.code_version is None)
        ):
            unproven = True
        ids.add((v.pair_id, v.run_index))

    # Only a caller that tracks provenance can be let down by its absence. When
    # none was passed there is nothing to prove the file against, and demanding
    # proof would break every call site that predates #13.
    tracks_provenance = any(x is not None for x in (base_url, config_digest, code_version))
    if unproven and tracks_provenance:
        _resolve_unproven_file(
            results_path,
            base_url=base_url,
            allow_unknown_provenance=allow_unknown_provenance,
        )
    return ids


def _check_provenance(
    results_path: Path,
    *,
    found_base_url: str | None,
    found_config_digest: str | None,
    found_code_version: str | None,
    base_url: str | None,
    config_digest: str | None,
    code_version: str | None,
) -> None:
    """Raise if a row's recorded provenance contradicts this run's.

    Compared only where BOTH sides know the answer: a None on the row is a
    legacy record, and a None in the expectation is a caller that does not
    track provenance. Neither is evidence of a mismatch.
    """
    if found_base_url is not None and base_url is not None and found_base_url != base_url:
        raise ValueError(
            f"{results_path} contains verdicts served by a different host: "
            f"found base_url={found_base_url!r}, expected base_url={base_url!r}. "
            f"Two providers serving the same model_id are not one run. "
            f"Use a fresh --out directory."
        )
    if found_code_version is not None and code_version is not None and found_code_version != code_version:
        raise ValueError(
            f"{results_path} contains verdicts produced by different code: "
            f"found code_version={found_code_version!r}, expected code_version={code_version!r}. "
            f"Use a fresh --out directory for a new code version."
        )
    if (
        found_config_digest is not None
        and config_digest is not None
        and found_config_digest != config_digest
    ):
        raise ValueError(
            f"{results_path} contains verdicts from a different backend configuration: "
            f"found config_digest={found_config_digest!r}, expected config_digest={config_digest!r}. "
            f"Every field recorded on the row matches, so the difference is one the row does not "
            f"name — api or structured_output. Compare "
            f"{results_path.with_name(results_path.stem + '.manifest.json').name} against the "
            f"backend entry you are running. Use a fresh --out directory."
        )


def _resolve_unproven_file(
    results_path: Path,
    *,
    base_url: str | None,
    allow_unknown_provenance: bool,
) -> None:
    """Decide whether a file containing provenance-less rows may be resumed.

    The sidecar manifest is consulted first: it records the host even for runs
    that predate per-row provenance, so most legacy files get a real check
    rather than a free pass. Only a file with neither row provenance nor a
    readable manifest requires the caller to accept the risk explicitly.
    """
    manifest = _manifest_beside(results_path)
    if manifest is not None and base_url is not None:
        recorded = manifest.get("base_url")
        if recorded is not None and recorded != base_url:
            raise ValueError(
                f"{results_path} holds verdicts with no recorded host, and its manifest says "
                f"they were served by base_url={recorded!r}, not {base_url!r}. "
                f"Use a fresh --out directory."
            )
        if recorded is not None:
            return
    if not allow_unknown_provenance:
        raise ValueError(
            f"{results_path} holds verdicts with no recorded host or code version, and no "
            f"manifest beside it establishes one. Nothing can prove this file is compatible "
            f"with the current run. Use a fresh --out directory, or pass "
            f"--allow-unknown-provenance to resume it anyway."
        )


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


#: What makes two launches the same run. Deliberately the fields the row-based
#: resume guard compares in `already_judged_ids` — the manifest history has to
#: mean the same thing "one run" means everywhere else.
LAUNCH_IDENTITY_FIELDS = (
    "model_id",
    "base_url",
    "prompt_version",
    "temperature",
    "reasoning_effort",
    "config_digest",
    "code_version",
)


def _same_run(previous: dict, identity: dict) -> bool:
    """Whether a prior manifest describes the same configuration as this launch.

    Absent on either side means UNKNOWN, not different — manifests written
    before provenance existed carry no config_digest or code_version, and
    treating that as a mismatch would discard every legacy history. Same rule
    the resume guard uses for provenance on a row.
    """
    for field in LAUNCH_IDENTITY_FIELDS:
        before, now = previous.get(field), identity.get(field)
        if before is not None and now is not None and before != now:
            return False
    return True


def accumulate_launches(
    previous: dict | None, health: dict, identity: dict | None = None
) -> tuple[list[dict], dict]:
    """Fold this launch's health into the history from previous launches.

    Resume is the NORMAL way a run completes here, and the manifest used to be
    rewritten wholesale each time — so the surviving record described only the
    final, smallest launch. A backend that lost 50 calls in its first launch
    reported `calls_failed: 0` once a 47-call resume finished (issue #40).

    Counts are summed; LATENCY IS NOT. A median cannot be merged with another
    median — combining them needs the raw samples, and a median-of-medians is
    simply a wrong number. So each launch keeps its own distribution, where it
    is meaningful, and only the summable fields appear in the cumulative view.
    That is also the honest shape: 1.5s over a 47-call resume is not comparable
    to 12s over a 600-call run, and averaging them would imply it is.

    A manifest written before this existed carries its one launch in `health`;
    that is adopted as the first entry rather than discarded, because it is the
    only record of that launch.
    """
    previous = previous or {}
    discarded = 0
    if identity is not None and previous and not _same_run(previous, identity):
        # A launch that failed every call writes a manifest but NO verdict rows,
        # so the row-based guard has nothing to compare and cannot fire. Without
        # this check the old configuration's failures would be filed under the
        # new one — the wrong provider blamed for another's timeouts.
        #
        # Dropped rather than refused: with no rows there is nothing to protect,
        # so blocking a legitimate retry after a config fix would be
        # disproportionate. Counted, because silently losing a failure record is
        # the thing this function exists to prevent.
        discarded = len(previous.get("launches") or ([previous["health"]] if previous.get("health") else []))
        previous = {}

    launches = list(previous.get("launches") or [])
    if not launches and previous.get("health"):
        launches = [previous["health"]]
    launches.append(health)

    error_kinds: Counter[str] = Counter()
    for entry in launches:
        error_kinds.update(entry.get("error_kinds") or {})
    cumulative = {
        "launches": len(launches),
        "calls_ok": sum(e.get("calls_ok", 0) for e in launches),
        "calls_failed": sum(e.get("calls_failed", 0) for e in launches),
        # Sorted for the same reason the per-launch block is: dict order from a
        # Counter follows arrival, which concurrency makes arbitrary, and the
        # manifest must diff cleanly between runs.
        "error_kinds": dict(sorted(error_kinds.items())),
    }
    if discarded:
        cumulative["discarded_prior_launches"] = discarded
    return launches, cumulative


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
    code_version: str | None = None,
    config_digest: str | None = None,
    prompt_variants: dict[str, int] | None = None,
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
        # Provenance (issue #13). base_url is above and predates this; these two
        # are what a config_digest mismatch tells the operator to come read.
        "code_version": code_version,
        "config_digest": config_digest,
        # How many pairs rendered each system-prompt variant. Since prompt v2
        # the system prompt describes only the fields a pair actually has, and
        # request_payload samples only pairs[0], so on a mixed corpus this
        # histogram is the only record of what was sent. Joint, not marginal:
        # per-attribute counts cannot tell one pair carrying both fields from
        # two pairs carrying one each. See issue #14.
        "prompt_variants": prompt_variants or {},
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
    code_version: str | None = None,
    config_digest: str | None = None,
) -> None:
    """Record what this run actually sent, once the run is over.

    Written AFTER the calls so observed_models reflects every snapshot that
    answered, including a provider swapping the model mid-run.
    """
    manifest_path = out_dir / f"{backend.name}.manifest.json"
    manifest = run_manifest(
        backend,
        votes=votes,
        prompt_version=PROMPT_VERSION,
        n_pairs=len(pairs),
        sample_payload=client.request_body(pairs[0]),
        observed_models=client.observed_models,
        code_version=code_version,
        config_digest=config_digest,
        prompt_variants=prompt_variant_counts(pairs),
        latencies=health.latencies,
        errors=health.errors,
        failed_latencies=health.failed_latencies,
        usages=health.usages,
    )
    # Fold this launch into the history rather than replacing it. `health` stays
    # the LATEST launch so existing readers are unaffected; `launches` and
    # `cumulative` are what survive a resume (issue #40).
    try:
        previous = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        previous = None
    manifest["launches"], manifest["cumulative"] = accumulate_launches(
        previous, manifest["health"], identity=manifest
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def run_backend(
    backend: Backend,
    pairs: list[Pair],
    out_dir: Path,
    *,
    votes: int,
    concurrency: int = 1,
    limiters: LimiterRegistry | None = None,
    code_version: str | None = None,
    allow_unknown_provenance: bool = False,
    stop: threading.Event | None = None,
    judge_client: Callable[..., JudgeClient] = JudgeClient,
    client_factory: Callable[[Backend], JudgeClient] | None = None,
    writer_factory: Callable[[Path], ResultWriter] = ResultWriter,
) -> None:
    """Judge every owed vote for one backend, `concurrency` calls in flight.

    At concurrency=1 this is the original serial loop: one worker consuming a
    FIFO queue in submission order, so the results file is written in exactly
    the pair-then-vote order prior runs produced.

    `code_version` is resolved once for the whole sweep by the caller (see
    main), not per backend: one run, one code version, even if the tree changes
    while a multi-hour sweep is in flight.
    """
    results_path = out_dir / f"{backend.name}.jsonl"
    digest = config_digest(backend, prompt_version=PROMPT_VERSION)
    judged = already_judged_ids(
        results_path,
        model_id=backend.model_id,
        prompt_version=PROMPT_VERSION,
        temperature=backend.temperature,
        reasoning_effort=backend.reasoning_effort,
        base_url=backend.base_url,
        config_digest=digest,
        code_version=code_version,
        allow_unknown_provenance=allow_unknown_provenance,
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

    # The default path builds the client itself so the run's code_version cannot
    # be left off the rows. `client_factory` stays as the full override for tests
    # that substitute a fake; `judge_client` swaps only the class, keeping the
    # provenance wiring intact.
    client = client_factory(backend) if client_factory else judge_client(backend, code_version=code_version)
    limiter = (limiters or LimiterRegistry()).for_backend(backend)
    health = RunHealth()

    def judge_one(item: tuple[Pair, int], writer: ResultWriter) -> None:
        pair, run_index = item
        # The slate has been told to stop. Checked HERE, on the worker thread,
        # because that is the only place the decision reaches a running backend:
        # KeyboardInterrupt is delivered to the main thread alone, and
        # cancel_futures only drops tasks that have not started yet. At
        # --concurrency 8 every backend is already inside its own pool, so
        # without this check they keep judging — and keep spending — for as long
        # as their work lists last.
        #
        # Returning rather than raising: a stop is an operator decision, not a
        # backend failure, and recording it in error_kinds would put an
        # operator's Ctrl-C on a model's reliability record.
        if stop is not None and stop.is_set():
            return
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
                # Submit a sliding window rather than the whole todo list.
                # One Future per pending vote is 120,000 of them per backend at
                # the 40k-pair scale #9 exists to serve, each holding a lock, a
                # condition variable, a callback list and its closure arguments
                # until the backend finishes (issue #17). The window is sized by
                # `concurrency`, so the pending set no longer grows with the
                # work list.
                #
                # 2x concurrency, not 1x: a worker that finishes finds queued
                # work already waiting instead of idling until the main thread
                # wakes up to submit more.
                #
                # Order is unchanged at concurrency=1. The executor's queue is
                # FIFO and there is a single worker, so tasks still run in
                # submission order and the results file is written in the same
                # pair-then-vote order as before.
                queued = iter(todo)
                window = 2 * concurrency
                pending = {
                    pool.submit(judge_one, item, writer)
                    for item in itertools.islice(queued, window)
                }
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()  # surface anything judge_one did not catch
                    # Top back up to the window. When `queued` is exhausted this
                    # adds nothing and the loop drains what is still in flight.
                    #
                    # A set stop drains rather than tops up. judge_one's own
                    # check is what makes the stop correct — this only keeps a
                    # stopped backend from submitting the remaining 120,000
                    # tasks just to have each one return immediately.
                    if stop is not None and stop.is_set():
                        continue
                    pending |= {
                        pool.submit(judge_one, item, writer)
                        for item in itertools.islice(queued, len(done))
                    }
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
    except BaseException:
        # Already failing. Record what this launch learned — a backend that was
        # killed used to leave NO manifest at all, so the one whose failures
        # mattered most left the least evidence (issue #40) — but never let a
        # manifest problem replace the traceback explaining why the run stopped.
        try:
            _write_manifest(
                backend, pairs, out_dir, votes=votes, client=client, health=health,
                code_version=code_version, config_digest=digest,
            )
        except Exception as exc:  # noqa: BLE001 - the original failure wins
            print(f"[{backend.name}] could not write manifest: {exc!r}", file=sys.stderr)
        raise
    else:
        # Succeeded, so a manifest failure is THE failure. Suppressing it here
        # would let a run whose audit metadata and usage totals never reached
        # disk exit 0, and automation would read that as complete.
        _write_manifest(
            backend, pairs, out_dir, votes=votes, client=client, health=health,
            code_version=code_version, config_digest=digest,
        )
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
    code_version: str | None = None,
    allow_unknown_provenance: bool = False,
    client_factory: Callable[[Backend], JudgeClient] | None = None,
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

    # Set once, read by every backend's workers. This is the slate's only way to
    # reach a backend that is already running: the interrupt lands on the main
    # thread, and the backends are elsewhere.
    stop = threading.Event()

    def run_one(backend: Backend) -> None:
        run_backend(
            backend,
            pairs,
            out_dir,
            votes=votes,
            concurrency=concurrency,
            limiters=limiters,
            code_version=code_version,
            allow_unknown_provenance=allow_unknown_provenance,
            stop=stop,
            client_factory=client_factory,
        )
        if on_backend_done is not None:
            on_backend_done(backend.name)

    if concurrency == 1:
        for backend in backends:
            run_one(backend)
        return

    pool = ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="backend")
    failures = []
    try:
        futures = {pool.submit(run_one, backend): backend for backend in backends}
        # as_completed, not futures.items(): a backend that dies must be seen
        # when it dies. Iterating in slate order blocks on the FIRST backend, so
        # an in-worker BaseException from any other one would not be noticed —
        # and the stop not set — until the slowest backend ahead of it finished.
        for future in as_completed(futures):
            backend = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - report all, not just the first
                print(f"[{backend.name}] FAILED: {exc}", file=sys.stderr)
                failures.append(f"{backend.name}: {exc}")
    except BaseException:
        # A Ctrl-C, or a BaseException raised inside a backend. Previously the
        # `with` block's __exit__ ran shutdown(wait=True) with no cancellation,
        # so an operator's interrupt bought them a wait for the ENTIRE slate to
        # finish paying for itself. Set the stop FIRST — the running backends
        # are what the money is going to; cancel_futures only helps the ones
        # still queued behind them.
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
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
        "--allow-dirty",
        action="store_true",
        help=(
            "run from a working tree with uncommitted changes (default: refuse). "
            "The verdicts are tagged with a hash of the diff so the run stays "
            "identifiable, but it cannot be reproduced from a commit"
        ),
    )
    parser.add_argument(
        "--allow-unknown-provenance",
        action="store_true",
        help=(
            "resume a results file whose rows predate provenance and that has no "
            "manifest to establish a host (default: refuse). Nothing can prove such "
            "a file was produced by the run you are about to continue"
        ),
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

    # Fail closed BEFORE any paid call: a run whose code cannot be identified
    # writes rows that can never be audited, which is the defect this guards
    # against. Resolved once for the whole sweep so every row of every backend
    # carries one value even if the tree changes mid-run.
    try:
        code_version = resolve_code_version(allow_dirty=args.allow_dirty)
    except (GitUnavailable, DirtyTree) as exc:
        print(f"refusing to start: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"code version: {code_version}")

    pairs = load_pairs(args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    runnable = []
    for backend in backends:
        if check_key_presence(backend).status == "skipped":
            print(f"[{backend.name}] SKIPPED: ${backend.api_key_env} is not set", file=sys.stderr)
            continue
        runnable.append(backend)

    # One resolved value feeds BOTH consumers: the guard that compares against
    # the existing file, and the client that stamps the new rows. run_backend
    # builds the client from this same argument, so the two cannot diverge.
    run_slate(
        runnable,
        pairs,
        args.out,
        votes=args.votes,
        concurrency=args.concurrency,
        code_version=code_version,
        allow_unknown_provenance=args.allow_unknown_provenance,
    )


if __name__ == "__main__":
    main()
