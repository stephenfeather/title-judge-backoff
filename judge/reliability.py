"""Reliability as a reported metric rather than an inference from coverage.

The bake-off exists to pick a judge for a large production catalogue. At that
scale a backend that cannot sustain the call volume is disqualified REGARDLESS
of its agreement scores — it never produces enough of them to score. Yet the
leaderboard ranks on kappa, accuracy and flip rates, and reliability appears
only as a coverage percentage (issue #43).

Coverage is the wrong instrument for two reasons.

It merges causes that are not equivalent. A backend at 40% because the host
timed out, one at 40% because it answered in the wrong shape, and one at 40%
because the operator stopped the run are three different findings that coverage
renders identically.

And it is silent about the shape of what IS there. Votes are attempted in
pairs-file order, so a backend that degrades partway through completes a
contiguous PREFIX of the file. The calibration file is ordered by cohort, so
that prefix is a cohort skew rather than a sample, and quality metrics computed
over it cannot be compared like-for-like against a complete backend's.

Both halves are measured here rather than asserted: `scope` records what the
evidence actually covers, and `CoverageShape.is_prefix` is derived from the
observed ordering instead of assumed from the row count.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Failure kinds that mean the host never answered in time. Matched by suffix
#: because httpx raises ReadTimeout, ConnectTimeout, WriteTimeout and
#: PoolTimeout, and a whitelist would silently reclassify the one it missed as
#: an ordinary error.
TIMEOUT_SUFFIX = "Timeout"

#: The exception judge.prompts.parse_judge_response raises when a model answers
#: in the wrong shape (issue #41). Named rather than imported so this module
#: stays free of the httpx-bearing import chain, as scenario_report requires.
CONTRACT_FAILURE = "JudgeResponseError"

#: The exception judge.client raises for a 429 that means "this account cannot
#: bill", as opposed to "slow down" (issue #52). Named here for the same reason
#: as CONTRACT_FAILURE — this module stays free of the httpx import chain.
QUOTA_FAILURE = "QuotaExhausted"

#: At or above this many attempts per completed vote, a backend needs more calls
#: than the set has votes. Not a tuned threshold — it is the point where the
#: majority of calls fail, so every resume pass costs more than it recovers and
#: the run does not converge. That is what "unusable at scale" means here.
UNUSABLE_ATTEMPTS_PER_VOTE = 2.0

#: How far past its own row count a partial backend's pairs may reach before it
#: stops being a contiguous prefix. Not zero: at --concurrency > 1 the sliding
#: submission window lets a few later pairs land ahead of earlier ones, so an
#: exact test would call a genuine prefix "spread" for the sake of a handful of
#: reordered rows.
PREFIX_SLACK = 1.1


@dataclass(frozen=True)
class Reliability:
    """What one backend attempted, and how it failed.

    Counts are `None`, never 0, when no manifest recorded them. The distinction
    is the whole point: kimi-k2.6 in the 2026-08-11 run was killed mid-sweep and
    wrote no manifest, so it has 255 rows on disk and no failure record at all.
    Rendering that as `failed=0` would give a clean bill of health to the one
    backend the run proved unusable.
    """

    backend: str
    rows_on_disk: int
    distinct_pairs: int
    attempted: int | None
    succeeded: int | None
    failed: int | None
    error_kinds: dict[str, int]
    launches: int | None
    #: "cumulative" (survives resume), "last-launch" (pre-#40 manifest), or
    #: "none" (no manifest). Reported, not hidden — a reader comparing two
    #: backends needs to know whether they are comparable.
    scope: str
    latency_p50: float | None
    latency_p95: float | None

    @property
    def timeouts(self) -> int:
        return sum(n for kind, n in self.error_kinds.items() if kind.endswith(TIMEOUT_SUFFIX))

    @property
    def contract_failures(self) -> int:
        return self.error_kinds.get(CONTRACT_FAILURE, 0)

    @property
    def quota_failures(self) -> int:
        return self.error_kinds.get(QUOTA_FAILURE, 0)

    @property
    def could_not_bill(self) -> bool:
        return self.quota_failures > 0

    @property
    def failure_rate(self) -> float | None:
        if not self.attempted:
            return None
        return (self.failed or 0) / self.attempted

    @property
    def attempts_per_vote(self) -> float | None:
        """Calls spent per vote actually obtained. A healthy backend is ~1.0."""
        if not self.succeeded:
            return None
        return (self.attempted or 0) / self.succeeded

    @property
    def unusable_at_scale(self) -> bool | None:
        """True, False, or None for "no evidence" — never False by default.

        A backend with no manifest cannot be cleared. Defaulting to False would
        clear precisely the backends that died before they could report.
        """
        if self.scope == "none" or self.attempted is None:
            return None
        if self.quota_failures and self.quota_failures == self.failed:
            # EVERY failure was a billing refusal, so this says nothing about
            # the model (issue #52). Zero rows for want of credit and zero rows
            # for want of throughput look identical and mean opposite things:
            # one is "add credit", the other is "do not pick this judge".
            # Reported under "could not bill" instead.
            return None
        if self.succeeded == 0:
            return True
        rate = self.attempts_per_vote
        return None if rate is None else rate >= UNUSABLE_ATTEMPTS_PER_VOTE


def _counts(manifest: dict) -> tuple[dict | None, str]:
    """The best available call counts for a backend, and what they cover.

    Prefers `cumulative` because it survives resume. Resume is the normal way a
    run completes here, so `health` alone describes the final, smallest launch
    segment — nemotron's 2026-08-11 manifest claims 24 calls and 0 failures for
    a backend with 600 rows on disk.
    """
    cumulative = manifest.get("cumulative")
    if cumulative:
        return cumulative, "cumulative"
    health = manifest.get("health")
    if health:
        return health, "last-launch"
    return None, "none"


def reliability_rows(
    by_model: dict[str, list], manifests: dict[str, dict]
) -> list[Reliability]:
    """One row per backend that produced verdicts or a manifest.

    Keyed off the union, so a backend that wrote rows but no manifest still
    appears — that is the case worth seeing, not the one to drop.
    """
    out = []
    for name in sorted(set(by_model) | set(manifests)):
        verdicts = by_model.get(name) or []
        manifest = manifests.get(name) or {}
        counts, scope = _counts(manifest)
        # Latency comes from `health` even when counts come from `cumulative`:
        # a median cannot be merged across launches, so the latest launch's
        # distribution is reported as its own thing rather than averaged into a
        # number that describes no run that ever happened (#40).
        health = manifest.get("health") or {}
        ok = counts.get("calls_ok") if counts else None
        failed = counts.get("calls_failed") if counts else None
        out.append(
            Reliability(
                backend=name,
                rows_on_disk=len(verdicts),
                distinct_pairs=len({v.pair_id for v in verdicts}),
                attempted=None if counts is None else (ok or 0) + (failed or 0),
                succeeded=ok,
                failed=failed,
                error_kinds=(counts or {}).get("error_kinds") or {},
                launches=(counts or {}).get("launches"),
                scope=scope,
                latency_p50=health.get("latency_median"),
                latency_p95=health.get("latency_p95"),
            )
        )
    return out


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def render_reliability_section(rows: list[Reliability]) -> list[str]:
    """The reliability table, plus prose for the backends that need calling out."""
    if not rows:
        return []
    lines = [
        "## Reliability",
        "",
        "Reliability is a **selection criterion**, not a footnote: a judge that",
        "cannot sustain the call volume is disqualified regardless of how well it",
        "agrees, because it never produces enough judgments to agree with. Read",
        "this table before the quality tables that follow (issue #43).",
        "",
        "| Backend | Evidence | Attempted | OK | Failed | Timeouts | Contract | Billing | "
        "Attempts/vote | p50 (s) | p95 (s) | Rows on disk |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        unknown = r.attempted is None
        lines.append(
            f"| {r.backend} | {r.scope} | "
            f"{'unknown' if unknown else r.attempted} | "
            f"{'unknown' if r.succeeded is None else r.succeeded} | "
            f"{'unknown' if r.failed is None else r.failed} | "
            f"{'unknown' if unknown else r.timeouts} | "
            f"{'unknown' if unknown else r.contract_failures} | "
            f"{'unknown' if unknown else r.quota_failures} | "
            f"{_num(r.attempts_per_vote)} | "
            f"{_num(r.latency_p50)} | {_num(r.latency_p95)} | {r.rows_on_disk} |"
        )
    lines += [
        "",
        "`Evidence` says what the counts cover. **cumulative** survives resume;",
        "**last-launch** is a manifest written before launch history existed and",
        "describes only the final resume segment, which is typically the smallest;",
        "**none** means no manifest was written at all.",
        "",
        "Latency is the LATEST launch only, in every case. A median cannot be",
        "merged with another median without the raw samples, so summing launches",
        "would produce a number describing no run that actually happened.",
        "",
    ]

    unfunded = [r for r in rows if r.could_not_bill]
    if unfunded:
        lines += [
            "### Could not bill",
            "",
            "These backends were refused for want of credit or quota, not for",
            "anything they did. **This is an account state, not a finding about the",
            "model** — a backend with zero rows here has not failed the bake-off, it",
            "was never allowed to enter (issue #52).",
            "",
            "The host returns HTTP 429 for this, the same status it uses for",
            "throttling, so it is easy to misread as a rate limit. It is not: no",
            "amount of backing off fixes it, and it is charged to the backend rather",
            "than to the host, so its neighbours on that endpoint are unaffected.",
            "",
        ]
        for r in unfunded:
            share = "" if r.failed is None else f" of {r.failed} failures"
            lines.append(f"- **{r.backend}** — {r.quota_failures} billing refusals{share}.")
        lines.append("")

    unusable = [r for r in rows if r.unusable_at_scale]
    if unusable:
        lines += [
            "### Unusable at scale",
            "",
            f"At or above {UNUSABLE_ATTEMPTS_PER_VOTE:g} attempts per completed vote, more than half of",
            "a backend's calls fail: every resume pass costs more than it recovers and",
            "the run does not converge. That is a result about the model, not an",
            "accident of the run, and it disqualifies the backend no matter what its",
            "agreement scores say.",
            "",
        ]
        for r in unusable:
            detail = f"{r.failed} of {r.attempted} calls failed"
            if r.timeouts:
                detail += f", {r.timeouts} of them timeouts"
            lines += [f"- **{r.backend}** — {detail}."]
        lines.append("")

    unknown = [r for r in rows if r.scope == "none"]
    if unknown:
        lines += [
            "### Reliability unknown",
            "",
            "These backends wrote verdicts but no manifest, so how often they failed",
            "is **unknown** — which is not the same as zero. A backend killed mid-sweep",
            "is exactly the one whose failures mattered most, and it is the one that",
            "records the least. Do not read a blank row as a clean run.",
            "",
        ]
        for r in unknown:
            lines.append(f"- **{r.backend}** — {r.rows_on_disk} rows on disk, no failure record.")
        lines.append("")
    return lines


@dataclass(frozen=True)
class CoverageShape:
    """Where in the pairs file a backend's completed rows actually sit."""

    backend: str
    distinct_pairs: int
    total_pairs: int
    #: 1 + the highest position this backend reached in the reference order.
    #: Equal to `distinct_pairs` for a contiguous prefix; equal to `total_pairs`
    #: for rows spread across the whole set.
    span: int
    #: "pairs-file" when ranks came from the calibration file, "reconstructed"
    #: when they were inferred from results order. Reported because the two are
    #: not equally trustworthy — see `coverage_shapes`.
    order_source: str = "reconstructed"
    #: Where `total_pairs` came from: "pairs-file", "manifest", or "observed".
    universe_source: str = "observed"

    @property
    def complete(self) -> bool:
        return self.distinct_pairs >= self.total_pairs

    @property
    def stopped_early(self) -> bool:
        """Whether the backend's furthest pair falls short of the end of the file.

        This is what separates a prefix from a set of holes, and the first
        version of `is_prefix` missed it. Gap COUNT cannot tell them apart: a
        backend that stopped at pair 100 having missed one, and a backend that
        ran all 200 having missed one, have identical gap counts and opposite
        meanings.
        """
        return self.span < self.total_pairs

    @property
    def is_prefix(self) -> bool:
        """Whether the completed rows form a contiguous run from the start.

        Measured, not assumed. A partial backend whose rows are genuinely spread
        across the file IS closer to a sample, and calling that a prefix would be
        a claim the data refutes.

        Two conditions, and the first is the one that was missing. The 2026-08-12
        v3 sweep produced backends at 197/200 whose furthest pair was position
        200 — they finished the file and dropped a few pairs to transient errors.
        The slack test alone passed them (197 x 1.1 = 216, above the whole
        200-pair set), so ANY backend past ~91% coverage was labelled a prefix
        regardless of shape, and the report claimed cohort skew that was not
        there.
        """
        if self.complete:
            return False
        return self.stopped_early and self.span <= self.distinct_pairs * PREFIX_SLACK


def _reconstructed_order(by_model: dict[str, list]) -> list[str]:
    """Pairs-file order inferred from results, best effort.

    Rows are appended when their call RETURNS, so at --concurrency > 1 a results
    file records completion order rather than submission order. The backend that
    saw the most pairs is used as the reference, and every pair only other
    backends saw is appended after it — unrankable, but still real.
    """
    reference_name = max(sorted(by_model), key=lambda n: len({v.pair_id for v in by_model[n]}))
    order: list[str] = []
    seen: set[str] = set()
    for v in by_model[reference_name]:
        if v.pair_id not in seen:
            seen.add(v.pair_id)
            order.append(v.pair_id)
    for name in sorted(by_model):
        for pair_id in sorted({v.pair_id for v in by_model[name]} - seen):
            seen.add(pair_id)
            order.append(pair_id)
    return order


def coverage_shapes(
    by_model: dict[str, list],
    *,
    pair_order: list[str] | None = None,
    total_pairs: int | None = None,
) -> list[CoverageShape]:
    """How each backend's rows are positioned within the pairs file.

    `pair_order` — the calibration file's ids, in file order — is authoritative
    and should be passed whenever it is available. Without it the order is
    reconstructed from the results, which is only an approximation: rows are
    appended when their HTTP call returns, so at --concurrency > 1 a file records
    COMPLETION order. A genuine cohort prefix can be shuffled past `PREFIX_SLACK`
    that way and reported as "spread across the set", suppressing the warning.

    `total_pairs` — the manifests' `n_pairs` — is the universe when no pairs file
    is given. Without it the observed union stands in, and that is wrong in the
    case that matters most: if EVERY backend stopped at the same point, the union
    is itself a prefix, the most-complete backend measures as complete, and no
    bias caveat renders at all. The observed count is still a floor, since a
    stale manifest must never shrink a set below the rows actually on disk.
    """
    if not by_model:
        return []
    if pair_order:
        order = list(pair_order)
        order_source, universe_source = "pairs-file", "pairs-file"
    else:
        order = _reconstructed_order(by_model)
        order_source = "reconstructed"
        universe_source = "manifest" if total_pairs else "observed"
    rank = {pair_id: i for i, pair_id in enumerate(order)}

    # Anything judged but absent from the reference (a pairs file that has moved
    # on, say) still happened. Rank it after everything known rather than drop it.
    for name in sorted(by_model):
        for pair_id in sorted({v.pair_id for v in by_model[name]} - set(rank)):
            rank[pair_id] = len(rank)

    universe = max(len(order), len(rank), total_pairs or 0)
    if universe > len(rank):
        # The manifest knows about pairs nobody reached. They are unrankable but
        # they are in the denominator, which is the entire point.
        universe_source = "manifest" if pair_order is None else "pairs-file"

    shapes = []
    for name in sorted(by_model):
        ids = {v.pair_id for v in by_model[name]}
        if not ids:
            continue
        shapes.append(
            CoverageShape(
                backend=name,
                distinct_pairs=len(ids),
                total_pairs=universe,
                span=max(rank[p] for p in ids) + 1,
                order_source=order_source,
                universe_source=universe_source,
            )
        )
    return shapes


def render_coverage_bias_caveat(shapes: list[CoverageShape]) -> list[str]:
    """The caveat that must travel with a partial backend's quality numbers."""
    partial = [s for s in shapes if not s.complete]
    if not partial:
        return []
    lines = [
        "### Partial backends are measured on less than the full set",
        "",
        "Every quality number below for these backends is computed over a subset,",
        "so comparing it against a complete backend's full-set figures is not",
        "quite like-for-like. **How much that matters depends on the shape**, which",
        "is why the shape is measured rather than assumed:",
        "",
        "- **contiguous prefix** — the backend stopped partway through. Votes are",
        "  attempted in pairs-file order and the calibration set is ordered by",
        "  cohort (most-changed, quarantined-brand, representative, unchanged), so",
        "  a prefix is a **cohort skew**, not a random sample. This is the case",
        "  that can move a number materially, and the direction depends on which",
        "  cohorts it reached.",
        "- **spread, stopped early** — stopped short, but with holes throughout.",
        "  Weaker cohort skew than a clean prefix.",
        "- **scattered gaps, ran to the end** — the backend covered the whole file",
        "  and lost individual calls along the way. There is no cohort story here;",
        "  read it as a handful of failed calls, and check the reliability table",
        "  above for why they failed.",
        "",
        "| Backend | Pairs judged | Of | Reaches position | Shape |",
        "|---|---|---|---|---|",
    ]
    for s in partial:
        if s.is_prefix:
            shape = "contiguous prefix"
        elif s.stopped_early:
            shape = "spread, stopped early"
        else:
            # Ran the whole file and lost a few calls on the way. Neither of the
            # other two labels fits, and both imply a sampling story where the
            # honest answer is "some calls failed".
            shape = "scattered gaps, ran to the end"
        lines.append(
            f"| {s.backend} | {s.distinct_pairs} | {s.total_pairs} | {s.span} | {shape} |"
        )
    lines += [
        "",
        "`Reaches position` is how far into the file the backend's furthest pair",
        "sits. Equal to the pairs judged means a clean prefix; close to the total",
        "means the rows are spread out and the cohort skew is weaker.",
        "",
    ]
    order_source = partial[0].order_source
    universe_source = partial[0].universe_source
    if order_source == "pairs-file":
        lines.append("Positions are ranked against the calibration file itself.")
    else:
        lines += [
            "Positions are **reconstructed** from results order, not read from the",
            "calibration file: rows are appended when their call returns, so a",
            "concurrent run records completion order. Treat `Shape` as indicative.",
            "Pass `--pairs` to rank against the file itself.",
        ]
    if universe_source == "observed":
        lines += [
            "",
            "The total is the **observed** union of judged pairs, not a known set size.",
            "If every backend stopped at the same point, that union is itself a prefix",
            "and this table understates how much of the set is missing.",
        ]
    elif universe_source == "manifest":
        lines.append("The total comes from the manifests' recorded pair count.")
    lines.append("")
    return lines
