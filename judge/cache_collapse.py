"""Detect a caching host collapsing majority-of-N into n=1 (issue #15).

The votes of a majority-of-N send byte-identical request payloads:
`JudgeClient.judge()` builds the body from `(backend, pair)` alone, and
`run_index` never enters the request. Against a host that caches RESPONSES —
not merely input-token prefixes — identical POSTs can return one identical
answer. Majority-of-3 becomes n=1, the flip rate reads exactly 0.0, and
`kappa_sd` falls toward 0.

That is the dangerous shape: **the metrics improve as the instrument stops
working.** Nothing reads as an error, a gap, or a shortfall — the run simply
looks better than the last one. So it has to be asserted, not noticed.

Neither signal is sufficient alone:

* Cache hits alone are NOT a collapse. Prompt-prefix caching is normal and
  reuses input tokens without reusing the response.
* A flat flip rate alone is NOT a collapse. A genuinely deterministic backend
  is the expected result for an easy pair set.

Only the conjunction is suspicious, and an absent measurement is its own
answer — "no cache hits" and "no data about cache hits" must never read alike.

Evidence comes from the VERDICT ROWS, not the run manifest. The manifest is
rewritten per launch (see the caveats section of scenario_report), while the
rows accumulate across resumes — so a resumed run's cumulative flip rate must
be judged against cumulative usage, or a final clean segment would vouch for
earlier segments it knows nothing about.

CALIBRATION, still open: `cached_tokens` is an input-side metric in both API
dialects, so it evidences prompt caching rather than response reuse, and this
harness sends the same rubric prefix on every pair. Whether `> 0` is the right
line, or whether it should be a fraction of calls, cannot be settled until a
real run carries usage data — no results file in the repo has any. The counts
are reported rather than reduced to a boolean so that judgement can be made
from a real run.
"""

from __future__ import annotations

from dataclasses import dataclass

from judge.schema import Verdict
from judge.vote import tally_votes

COLLAPSE_SUSPECTED = "collapse_suspected"
UNVERIFIABLE = "unverifiable"
OK = "ok"


@dataclass(frozen=True)
class CacheFinding:
    """What one backend's repeated votes and per-call usage say about caching."""

    backend: str
    status: str
    max_votes: int  # most votes seen on any ONE pair, not a total
    repeated_pairs: int  # pairs voted more than once — the whole evidence base
    verdict_flip_rate: float  # over repeated pairs only
    reason_flip_rate: float  # ditto
    calls_measuring_cache: int  # calls that reported a cached_tokens value
    calls_with_cache_hit: int
    total_calls: int


def _cache_counts(verdicts: list[Verdict]) -> tuple[int, int]:
    """(calls that MEASURED caching, calls that reported a hit).

    A row whose usage is absent, or whose usage omits `cached_tokens`, measured
    nothing about caching. Counting it as a zero would turn "we did not look"
    into "we looked and found none" — the error this module exists to prevent,
    committed against itself.
    """
    measuring = [v for v in verdicts if v.usage is not None and v.usage.cached_tokens is not None]
    hits = sum(1 for v in measuring if (v.usage.cached_tokens or 0) > 0)
    return len(measuring), hits


def cache_findings(by_model: dict[str, list[Verdict]]) -> list[CacheFinding]:
    """One finding per backend, in backend order.

    Flatness is measured over pairs that were actually voted MORE THAN ONCE.
    `flip_rate` over a single value is 0.0 by construction, so averaging across
    single-vote pairs lets a run that lost most of its votes to API errors
    dilute its mean toward zero and manufacture a collapse out of the handful
    of pairs that did repeat.
    """
    findings = []
    for backend, verdicts in sorted(by_model.items()):
        voted = tally_votes(verdicts)
        if not voted:
            continue
        repeated = [r for r in voted if r.n_votes >= 2]
        measuring, hits = _cache_counts(verdicts)

        if not repeated:
            # Repetition is the only thing that can reveal a reused response.
            status = OK
            verdict_flip = reason_flip = 0.0
        else:
            verdict_flip = sum(r.verdict_flip_rate for r in repeated) / len(repeated)
            reason_flip = sum(r.reason_flip_rate for r in repeated) / len(repeated)
            flat = verdict_flip == 0.0 and reason_flip == 0.0
            if not flat:
                status = OK
            elif measuring == 0:
                status = UNVERIFIABLE
            elif hits > 0:
                status = COLLAPSE_SUSPECTED
            else:
                status = OK

        findings.append(
            CacheFinding(
                backend=backend,
                status=status,
                max_votes=max(r.n_votes for r in voted),
                repeated_pairs=len(repeated),
                verdict_flip_rate=verdict_flip,
                reason_flip_rate=reason_flip,
                calls_measuring_cache=measuring,
                calls_with_cache_hit=hits,
                total_calls=len(verdicts),
            )
        )
    return findings


def render_cache_warning(findings: list[CacheFinding]) -> list[str]:
    """A loud section when anything is wrong, a one-line receipt when nothing is.

    Rendered ABOVE the stability table on purpose: these findings say that the
    numbers below them may be measuring nothing, and a warning printed after
    the thing it undermines arrives once it has already been believed.

    A clean result still prints. Silence would make "checked, clear" identical
    to a report from before this check existed — the same absent-versus-measured
    confusion the module argues against, one level up.
    """
    suspected = [f for f in findings if f.status == COLLAPSE_SUSPECTED]
    unverifiable = [f for f in findings if f.status == UNVERIFIABLE]
    checkable = [f for f in findings if f.repeated_pairs > 0]
    if not checkable:
        # No pair was voted twice anywhere, so nothing was checked and nothing
        # can be claimed. Saying "clear" here would be a false receipt.
        return []

    lines = ["## Vote independence", ""]
    if suspected:
        lines += [
            "**⚠️ A caching host may have collapsed majority-of-N to n=1.** These",
            "backends reported cache hits AND never once disagreed with themselves",
            "across repeated votes. The votes send byte-identical payloads, so a host",
            "that caches responses returns one answer N times — which reads as perfect",
            "stability rather than as a fault.",
            "",
            "Treat every stability number for these backends as unproven until a run",
            "with no cache hits reproduces it.",
            "",
            "| Backend | Repeated pairs | Verdict flip | Reason flip | Cache hits / measured |",
            "|---|---|---|---|---|",
        ]
        for f in suspected:
            lines.append(
                f"| {f.backend} | {f.repeated_pairs} | {f.verdict_flip_rate:.3f} | "
                f"{f.reason_flip_rate:.3f} | {f.calls_with_cache_hit} / {f.calls_measuring_cache} |"
            )
        lines.append("")
    if unverifiable:
        lines += [
            "**Flat, and not checkable.** These backends never disagreed with",
            "themselves, but no call reported a `cached_tokens` value, so whether a",
            "cache served them cannot be answered either way. This is NOT the same as",
            "no cache hits.",
            "",
            "| Backend | Repeated pairs | Verdict flip | Reason flip | Calls measuring cache |",
            "|---|---|---|---|---|",
        ]
        for f in unverifiable:
            lines.append(
                f"| {f.backend} | {f.repeated_pairs} | {f.verdict_flip_rate:.3f} | "
                f"{f.reason_flip_rate:.3f} | none of {f.total_calls} |"
            )
        lines.append("")
    if not suspected and not unverifiable:
        lines += [
            f"Checked against per-call usage: {len(checkable)} backend(s) clear — each",
            "either disagreed with itself somewhere, or measured zero cache hits.",
            "",
        ]
    return lines
