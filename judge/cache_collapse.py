"""Detect a caching host collapsing majority-of-N into n=1 (issue #15).

The three votes of a majority-of-N send byte-identical request payloads:
`JudgeClient.judge()` builds the body from `(backend, pair)` alone, and
`run_index` never enters the request. Against a host that caches RESPONSES —
not merely input-token prefixes — three identical POSTs can return one identical
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

Only the conjunction is suspicious, and the absence of usage data is its own
answer — "no cache hits" and "no data about cache hits" must not read alike.
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
    """What one backend's votes and usage block say about response caching."""

    backend: str
    status: str
    n_votes: int
    verdict_flip_rate: float
    reason_flip_rate: float
    calls_with_cache_hit: int | None  # None = the host reported no usage at all
    calls_unmeasured: int | None  # None = no usage block, so not even a count


def _usage(manifest: dict) -> tuple[int | None, int | None]:
    """(calls_with_cache_hit, calls_unmeasured), with None for "never reported".

    A run whose calls were all unmeasured knows nothing about caching, so its
    zero cache hits are absence of evidence rather than evidence of absence.
    """
    usage = manifest.get("usage") or {}
    if not usage:
        # No block at all: not "zero unmeasured", which would read as "all
        # measured" beside a finding that says the run is not checkable.
        return None, None
    if usage.get("calls_measured", 0) == 0:
        return None, int(usage.get("calls_unmeasured", 0) or 0)
    hits = usage.get("calls_with_cache_hit")
    return (None if hits is None else int(hits)), int(usage.get("calls_unmeasured", 0) or 0)


def cache_findings(
    by_model: dict[str, list[Verdict]], manifests: dict[str, dict]
) -> list[CacheFinding]:
    """One finding per backend, in backend order.

    A backend judged at only ONE vote per pair is never flagged: `flip_rate`
    over a single value is 0.0 by construction, so every `--votes 1` run would
    otherwise report a collapse on every backend. Repetition is the only thing
    that can reveal a reused response, so without it there is nothing to say.
    """
    findings = []
    for backend, verdicts in sorted(by_model.items()):
        voted = tally_votes(verdicts)
        if not voted:
            continue
        n = len(voted)
        max_votes = max(r.n_votes for r in voted)
        verdict_flip = sum(r.verdict_flip_rate for r in voted) / n
        reason_flip = sum(r.reason_flip_rate for r in voted) / n
        hits, unmeasured = _usage(manifests.get(backend, {}))

        flat = verdict_flip == 0.0 and reason_flip == 0.0
        if max_votes < 2 or not flat:
            status = OK
        elif hits is None:
            status = UNVERIFIABLE
        elif hits > 0:
            status = COLLAPSE_SUSPECTED
        else:
            status = OK

        findings.append(
            CacheFinding(
                backend=backend,
                status=status,
                n_votes=max_votes,
                verdict_flip_rate=verdict_flip,
                reason_flip_rate=reason_flip,
                calls_with_cache_hit=hits,
                calls_unmeasured=unmeasured,
            )
        )
    return findings


def render_cache_warning(findings: list[CacheFinding]) -> list[str]:
    """A loud section, or nothing at all when every backend is fine.

    Rendered ABOVE the stability table on purpose: these findings say that the
    numbers below them may be measuring nothing, and a warning printed after
    the thing it undermines has already been believed.
    """
    suspected = [f for f in findings if f.status == COLLAPSE_SUSPECTED]
    unverifiable = [f for f in findings if f.status == UNVERIFIABLE]
    if not suspected and not unverifiable:
        return []

    lines = ["## ⚠️ Vote independence", ""]
    if suspected:
        lines += [
            "**A caching host may have collapsed majority-of-N to n=1.** The votes",
            "below reported cache hits AND never once disagreed with themselves. The",
            "three votes send byte-identical payloads, so a host that caches responses",
            "returns one answer three times — which reads as perfect stability.",
            "",
            "Treat every stability number for these backends as unproven until a run",
            "with cache hits at zero reproduces it.",
            "",
            "| Backend | Votes | Verdict flip | Reason flip | Calls with cache hit |",
            "|---|---|---|---|---|",
        ]
        for f in suspected:
            lines.append(
                f"| {f.backend} | {f.n_votes} | {f.verdict_flip_rate:.3f} | "
                f"{f.reason_flip_rate:.3f} | {f.calls_with_cache_hit} |"
            )
        lines.append("")
    if unverifiable:
        lines += [
            "**Flat, and not checkable.** These backends never disagreed with",
            "themselves, but reported no usage data, so whether a cache served them",
            "cannot be answered either way. This is not the same as no cache hits.",
            "",
            "| Backend | Votes | Verdict flip | Reason flip | Calls unmeasured |",
            "|---|---|---|---|---|",
        ]
        for f in unverifiable:
            lines.append(
                f"| {f.backend} | {f.n_votes} | {f.verdict_flip_rate:.3f} | "
                f"{f.reason_flip_rate:.3f} | "
                f"{'—' if f.calls_unmeasured is None else f.calls_unmeasured} |"
            )
        lines.append("")
    return lines
