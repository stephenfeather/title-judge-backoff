# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Scenario report for a bake-off run with NO operator rulings yet.

The E10 pack ships 200 title changes and zero verdicts, so kappa-vs-human is
not computable. Everything here is what IS measurable without ground truth:

  * per-backend stability — flip rates for verdicts and reason codes, tracked
    separately, because effort was measured to move the reason while leaving
    the binary verdict alone;
  * cross-model agreement — where the candidate judges actually diverge;
  * reliability — attempts, failures by kind and latency tail per backend,
    reported as a selection criterion rather than left to be inferred from a
    coverage percentage, since a judge that cannot sustain the call volume is
    disqualified however well it agrees;
  * operational health — latency distribution and error kinds per backend;
  * a ruling queue — items ranked by how contested they are, which is the
    handoff to the operator's ruling session. Contested items are where the
    rubric is ambiguous, so ruling those first buys the most calibration per
    minute of human attention.

Usage:
    uv run scenario_report.py --results results/2026-08-06/ --out scenario-report.md
"""

from __future__ import annotations

import argparse
import json
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path

from judge.agreement import agreement_matrix, reason_cross_tab, reason_distribution
from judge.cache_collapse import cache_findings, render_cache_warning
from judge.deadlocks import backend_deadlocks, deadlock_records, render_deadlock_section
from judge.reliability import (
    coverage_shapes,
    reliability_rows,
    render_coverage_bias_caveat,
    render_reliability_section,
)
from judge.prompts import PROMPT_VERSION
from judge.schema import RETIRED_REASON_CODES, Verdict, verdict_from_json_line
from judge.vote import tally_votes


@dataclass(frozen=True)
class ContentionRow:
    """How contested one pair is, across models and across repeated votes."""

    pair_id: str
    n_models: int
    cross_model_disagreement: float  # fraction of models differing from the modal verdict
    cross_model_reason_disagreement: float  # ditto for reason codes
    mean_verdict_flip_rate: float  # mean within-model flip rate across models
    mean_reason_flip_rate: float
    n_unsettled: int = 0  # models whose own votes reached no majority on this pair

    @property
    def contention(self) -> float:
        """Single sortable score. Cross-model disagreement dominates, since two
        models disagreeing is stronger evidence of an ambiguous rubric than one
        model wavering between its own votes.

        A model that could not decide AT ALL counts for more than one that
        decided differently from its neighbour. That is the point of issue #12:
        the queue used to rank an unsettled pair on whatever its tie-break
        invented, which could push it up or bury it depending on whether the
        invented value happened to match another model's. Now it ranks on the
        fact of being unsettled, which is exactly the ambiguous-rubric signal
        the queue exists to surface.
        """
        unsettled_fraction = self.n_unsettled / self.n_models if self.n_models else 0.0
        return (
            2 * self.cross_model_disagreement
            + self.cross_model_reason_disagreement
            + self.mean_verdict_flip_rate
            + 0.5 * self.mean_reason_flip_rate
            + 2 * unsettled_fraction
        )


def load_results(results_dir: Path) -> dict[str, list[Verdict]]:
    """{backend name: verdicts}, one entry per <backend>.jsonl in the directory."""
    return {
        path.stem: [
            verdict_from_json_line(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        for path in sorted(results_dir.glob("*.jsonl"))
    }


def load_manifests(results_dir: Path) -> dict[str, dict]:
    return {
        path.name.removesuffix(".manifest.json"): json.loads(path.read_text())
        for path in sorted(results_dir.glob("*.manifest.json"))
    }


def _disagreement(values: list[str]) -> float:
    """Fraction of values differing from the most common one."""
    if not values:
        return 0.0
    winner = max(set(values), key=values.count)
    return sum(v != winner for v in values) / len(values)


def contention_ranking(by_model: dict[str, list[Verdict]]) -> list[ContentionRow]:
    """Every pair, most contested first."""
    # Collapse each model's votes, keeping the within-model flip rates.
    voted = {model: {r.pair_id: r for r in tally_votes(vs)} for model, vs in by_model.items()}

    pair_ids = sorted({pid for rulings in voted.values() for pid in rulings})
    rows = []
    for pid in pair_ids:
        present = [rulings[pid] for rulings in voted.values() if pid in rulings]
        # Cross-model disagreement compares only the models that actually
        # reached a majority. An unsettled model has no ruling to disagree
        # with; its contribution is counted as n_unsettled instead, which the
        # contention score weights directly.
        rows.append(
            ContentionRow(
                pair_id=pid,
                n_models=len(present),
                n_unsettled=sum(1 for r in present if not r.settled),
                cross_model_disagreement=_disagreement(
                    [r.verdict for r in present if r.verdict is not None]
                ),
                cross_model_reason_disagreement=_disagreement(
                    [r.reason.value for r in present if r.reason is not None]
                ),
                mean_verdict_flip_rate=sum(r.verdict_flip_rate for r in present) / len(present),
                mean_reason_flip_rate=sum(r.reason_flip_rate for r in present) / len(present),
            )
        )
    return sorted(rows, key=lambda r: (-r.contention, r.pair_id))


def _stability_table(by_model: dict[str, list[Verdict]]) -> list[str]:
    lines = [
        "| Backend | Pairs | Votes | Verdict flip | Reason flip | Unstable pairs |",
        "|---|---|---|---|---|---|",
    ]
    for model, verdicts in sorted(by_model.items()):
        voted = tally_votes(verdicts)
        n = len(voted)
        if n == 0:
            # A backend whose every call errored judged nothing. Dividing by n
            # here took the WHOLE report down, so one dead backend destroyed the
            # output for every healthy one. Report it as errored instead — its
            # absence is itself a finding worth seeing.
            lines.append(f"| {model} | 0 | 0 | — | — | no verdicts (all calls failed) |")
            continue
        unstable = sum(1 for r in voted if r.verdict_flip_rate or r.reason_flip_rate)
        lines.append(
            f"| {model} | {n} | {max((r.n_votes for r in voted), default=0)} | "
            f"{sum(r.verdict_flip_rate for r in voted) / n:.3f} | "
            f"{sum(r.reason_flip_rate for r in voted) / n:.3f} | {unstable} |"
        )
    return lines


def _reason_cross_tab_sections(by_model: dict[str, list[Verdict]]) -> list[str]:
    """Reason-code cross-tabs for each pair of backends.

    The per-backend distribution shows WHAT each model reaches for; this shows
    WHERE two models diverge on the same item. Off-diagonal mass is the case the
    binary verdict hides — both said reject, for different reasons — which is
    exactly the shape reasoning effort was measured to produce.
    """
    models = sorted(by_model)
    if len(models) < 2:
        return []  # nothing to cross-tabulate against
    lines: list[str] = []
    for i, left in enumerate(models):
        for right in models[i + 1 :]:
            tab = reason_cross_tab(by_model[left], by_model[right])
            if not tab:
                continue
            lines.append(f"**{left}** (rows) vs **{right}** (columns)")
            lines.append("")
            lines.append(f"| {left} \\ {right} | Reason | Count |")
            lines.append("|---|---|---|")
            # Total order: count descending, then the reason pair. Sorting on
            # count alone left ties in dict insertion order, which comes from
            # iterating a SET intersection — and Python randomizes str hashing
            # per process, so two runs over identical data produced different
            # files. That makes "regenerate the report and diff it" unusable as
            # a check, which is the verification this project keeps relying on.
            for (lreason, rreason), count in sorted(tab.items(), key=lambda kv: (-kv[1], kv[0])):
                marker = "" if lreason == rreason else " ⟵ divergence"
                lines.append(f"| {lreason} | {rreason} | {count}{marker} |")
            lines.append("")
    return lines


def _agreement_table(by_model: dict[str, list[Verdict]]) -> list[str]:
    models = sorted(by_model)
    matrix = agreement_matrix(by_model)
    lines = ["| | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for left in models:
        cells = []
        for right in models:
            value = matrix[(left, right)]
            cells.append("n/a" if value is None else f"{value:.3f}")
        lines.append(f"| **{left}** | " + " | ".join(cells) + " |")
    return lines


def _seconds(health: dict, key: str) -> str:
    """One latency figure, or "-" when the run never recorded it.

    Module level, taking `health` as an argument, rather than a closure defined
    inside the loop over backends. The closure form captured the loop variable
    by reference (ruff B023): it was accidentally correct only because every
    call happened eagerly within the same iteration, and any future change that
    deferred one — a lazy join, a generator, reuse after the loop — would have
    given every backend the LAST backend's latencies, silently and plausibly.
    """
    value = health.get(key)
    return "-" if value is None else f"{value:.1f}"


def _health_table(manifests: dict[str, dict]) -> list[str]:
    lines = [
        "| Backend | Effort | Temp | OK | Failed | Latency min/median/max (s) | "
        "Failed latency median/max (s) | Errors | Snapshots |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in sorted(manifests.items()):
        h = m.get("health", {})
        errors = h.get("error_kinds") or {}
        error_text = ", ".join(f"{k}x{v}" for k, v in sorted(errors.items())) or "none"
        lines.append(
            f"| {name} | {m.get('reasoning_effort') or 'default'} | "
            f"{'omitted' if m.get('temperature') is None else m['temperature']} | "
            f"{h.get('calls_ok', 0)} | {h.get('calls_failed', 0)} | "
            f"{_seconds(h, 'latency_min')} / {_seconds(h, 'latency_median')} / {_seconds(h, 'latency_max')} | "
            # Failures are timed separately — a backend that times out on every
            # call would otherwise show no latency at all.
            f"{_seconds(h, 'failed_latency_median')} / {_seconds(h, 'failed_latency_max')} | "
            f"{error_text} | {', '.join(m.get('observed_models') or []) or '-'} |"
        )
    return lines


def _completion_table(
    by_model: dict[str, list[Verdict]], manifests: dict[str, dict]
) -> list[str]:
    """Rows actually on disk per backend, next to what the health block claims.

    `health.calls_ok` covers only the LAST launch segment, so any backend that
    was resumed under-reports — four of seven manifests in the 2026-08-06 run
    misreport completion this way. Completion is the row count and the distinct
    pair count; the health block is a per-launch operational record and nothing
    more. Both are shown together so the discrepancy is visible rather than a
    thing you have to know.
    """
    lines = [
        "| Backend | Rows on disk | Distinct pairs | health.calls_ok (last launch only) |",
        "|---|---|---|---|",
    ]
    for name in sorted(by_model):
        verdicts = by_model[name]
        claimed = manifests.get(name, {}).get("health", {}).get("calls_ok")
        lines.append(
            f"| {name} | {len(verdicts)} | {len({v.pair_id for v in verdicts})} | "
            f"{'-' if claimed is None else claimed} |"
        )
    return lines


def unsettled_reason_pairs(verdicts: list[Verdict]) -> list[str]:
    """Pairs whose reason votes reached no majority.

    Asks `tally_votes` rather than re-deriving the rule. The first version of
    this function looked for "every reason different", which is not the same
    question: four votes splitting 2-2 are equally undecided and were missed.
    One definition of "settled", in judge/vote.py, and everything else asks it.
    """
    return sorted(r.pair_id for r in tally_votes(verdicts) if not r.reason_settled)


def _caveats_section(by_model: dict[str, list[Verdict]], manifests: dict[str, dict]) -> list[str]:
    lines = [
        "## Caveats — read before quoting any number above",
        "",
        "### Completion is rows on disk, never health.calls_ok",
        "",
        "The manifest health block describes only the **last launch segment**, so a",
        "backend that was resumed reports far fewer calls than it actually has rows.",
        "Use this table for completion:",
        "",
        *_completion_table(by_model, manifests),
        "",
    ]

    fabricated = {
        name: unsettled_reason_pairs(verdicts)
        for name, verdicts in by_model.items()
        if unsettled_reason_pairs(verdicts)
    }
    lines += [
        "### Reason codes with no majority are fabricated, not agreed",
        "",
        "Where no reason code holds the top count on its own there is **no**",
        "**majority**, so these pairs now carry no reason at all rather than one the",
        "tie-break invented. They are EXCLUDED from the reason distributions and",
        "the per-model reason tables above, and from `score.py`'s reason confusion —",
        "listed here so the exclusion is visible rather than silent (issue #12).",
        "",
        "Their flip rates are still counted: an unsettled pair is a contested pair,",
        "not a missing one, and it ranks high in the ruling queue for that reason.",
        "",
    ]
    if fabricated:
        for name in sorted(fabricated):
            lines.append(f"- **{name}**: {', '.join(fabricated[name])}")
    else:
        lines.append("- None in this run.")
    lines += ["", *_compliance_caveat(manifests)]
    lines += ["", *_host_concentration_caveat(by_model, manifests), ""]
    return lines


#: The exception `judge.prompts.parse_judge_response` raises when a model answers
#: in the wrong shape. Named here rather than imported: this script declares
#: dependencies = [] and judge.prompts pulls in judge.schema.
CONTRACT_FAILURE = "JudgeResponseError"


def _compliance_caveat(manifests: dict[str, dict]) -> list[str]:
    """Backends that answered, but not in the shape the contract requires.

    Separated from transport failures on purpose (issue #41). A dropped
    connection says nothing about a model; returning a reason code in the
    verdict field says a great deal. Coverage merges the two, and the merge
    flatters the non-compliant model: its fumbled pairs vanish from its own
    accuracy and kappa, which are then computed over the subset it managed.
    """
    offenders = []
    for name, m in sorted(manifests.items()):
        # Prefer the cumulative block: it survives resume, where `health`
        # describes only the last launch and would report a clean run for a
        # backend that failed hundreds of calls earlier (issue #40).
        counts = m.get("cumulative") or m.get("health") or {}
        violations = (counts.get("error_kinds") or {}).get(CONTRACT_FAILURE, 0)
        if violations:
            attempted = counts.get("calls_ok", 0) + counts.get("calls_failed", 0)
            offenders.append((name, violations, attempted))
    if not offenders:
        return []

    lines = [
        "### Output-contract compliance",
        "",
        "These backends ANSWERED but not in the required shape — a reason code in",
        "the verdict field, unparseable JSON, an unknown reason. That is evidence",
        "about the model, not about the network, and it is counted separately for",
        "that reason.",
        "",
        "Such calls write no verdict row, so the pairs simply vanish from the",
        "backend's accuracy and kappa rather than counting against them. Read a",
        "high violation count as a quality finding, not merely as lost coverage.",
        "",
        "| Backend | Contract failures | Calls attempted (this launch) |",
        "|---|---|---|",
    ]
    for name, violations, attempted in offenders:
        lines.append(f"| {name} | {violations} | {attempted or 'unknown'} |")
    lines += [
        "",
        "Counts are cumulative across every launch of this run, falling back to the",
        "last launch for a manifest written before launch history existed.",
        "",
    ]
    return lines


def _host_concentration_caveat(
    by_model: dict[str, list[Verdict]], manifests: dict[str, dict]
) -> list[str]:
    """How much of this run sat behind one rate-limit bucket.

    Derived from the manifests rather than written down, because the prose
    version went stale: it named six backends on a host the slate no longer
    uses at all, while the real concentration had moved elsewhere and grown.
    A caveat that has to be hand-edited to stay true will eventually be false.
    """
    # Read the host from the VERDICT ROWS first, falling back to the manifest.
    # Rows carry base_url since #13 and survive an interrupted run; the manifest
    # is only written when a backend completes, so a backend that was killed is
    # invisible to a manifest-only count. That undercounted this very report by
    # one backend before the fallback was added (see #40).
    hosts: dict[str, list[str]] = {}
    for name in sorted(set(by_model) | set(manifests)):
        rows = by_model.get(name) or []
        base_url = next((r.base_url for r in rows if r.base_url), None)
        if base_url is None:
            base_url = manifests.get(name, {}).get("base_url")
        if base_url:
            # netloc, not judge.client.limiter_host: this script declares
            # dependencies = [] and importing that module pulls in httpx.
            # The schemeless fallback there is dead anyway — Backend refuses
            # a non-HTTPS base_url at load since #27, so netloc is always set.
            hosts.setdefault(urlparse(base_url).netloc, []).append(name)
    if not hosts:
        # Absent, not "nothing shared" — the same distinction the cache and
        # deadlock sections draw. Silence here would read as "hosts checked,
        # none shared", which is a stronger claim than no manifest supports.
        return [
            "### Cross-host parallelism is unknown for this run",
            "",
            "No manifest recorded a base_url, so how many backends shared a rate-limit",
            "bucket cannot be determined. That is not the same as no host being shared.",
            "",
        ]

    top, shared = max(hosts.items(), key=lambda kv: len(kv[1]))
    total = sum(len(v) for v in hosts.values())
    lines = [
        "### Cross-host parallelism buys less than the backend count suggests",
        "",
        f"This run used {total} backend(s) across {len(hosts)} host(s). Rate limits are",
        "enforced per HOST, so backends sharing one endpoint share one bucket and do",
        "not run independently no matter how high `--concurrency` goes.",
        "",
    ]
    if len(shared) > 1:
        lines += [
            f"**{len(shared)} of {total} share `{top}`**: "
            + ", ".join(f"`{n}`" for n in sorted(shared)) + ".",
            "",
            "A 429 from any one of them slows all of them, and a slow model on that",
            "host delays its neighbours rather than only itself.",
            "",
        ]
    else:
        lines += ["No two backends in this run shared a host.", ""]
    return lines


def _retired_code_note(by_model: dict[str, list[Verdict]]) -> list[str]:
    """Name any reason code in this run that the rubric no longer offers.

    A run judged under an older PROMPT_VERSION keeps its rows, so a retired code
    goes on appearing in the distribution above forever. Without this the reader
    has no way to tell a code the models declined to use from one they were never
    offered, and would compare its count against a current run as though the two
    had the same menu (issue #44).

    The scan is over raw VOTES, while the distribution table above counts
    settled majorities — a code that fired only in scattered minority votes is
    still named here, which is why the note says "appeared in votes" rather
    than pointing at counts the table may not render. Rows are split on the
    prompt version actually recorded on them rather than assumed: a retired
    code on a CURRENT-version row is not history, it is a model that emitted
    a code it was never offered, and the note must not claim otherwise.
    """
    rows = [
        v for verdicts in by_model.values() for v in verdicts if v.reason in RETIRED_REASON_CODES
    ]
    if not rows:
        return []
    historical = sorted({v.reason.value for v in rows if v.prompt_version != PROMPT_VERSION})
    current = sorted({v.reason.value for v in rows if v.prompt_version == PROMPT_VERSION})
    lines = [""]
    if historical:
        lines += [
            f"**Retired code(s) present: {', '.join(f'`{c}`' for c in historical)}.** These rows",
            "were judged under an earlier prompt version that still offered the code; the",
            "current rubric does not. The code appeared in votes, so do not compare its",
            "counts against a run judged under the current prompt — the two were offered",
            "different menus.",
        ]
    if current:
        lines += [
            f"**Retired code(s) on current-version ({PROMPT_VERSION}) rows: "
            f"{', '.join(f'`{c}`' for c in current)}.** {PROMPT_VERSION} never offered",
            "these codes, so each such row is a model emitting a code it was never",
            "offered — the contract breach the parse path exists to refuse — not",
            "evidence the code describes something in this corpus.",
        ]
    return lines


def render_scenario_report(
    by_model: dict[str, list[Verdict]],
    manifests: dict[str, dict],
    *,
    queue_size: int = 40,
    leg: str = "",
) -> str:
    ranking = contention_ranking(by_model)
    contested = [r for r in ranking if r.contention > 0]

    lines = [
        "# Scenario report",
        "",
        "**There are no operator rulings for this set, so Cohen's kappa and",
        "accuracy are not computable and are deliberately absent.** Everything",
        "below is measurable without ground truth. Nothing here says which judge",
        "is *correct* — only which are stable and where they disagree.",
        "",
        # Above the stability table deliberately: these findings say the numbers
        # below them may be measuring nothing, and a caching collapse makes them
        # look BETTER rather than worse. A warning printed afterwards arrives
        # once the reader has already believed the table (issue #15).
        *render_cache_warning(cache_findings(by_model)),
        # Per-backend deadlocks, which the pooled ruling card cannot show: it
        # groups by pair_id, so one judge's tie is absorbed into the slate's
        # majority. Deliberately quieter than the card's UNDECIDED marker,
        # which means the whole slate deadlocked (issue #23).
        *render_deadlock_section(backend_deadlocks(by_model, leg=leg)),
        # Reliability sits ABOVE every quality table, for the same reason the
        # cache warning does: it can disqualify a backend outright, and a reader
        # who meets kappa first has already formed a view of a judge that cannot
        # produce enough judgments to hold one (issue #43).
        *render_reliability_section(reliability_rows(by_model, manifests)),
        # And immediately before the quality tables, not in the caveats at the
        # bottom — "say so at the point its quality numbers are shown". A caveat
        # printed afterwards arrives once the reader has already believed the
        # table.
        *render_coverage_bias_caveat(coverage_shapes(by_model)),
        "## Stability (within-model, across repeated votes)",
        "",
        "Verdict and reason flips are tracked separately: a model can be perfectly",
        "stable on approve/reject while its reason code moves, and the reason code",
        "is what the per-reason confusion matrix will score once rulings exist.",
        "",
        *_stability_table(by_model),
        "",
        "## Cross-model verdict agreement",
        "",
        "Fraction of commonly-judged pairs where two backends returned the same",
        "verdict. Computed over the intersection, so a backend that errored on",
        "some pairs does not drag another's number down.",
        "",
        *_agreement_table(by_model),
        "",
        "## Reason-code distribution",
        "",
        "| Backend | Distribution |",
        "|---|---|",
    ]
    for model, verdicts in sorted(by_model.items()):
        dist = reason_distribution(verdicts)
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1]))
        lines.append(f"| {model} | {rendered} |")
    lines += _retired_code_note(by_model)

    cross_tabs = _reason_cross_tab_sections(by_model)
    if cross_tabs:
        lines += [
            "",
            "## Reason-code cross-tab (per backend pair)",
            "",
            "The distribution above shows what each backend reaches for; this shows",
            "where two backends diverge on the same item. Rows marked *divergence*",
            "are pairs both models judged but labelled differently — the disagreement",
            "the binary verdict hides.",
            "",
            *cross_tabs,
        ]

    lines += [
        "",
        "## Operational health",
        "",
        "Per-LAUNCH figures, not cumulative — see Caveats below before reading",
        "these as completion. The Reliability section above is the cumulative",
        "view; this one adds the per-launch detail it deliberately omits (min/max,",
        "failed-call latency, observed model snapshots).",
        "",
        *(_health_table(manifests) if manifests else ["_No manifests found._"]),
        "",
        *_caveats_section(by_model, manifests),
        "## Ruling queue (most contested first)",
        "",
        f"{len(contested)} of {len(ranking)} pairs are contested — models disagreed with",
        "each other, or a model disagreed with itself across votes. These are the",
        "highest-value rows to rule first: they are where the rubric is ambiguous,",
        "so each ruling resolves more calibration than a row everyone already agrees",
        "on. Uncontested pairs are not evidence of correctness — every model can be",
        "confidently wrong together.",
        "",
        "`Undecided` counts models whose OWN votes reached no majority on this pair.",
        "It weighs heavily: a judge that cannot decide is stronger evidence of an",
        "ambiguous rubric than two judges deciding differently.",
        "",
        "| Pair | Models | Undecided | Cross-model verdict | Cross-model reason | Verdict flip | Reason flip |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ranking[:queue_size]:
        lines.append(
            f"| {row.pair_id} | {row.n_models} | {row.n_unsettled} | "
            f"{row.cross_model_disagreement:.3f} | "
            f"{row.cross_model_reason_disagreement:.3f} | {row.mean_verdict_flip_rate:.3f} | "
            f"{row.mean_reason_flip_rate:.3f} |"
        )
    if len(ranking) > queue_size:
        lines.append("")
        lines.append(f"_Showing the top {queue_size} of {len(ranking)} pairs._")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="directory of verdict .jsonl files")
    parser.add_argument("--out", type=Path, default=Path("scenario-report.md"))
    parser.add_argument("--queue-size", type=int, default=40, help="rows in the ruling queue")
    args = parser.parse_args()

    by_model = load_results(args.results)
    if not by_model:
        parser.error(f"no verdict files found in {args.results}")
    manifests = load_manifests(args.results)
    # The leg is the results directory: scenario_report is invoked once per leg,
    # and the same backend runs in s1 and in each s2-* leg. Recording it keeps
    # those four populations apart in the audit trail (issue #23).
    leg = args.results.resolve().name
    args.out.write_text(
        render_scenario_report(by_model, manifests, queue_size=args.queue_size, leg=leg)
    )
    print(f"wrote {args.out} ({len(by_model)} backends)")

    # The audit trail, written BESIDE the report rather than into --results:
    # load_results globs *.jsonl there and would try to parse these records as
    # verdicts. Named from the report so two legs cannot overwrite each other.
    records = deadlock_records(backend_deadlocks(by_model, leg=leg))
    if records:
        audit_path = args.out.with_name(f"{args.out.stem}.deadlocks.jsonl")
        audit_path.write_text(records)
        print(f"wrote {audit_path} ({len(records.splitlines())} backend-level deadlocks)")


if __name__ == "__main__":
    main()
