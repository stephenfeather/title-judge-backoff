# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Score judge verdicts against operator ground truth and emit a leaderboard.

Usage:
    uv run score.py --data pairs.jsonl --results results/2026-08-05/ --out leaderboard.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from judge.schema import Pair, Verdict, pair_from_dict, verdict_from_json_line
from judge.stats import bootstrap_ci, intervals_overlap, mean_sd
from judge.vote import tally_votes

# Fixed so that re-scoring the same results reproduces the same intervals.
BOOTSTRAP_SEED = 20260806


@dataclass(frozen=True)
class ModelScore:
    model_id: str
    n: int  # pairs SCORED — judged, and their votes reached a verdict majority
    coverage: float  # fraction of ground-truth pairs this model actually judged
    # Pairs judged but excluded because their votes had no majority (#12).
    # Reported rather than dropped: a metric over 189 of 200 pairs is honest,
    # one over 200 where 11 were coin flips is not, and a silent exclusion is
    # its own kind of lie. Counted independently — a settled verdict with an
    # unsettled reason still scores for kappa, just not for reason confusion.
    n_unsettled_verdict: int
    n_unsettled_reason: int
    accuracy: float
    kappa: float  # on the majority verdict across votes
    false_approve_rate: float
    reason_confusion: dict[tuple[str, str], int]
    # Spread. n_votes is reported beside kappa_sd because sd=0.0 at n_votes=1
    # means "unmeasured", not "stable".
    n_votes: int
    kappa_run_mean: float  # mean of the per-run kappas
    kappa_sd: float  # sd of the per-run kappas — judge stochasticity
    kappa_ci: tuple[float, float]  # bootstrap CI over items, on majority verdicts
    # Stability. Tracked separately because effort/temperature can move the
    # reason code while leaving the binary verdict untouched.
    verdict_flip_rate: float
    reason_flip_rate: float
    unstable_pair_ids: list[str]  # any item whose votes disagreed at all


def cohens_kappa(truth: list[str], predicted: list[str]) -> float:
    """Cohen's kappa for two label sequences. 1.0 = perfect, 0.0 = chance."""
    n = len(truth)
    if n == 0:
        return 0.0
    observed = sum(t == p for t, p in zip(truth, predicted)) / n
    truth_counts = Counter(truth)
    pred_counts = Counter(predicted)
    expected = sum(
        (truth_counts[label] / n) * (pred_counts[label] / n)
        for label in truth_counts.keys() | pred_counts.keys()
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def _per_run_kappas(truth_by_id: dict[str, str], verdicts: list[Verdict]) -> list[float]:
    """Kappa computed independently for each vote index.

    Their spread IS the judge-stochasticity term: it answers "if we ran this
    again, how different would the number be?", which a single kappa cannot.

    Takes a plain id -> ground_truth map rather than Pairs so that unruled
    pairs cannot reach this function at all.
    """
    by_run: dict[int, list[Verdict]] = {}
    for v in verdicts:
        by_run.setdefault(v.run_index, []).append(v)
    return [
        cohens_kappa(
            [truth_by_id[v.pair_id] for v in run_verdicts],
            [v.verdict for v in run_verdicts],
        )
        for _, run_verdicts in sorted(by_run.items())
    ]


def score_model(pairs: list[Pair], verdicts: list[Verdict]) -> ModelScore:
    """Score one model against ground truth; unknown pair_ids are dropped.

    Votes for the same pair collapse to one majority ruling before scoring, so
    `n` counts pairs rather than API calls. With a single vote per pair this
    reduces to the original single-run behavior.
    """
    unruled = [p.id for p in pairs if not p.is_ruled]
    if unruled:
        raise ValueError(
            f"{len(unruled)} of {len(pairs)} pairs are unruled (first few: "
            f"{', '.join(unruled[:3])}). Kappa needs operator ground truth — None is "
            f"not a label. Use scenario_report.py for metrics that do not need "
            f"rulings (flip rates, cross-model agreement, reason distribution)."
        )

    by_id = {p.id: p for p in pairs}
    known = [v for v in verdicts if v.pair_id in by_id]
    if not known:
        raise ValueError("no verdicts matched any ground-truth pair")

    # Assert the one-model invariant instead of assuming it. `already_judged_ids`
    # refuses to RESUME a file whose rows carry a different model_id, but that
    # guards the writer — nothing stops a file being produced by concatenating
    # two backends' shards (a merge tool, a `cat` of two files, a recovery
    # script stitching a partial run back together). Such a file used to score
    # cleanly and be labelled with whichever row happened to sort first, which
    # is the reader-side counterpart of the provenance gap in issue #13.
    #
    # Every other ordering dependency here was removed deliberately (#10) so a
    # reported number cannot depend on the order concurrent workers wrote rows.
    # This was the last one.
    # The fields checked are exactly the run config `already_judged_ids` refuses
    # to resume across, because that is what "one run" means. prompt_version is
    # live rather than hypothetical since #14: results/ holds v1 files and new
    # runs write v2, so a stitched file mixing them is a real possibility, and
    # averaging two prompts averages two instruments.
    # Two groups, because `None` means OPPOSITE things in them.
    #
    # Config fields: None is a real value. temperature=None means the field was
    # omitted from the request, which already_judged_ids treats as a different
    # config from temperature=0.0 — skipping it as "unknown" would erase a
    # distinction the writer guard deliberately enforces.
    #
    # Provenance fields: None means "unknown, do not compare" (#13). Every row
    # written before that issue lacks all three, and a run resumed across the
    # boundary legitimately holds both kinds, so only rows that DO know are
    # compared. Shards can otherwise agree on model, prompt, temperature and
    # effort while differing in host or code — and config_digest is the only
    # field that distinguishes `api` and `structured_output` at all.
    mixed = {}
    for field in ("model_id", "prompt_version", "temperature", "reasoning_effort"):
        values = sorted({str(getattr(v, field)) for v in known})
        if len(values) > 1:
            mixed[field] = values
    for field in ("base_url", "config_digest", "code_version"):
        values = sorted({getattr(v, field) for v in known if getattr(v, field) is not None})
        if len(values) > 1:
            mixed[field] = values
    if mixed:
        detail = "; ".join(f"{f}: {', '.join(repr(v) for v in vs)}" for f, vs in mixed.items())
        hint = (
            " A config_digest difference is one no other column names — compare the "
            "`api` and `structured_output` settings in each run's manifest."
            if "config_digest" in mixed
            else ""
        )
        raise ValueError(
            f"results hold verdicts from more than one run config ({detail}), so there is "
            f"no single model to score. This file was most likely produced by "
            f"concatenating separate runs. Score each run's results file on its own.{hint}"
        )
    model_ids = sorted({v.model_id for v in known})

    # Ordered by the PAIRS file, not by first appearance in the results file.
    # bootstrap_ci resamples this sequence from a fixed seed, so ordering it by
    # the results file would make the reported CI depend on the order verdicts
    # happened to be written — which concurrent workers make arbitrary. The
    # pairs file is the same on every run, so this is stable by construction.
    ruled = {r.pair_id: r for r in tally_votes(known)}
    judged = [(pair, ruled[pair.id]) for pair in pairs if pair.id in ruled]

    # A pair whose votes had no majority was never decided by the model — the
    # tie-break decided it. Scoring it means scoring the tie-break, so it is
    # excluded and the exclusion is COUNTED (see #12). Verdict and reason are
    # excluded independently: the common shape is a settled verdict with an
    # unsettled reason, and the verdict there is real evidence.
    #
    # false_approve_rate is the sharpest case. It is a safety number, and a
    # coin flip recorded as an approve corrupts exactly the metric that governs
    # deployment risk.
    # `is not None` rather than the `settled` properties: identical meaning,
    # but it lets a type checker narrow away the Optional for everything below.
    matched = [(p, r) for p, r in judged if r.verdict is not None]
    n_unsettled_verdict = len(judged) - len(matched)
    n_unsettled_reason = sum(1 for _, r in judged if r.reason is None)
    if not matched:
        raise ValueError(
            f"no pair has a settled verdict: all {len(judged)} judged pairs had "
            f"votes with no majority, so there is nothing to score. Their flip "
            f"rates are still meaningful — see the scenario report."
        )

    # Narrowed once, here: the unruled guard above means every pair carries a
    # ruling, so the rest of this function works with plain strings.
    truth_by_id = {p.id: p.ground_truth for p in pairs if p.ground_truth is not None}
    reason_by_id = {p.id: p.reason.value for p in pairs if p.reason is not None}

    truth = [truth_by_id[p.id] for p, _ in matched]
    predicted = [r.verdict for _, r in matched if r.verdict is not None]
    n = len(matched)

    rejects = [(p, r) for p, r in matched if truth_by_id[p.id] == "reject"]
    false_approves = sum(r.verdict == "approve" for _, r in rejects)

    confusion = Counter(
        (reason_by_id[p.id], r.reason.value) for p, r in judged if r.reason is not None
    )

    run_kappas = _per_run_kappas(truth_by_id, known)
    kappa_mean, kappa_sd = mean_sd(run_kappas)

    # Resample whole items, matching how the metrics aggregate.
    labelled = list(zip(truth, predicted))
    lo, hi = bootstrap_ci(
        labelled,
        statistic=lambda sample: cohens_kappa([t for t, _ in sample], [p for _, p in sample]),
        seed=BOOTSTRAP_SEED,
    )

    return ModelScore(
        # From the asserted set, not from whichever row sorted first: the
        # ordering dependency is removed rather than merely guarded.
        model_id=model_ids[0],
        n=n,
        # Coverage answers "did this backend judge the whole set", which is a
        # different question from "did its votes decide". Computed over JUDGED
        # pairs so a contested pair does not trip the leaderboard's
        # partial-coverage gate and silently drop the model from the ranking.
        coverage=len(judged) / len(pairs),
        n_unsettled_verdict=n_unsettled_verdict,
        n_unsettled_reason=n_unsettled_reason,
        accuracy=sum(t == pr for t, pr in zip(truth, predicted)) / n,
        kappa=cohens_kappa(truth, predicted),
        false_approve_rate=false_approves / len(rejects) if rejects else 0.0,
        reason_confusion=dict(confusion),
        # The number of RUNS, not the max votes any one pair received. This sits
        # beside kappa_sd, which is the spread of the per-run kappas, so the two
        # must describe the same set — and a pair that lost a vote to an error
        # must not make the whole column under-report the runs that happened.
        n_votes=len(run_kappas),
        kappa_run_mean=kappa_mean,
        kappa_sd=kappa_sd,
        kappa_ci=(lo, hi),
        verdict_flip_rate=sum(r.verdict_flip_rate for _, r in matched) / n,
        reason_flip_rate=sum(r.reason_flip_rate for _, r in matched) / n,
        unstable_pair_ids=[
            r.pair_id for _, r in matched if r.verdict_flip_rate or r.reason_flip_rate
        ],
    )


def separability_tiers(scores: list[ModelScore]) -> list[list[ModelScore]]:
    """Group models whose kappa intervals overlap into joint, unranked tiers.

    Tiers are the CONNECTED COMPONENTS of the overlap graph, which guarantees
    the invariant the report depends on: any two models in different tiers have
    non-overlapping intervals, so the ordering between tiers is real.

    Comparing each model only against the most recently opened tier is not
    enough — overlap chains. With A=[0.80,0.90], B=[0.60,0.70], C=[0.50,0.85]:
    A and B are disjoint so B opens a new tier, C overlaps B and joins it, and
    A and C land in different tiers while actually overlapping. The report would
    then claim A beats C on evidence that does not separate them.
    """
    ordered = sorted(scores, key=lambda s: s.kappa, reverse=True)
    # Union-find over "intervals overlap", then read components out in kappa order.
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if intervals_overlap(ordered[i].kappa_ci, ordered[j].kappa_ci):
                parent[find(j)] = find(i)

    tiers: dict[int, list[ModelScore]] = {}
    for index, score in enumerate(ordered):
        tiers.setdefault(find(index), []).append(score)
    # dict preserves insertion order, which follows `ordered` — best tier first.
    return list(tiers.values())


def render_leaderboard(scores: list[ModelScore]) -> str:
    """Markdown leaderboard grouped by separability, best tier first.

    Only models with 100% coverage are ranked: metrics computed on a subset of
    pairs are not comparable (a backend that errored on most pairs could post
    perfect kappa on the few it answered). Incomplete models are listed
    separately — re-run the bake-off to fill their gaps.

    Within a tier, models are NOT ordered: their kappa intervals overlap, so
    any ordering would report sampling noise as a finding.
    """
    ranked = [s for s in scores if s.coverage == 1.0]
    excluded = [s for s in scores if s.coverage < 1.0]
    lines = [
        "# Judge bake-off leaderboard",
        "",
        "Kappa is computed on the majority verdict across votes. `Kappa sd` is the",
        "spread of per-run kappas — judge stochasticity, not sampling error — and is",
        "0.000 whenever votes = 1, meaning unmeasured rather than stable. Models whose",
        "95% intervals overlap share a tier and are deliberately left unordered.",
        "",
        "`n` counts pairs actually SCORED. A pair whose votes had no majority was",
        "decided by nothing — the tie-break would have invented the answer — so it is",
        "excluded and counted under `Unsettled` instead. Verdict and reason are",
        "excluded independently: a settled verdict still scores for kappa even when",
        "its reason did not settle.",
        "",
        "| Tier | Model | n | Unsettled (verdict/reason) | Votes | Coverage | Accuracy | Kappa | Kappa sd | Kappa 95% CI | Verdict flip | Reason flip | False-approve rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    tiers = separability_tiers(ranked)
    for tier_no, tier in enumerate(tiers, 1):
        label = f"{tier_no}" if len(tier) == 1 else f"{tier_no} (not separable)"
        for s in tier:
            lo, hi = s.kappa_ci
            lines.append(
                f"| {label} | {s.model_id} | {s.n} | "
                f"{s.n_unsettled_verdict}/{s.n_unsettled_reason} | "
                f"{s.n_votes} | {s.coverage:.0%} | "
                f"{s.accuracy:.3f} | {s.kappa:.3f} | {s.kappa_sd:.3f} | "
                f"[{lo:.3f}, {hi:.3f}] | {s.verdict_flip_rate:.3f} | "
                f"{s.reason_flip_rate:.3f} | {s.false_approve_rate:.3f} |"
            )
    lines.append("")
    multi = [t for t in tiers if len(t) > 1]
    if multi:
        lines.append(
            "Tiers marked *not separable* contain models whose kappa intervals overlap; "
            "picking between them needs more calibration pairs or more votes, not a "
            "closer look at the ranking."
        )
        lines.append("")
    unstable = [s for s in scores if s.unstable_pair_ids]
    if unstable:
        lines.append("## Unstable pairs (votes disagreed)")
        lines.append("")
        lines.append(
            "These are the borderline items — they carry most of the run-to-run "
            "instability and are the best candidates for sharpening the rubric. A "
            "rising count across runs is an early sign the provider changed the model."
        )
        lines.append("")
        for s in unstable:
            ids = ", ".join(s.unstable_pair_ids)
            lines.append(f"- {s.model_id} ({len(s.unstable_pair_ids)} of {s.n}): {ids}")
        lines.append("")
    if excluded:
        lines.append("## Excluded from ranking (incomplete coverage)")
        lines.append("")
        lines.append("Metrics on partial coverage are not comparable; re-run to fill gaps.")
        lines.append("")
        for s in sorted(excluded, key=lambda s: s.coverage, reverse=True):
            lines.append(f"- {s.model_id}: coverage {s.coverage:.0%} ({s.n} pairs judged)")
        lines.append("")
    for s in sorted(scores, key=lambda s: s.kappa, reverse=True):
        lines.append(f"## Reason confusion: {s.model_id}")
        lines.append("")
        lines.append("| Ground truth | Judged | Count |")
        lines.append("|---|---|---|")
        for (gt, judged), count in sorted(s.reason_confusion.items()):
            lines.append(f"| {gt} | {judged} | {count} |")
        lines.append("")
    return "\n".join(lines)


def load_pairs(path: Path) -> list[Pair]:
    return [pair_from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def load_verdicts(path: Path) -> list[Verdict]:
    return [verdict_from_json_line(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="ground-truth pairs.jsonl")
    parser.add_argument("--results", type=Path, required=True, help="directory of per-backend verdict .jsonl files")
    parser.add_argument("--out", type=Path, default=Path("leaderboard.md"))
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    scores = [score_model(pairs, load_verdicts(f)) for f in sorted(args.results.glob("*.jsonl"))]
    args.out.write_text(render_leaderboard(scores))
    print(f"wrote {args.out} ({len(scores)} models)")


if __name__ == "__main__":
    main()
