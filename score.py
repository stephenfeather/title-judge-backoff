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
    n: int
    coverage: float  # fraction of ground-truth pairs this model actually judged
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


def _per_run_kappas(pairs_by_id: dict[str, Pair], verdicts: list[Verdict]) -> list[float]:
    """Kappa computed independently for each vote index.

    Their spread IS the judge-stochasticity term: it answers "if we ran this
    again, how different would the number be?", which a single kappa cannot.
    """
    by_run: dict[int, list[Verdict]] = {}
    for v in verdicts:
        by_run.setdefault(v.run_index, []).append(v)
    return [
        cohens_kappa(
            [pairs_by_id[v.pair_id].ground_truth for v in run_verdicts],
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
    by_id = {p.id: p for p in pairs}
    known = [v for v in verdicts if v.pair_id in by_id]
    if not known:
        raise ValueError("no verdicts matched any ground-truth pair")

    voted = tally_votes(known)
    matched = [(by_id[r.pair_id], r) for r in voted]

    truth = [p.ground_truth for p, _ in matched]
    predicted = [r.verdict for _, r in matched]
    n = len(matched)

    rejects = [(p, r) for p, r in matched if p.ground_truth == "reject"]
    false_approves = sum(r.verdict == "approve" for _, r in rejects)

    confusion = Counter((p.reason.value, r.reason.value) for p, r in matched)

    run_kappas = _per_run_kappas(by_id, known)
    kappa_mean, kappa_sd = mean_sd(run_kappas)

    # Resample whole items, matching how the metrics aggregate.
    labelled = list(zip(truth, predicted))
    lo, hi = bootstrap_ci(
        labelled,
        statistic=lambda sample: cohens_kappa([t for t, _ in sample], [p for _, p in sample]),
        seed=BOOTSTRAP_SEED,
    )

    return ModelScore(
        model_id=known[0].model_id,
        n=n,
        coverage=n / len(pairs),
        accuracy=sum(t == pr for t, pr in zip(truth, predicted)) / n,
        kappa=cohens_kappa(truth, predicted),
        false_approve_rate=false_approves / len(rejects) if rejects else 0.0,
        reason_confusion=dict(confusion),
        n_votes=max(r.n_votes for _, r in matched),
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

    Walking down by kappa, a model joins the current tier if its interval
    overlaps ANY member's. Two models in one tier are not distinguishable at
    this sample size, so the report must not imply an order between them.
    """
    tiers: list[list[ModelScore]] = []
    for s in sorted(scores, key=lambda s: s.kappa, reverse=True):
        if tiers and any(intervals_overlap(s.kappa_ci, other.kappa_ci) for other in tiers[-1]):
            tiers[-1].append(s)
        else:
            tiers.append([s])
    return tiers


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
        "| Tier | Model | n | Votes | Coverage | Accuracy | Kappa | Kappa sd | Kappa 95% CI | Verdict flip | Reason flip | False-approve rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    tiers = separability_tiers(ranked)
    for tier_no, tier in enumerate(tiers, 1):
        label = f"{tier_no}" if len(tier) == 1 else f"{tier_no} (not separable)"
        for s in tier:
            lo, hi = s.kappa_ci
            lines.append(
                f"| {label} | {s.model_id} | {s.n} | {s.n_votes} | {s.coverage:.0%} | "
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
