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


@dataclass(frozen=True)
class ModelScore:
    model_id: str
    n: int
    coverage: float  # fraction of ground-truth pairs this model actually judged
    accuracy: float
    kappa: float
    false_approve_rate: float
    reason_confusion: dict[tuple[str, str], int]


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


def score_model(pairs: list[Pair], verdicts: list[Verdict]) -> ModelScore:
    """Score one model's verdicts against ground truth; unknown pair_ids are dropped."""
    by_id = {p.id: p for p in pairs}
    matched = [(by_id[v.pair_id], v) for v in verdicts if v.pair_id in by_id]
    if not matched:
        raise ValueError("no verdicts matched any ground-truth pair")

    truth = [p.ground_truth for p, _ in matched]
    predicted = [v.verdict for _, v in matched]
    n = len(matched)

    rejects = [(p, v) for p, v in matched if p.ground_truth == "reject"]
    false_approves = sum(v.verdict == "approve" for _, v in rejects)

    confusion = Counter((p.reason.value, v.reason.value) for p, v in matched)

    return ModelScore(
        model_id=matched[0][1].model_id,
        n=n,
        coverage=n / len(pairs),
        accuracy=sum(t == pr for t, pr in zip(truth, predicted)) / n,
        kappa=cohens_kappa(truth, predicted),
        false_approve_rate=false_approves / len(rejects) if rejects else 0.0,
        reason_confusion=dict(confusion),
    )


def render_leaderboard(scores: list[ModelScore]) -> str:
    """Markdown leaderboard sorted by Cohen's kappa, best first.

    Only models with 100% coverage are ranked: metrics computed on a subset of
    pairs are not comparable (a backend that errored on most pairs could post
    perfect kappa on the few it answered). Incomplete models are listed
    separately — re-run the bake-off to fill their gaps.
    """
    ranked = [s for s in scores if s.coverage == 1.0]
    excluded = [s for s in scores if s.coverage < 1.0]
    lines = [
        "# Judge bake-off leaderboard",
        "",
        "| Model | n | Coverage | Accuracy | Kappa | False-approve rate |",
        "|---|---|---|---|---|---|",
    ]
    for s in sorted(ranked, key=lambda s: s.kappa, reverse=True):
        lines.append(
            f"| {s.model_id} | {s.n} | {s.coverage:.0%} | {s.accuracy:.3f} | {s.kappa:.3f} | {s.false_approve_rate:.3f} |"
        )
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
