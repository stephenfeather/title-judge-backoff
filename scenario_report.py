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
from dataclasses import dataclass
from pathlib import Path

from judge.agreement import agreement_matrix, reason_distribution
from judge.schema import Verdict, verdict_from_json_line
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

    @property
    def contention(self) -> float:
        """Single sortable score. Cross-model disagreement dominates, since two
        models disagreeing is stronger evidence of an ambiguous rubric than one
        model wavering between its own votes."""
        return (
            2 * self.cross_model_disagreement
            + self.cross_model_reason_disagreement
            + self.mean_verdict_flip_rate
            + 0.5 * self.mean_reason_flip_rate
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
        rows.append(
            ContentionRow(
                pair_id=pid,
                n_models=len(present),
                cross_model_disagreement=_disagreement([r.verdict for r in present]),
                cross_model_reason_disagreement=_disagreement([r.reason.value for r in present]),
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
        unstable = sum(1 for r in voted if r.verdict_flip_rate or r.reason_flip_rate)
        lines.append(
            f"| {model} | {n} | {max((r.n_votes for r in voted), default=0)} | "
            f"{sum(r.verdict_flip_rate for r in voted) / n:.3f} | "
            f"{sum(r.reason_flip_rate for r in voted) / n:.3f} | {unstable} |"
        )
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


def _health_table(manifests: dict[str, dict]) -> list[str]:
    lines = [
        "| Backend | Effort | Temp | OK | Failed | Latency min/median/max (s) | Errors | Snapshots |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in sorted(manifests.items()):
        h = m.get("health", {})
        def fmt(key):
            value = h.get(key)
            return "-" if value is None else f"{value:.1f}"
        errors = h.get("error_kinds") or {}
        error_text = ", ".join(f"{k}x{v}" for k, v in sorted(errors.items())) or "none"
        lines.append(
            f"| {name} | {m.get('reasoning_effort') or 'default'} | "
            f"{'omitted' if m.get('temperature') is None else m['temperature']} | "
            f"{h.get('calls_ok', 0)} | {h.get('calls_failed', 0)} | "
            f"{fmt('latency_min')} / {fmt('latency_median')} / {fmt('latency_max')} | "
            f"{error_text} | {', '.join(m.get('observed_models') or []) or '-'} |"
        )
    return lines


def render_scenario_report(
    by_model: dict[str, list[Verdict]], manifests: dict[str, dict], *, queue_size: int = 40
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

    lines += [
        "",
        "## Operational health",
        "",
        *(_health_table(manifests) if manifests else ["_No manifests found._"]),
        "",
        "## Ruling queue (most contested first)",
        "",
        f"{len(contested)} of {len(ranking)} pairs are contested — models disagreed with",
        "each other, or a model disagreed with itself across votes. These are the",
        "highest-value rows to rule first: they are where the rubric is ambiguous,",
        "so each ruling resolves more calibration than a row everyone already agrees",
        "on. Uncontested pairs are not evidence of correctness — every model can be",
        "confidently wrong together.",
        "",
        "| Pair | Models | Cross-model verdict | Cross-model reason | Verdict flip | Reason flip |",
        "|---|---|---|---|---|---|",
    ]
    for row in ranking[:queue_size]:
        lines.append(
            f"| {row.pair_id} | {row.n_models} | {row.cross_model_disagreement:.3f} | "
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
    args.out.write_text(render_scenario_report(by_model, manifests, queue_size=args.queue_size))
    print(f"wrote {args.out} ({len(by_model)} backends)")


if __name__ == "__main__":
    main()
