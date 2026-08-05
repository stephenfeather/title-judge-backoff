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
import sys
from pathlib import Path

from judge.client import Backend, JudgeClient, RateLimiter, load_backends
from judge.schema import Pair, pair_from_dict, verdict_from_json_line, verdict_to_json_line


def already_judged_ids(results_path: Path) -> set[str]:
    """Pair ids already present in a backend's results file (empty if absent)."""
    if not results_path.exists():
        return set()
    return {
        verdict_from_json_line(line).pair_id
        for line in results_path.read_text().splitlines()
        if line.strip()
    }


def pending_pairs(pairs: list[Pair], judged: set[str]) -> list[Pair]:
    return [p for p in pairs if p.id not in judged]


def load_pairs(path: Path) -> list[Pair]:
    return [pair_from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def run_backend(backend: Backend, pairs: list[Pair], out_dir: Path) -> None:
    results_path = out_dir / f"{backend.name}.jsonl"
    todo = pending_pairs(pairs, already_judged_ids(results_path))
    label = " (eval-only backend)" if backend.eval_only else ""
    print(f"[{backend.name}]{label} {len(todo)} pending of {len(pairs)} pairs")
    if not todo:
        return

    client = JudgeClient(backend)
    limiter = RateLimiter(rpm=backend.rpm)
    try:
        with open(results_path, "a") as fh:
            for i, pair in enumerate(todo, 1):
                limiter.wait()
                try:
                    verdict = client.judge(pair)
                except Exception as exc:  # noqa: BLE001 - keep going, resume covers gaps
                    print(f"[{backend.name}] {pair.id}: ERROR {exc}", file=sys.stderr)
                    continue
                fh.write(verdict_to_json_line(verdict) + "\n")
                fh.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"[{backend.name}] {i}/{len(todo)}")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", type=Path, default=Path("backends.toml"))
    parser.add_argument("--data", type=Path, required=True, help="calibration pairs.jsonl (never committed)")
    parser.add_argument("--out", type=Path, required=True, help="output directory, e.g. results/2026-08-05/")
    args = parser.parse_args()

    pairs = load_pairs(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    for backend in load_backends(args.backends):
        run_backend(backend, pairs, args.out)


if __name__ == "__main__":
    main()
