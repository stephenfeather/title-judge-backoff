# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Turn the E10 operator QA pack into a calibration set.

The pack carries the 200 before/after title changes but no operator verdicts,
so this runs in two passes with a human step in between:

    # 1. emit a ruling template (verdicts blank) for the operator to fill in
    uv run adapt_qa_pack.py template --pack /path/to/title-qa-pack.md --out rulings-template.jsonl

    # 2. after the ground_truth/reason fields are filled in, build pairs.jsonl
    uv run adapt_qa_pack.py merge --template rulings-template.jsonl --out pairs.jsonl

`merge` accepts the filled-in template directly (rows carrying ground_truth and
reason), or a separate --rulings file keyed by pair id. Neither output belongs
in this repo: both contain vendor-derived titles and *.jsonl is gitignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from judge.qa_pack import merge_rulings, parse_qa_pack, row_id, template_line


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_template(args: argparse.Namespace) -> None:
    rows = parse_qa_pack(args.pack.read_text())
    attributes = {}
    if args.attributes:
        attributes = {rec["id"]: rec for rec in load_jsonl(args.attributes)}

    lines = [template_line(row, attributes.get(row_id(row))) for row in rows]
    args.out.write_text("\n".join(lines) + "\n")

    cohorts: dict[str, int] = {}
    for row in rows:
        cohorts[row.cohort] = cohorts.get(row.cohort, 0) + 1
    print(f"wrote {args.out} ({len(rows)} rows)")
    for cohort, count in sorted(cohorts.items()):
        print(f"  {cohort}: {count}")
    print("\nFill in ground_truth (approve|reject) and reason on every row, then run `merge`.")


def cmd_merge(args: argparse.Namespace) -> None:
    template = load_jsonl(args.template)

    if args.rulings:
        rulings = {rec["id"]: rec for rec in load_jsonl(args.rulings)}
    else:
        rulings = {
            rec["id"]: rec
            for rec in template
            if rec.get("ground_truth") is not None and rec.get("reason") is not None
        }

    pairs = merge_rulings(template, rulings)
    with open(args.out, "w") as fh:
        for pair in pairs:
            record = {
                "id": pair.id,
                "original": pair.original,
                "enriched": pair.enriched,
                "ground_truth": pair.ground_truth,
                "reason": pair.reason.value,
            }
            if pair.brand:
                record["brand"] = pair.brand
            if pair.mpn:
                record["mpn"] = pair.mpn
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(pairs)} ruled pairs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("template", help="parse the pack and emit a blank ruling template")
    t.add_argument("--pack", type=Path, required=True, help="title-qa-pack.md")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument(
        "--attributes",
        type=Path,
        help="optional JSONL of {id, brand, mpn} to join; omitted entirely when absent",
    )
    t.set_defaults(func=cmd_template)

    m = sub.add_parser("merge", help="combine the template with rulings into pairs.jsonl")
    m.add_argument("--template", type=Path, required=True)
    m.add_argument("--rulings", type=Path, help="optional separate rulings JSONL keyed by id")
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
