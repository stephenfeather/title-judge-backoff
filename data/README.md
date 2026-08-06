# Calibration data

Calibration pairs are **not committed** to this repo (they contain
vendor-derived titles). Supply them at run time via `--data /path/to/pairs.jsonl`.

## pairs.jsonl schema

One JSON object per line:

```json
{
  "id": "p0001",
  "original": "acme widget 3000 blk",
  "enriched": "Acme Widget 3000, Black",
  "brand": "Acme",
  "mpn": "W3000-BLK",
  "ground_truth": "approve",
  "reason": "ok"
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | Unique pair id; used for resume bookkeeping |
| `original` | string | Title before enrichment |
| `enriched` | string | Title after enrichment |
| `brand` | string | Known brand for the product |
| `mpn` | string | Manufacturer part number |
| `ground_truth` | `"approve"` \| `"reject"` | Operator ruling |
| `reason` | enum | `overcorrection`, `meaning_change`, `casing_error`, `truncation_worse`, `ok` |

`reason` should be `ok` when `ground_truth` is `approve`, and one of the
failure codes when it is `reject`.

`brand` and `mpn` are **optional** — the judge prompt omits those lines when
they are absent. The E10 QA pack carries neither.

## Building pairs.jsonl from the E10 QA pack

The pack (`infrastructure/enrichment/qa/title-qa-pack.md` in
featherarms-infrastructure) holds the 200 stratified before/after changes but
**no operator verdicts** — it is the proposal document, not the ruling. So
there is a human step in the middle:

```sh
# 1. Parse the pack into a ruling template (verdicts left blank)
uv run adapt_qa_pack.py template --pack /path/to/title-qa-pack.md --out rulings-template.jsonl

# 2. Operator fills in ground_truth (approve|reject) and reason on all 200 rows

# 3. Build the calibration set
uv run adapt_qa_pack.py merge --template rulings-template.jsonl --out pairs.jsonl
```

`merge` refuses to emit a partial calibration set: it fails if any row is
unruled, or if a ruling references an id no row has.

**Row ids are content-derived** (`e10-<sha256 of before+after>`), not
positional — so regenerating the pack keeps ids stable for unchanged rows and
previously collected rulings still apply.

`--attributes` optionally joins a `{id, brand, mpn}` JSONL if those fields
become available from another source; without it they are omitted entirely.

### Note on the `unchanged` cohort

20 of the 200 rows are no-ops — `original` and `enriched` are byte-identical
(the pipeline skipped titles already containing a lowercase character). Asking
a judge to approve or reject a change that was never made is a degenerate
case; decide deliberately whether those rows belong in the calibration set or
should be scored separately.
