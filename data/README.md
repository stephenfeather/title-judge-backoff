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

# 2. Rule the template — one keystroke per row, resumable
uv run rule.py run --template rulings-template.jsonl --journal rulings.journal.jsonl

# 3. Export the rulings and build the calibration set
uv run rule.py export --journal rulings.journal.jsonl --out rulings.jsonl
uv run adapt_qa_pack.py merge --template rulings-template.jsonl \
    --rulings rulings.jsonl --out pairs.jsonl
```

Step 2 can also be done by hand — filling `ground_truth` and `reason` into the
template and running `merge --template` alone still works. `rule.py` just makes
200 rows survivable; see [the ruling pass](#the-ruling-pass) below.

`merge` refuses to emit a partial calibration set: it fails if any row is
unruled, or if a ruling references an id no row has.

**Row ids are content-derived** (`e10-<sha256 of before+after>`), not
positional — so regenerating the pack keeps ids stable for unchanged rows and
previously collected rulings still apply.

`--attributes` optionally joins a `{id, brand, mpn}` JSONL if those fields
become available from another source; without it they are omitted entirely.

## The ruling pass

`rule.py` shows one change per screen and takes a single keystroke for the
verdict:

| Key | Action |
|---|---|
| `a` | approve (reason `ok`) |
| `r` | reject → one-key reason submenu, generated from `judge/schema.py` |
| ⏎ | on an `unchanged` row only: accept the pre-offered approve |
| `s` | skip — stays pending, re-presented on the next run |
| `u` | undo the last action of this session |
| `n` | annotate this row (with `--notes`) |
| `q` | quit and save |

Every keystroke appends one line to the journal and fsyncs, so a crash loses at
most the decision in flight. Re-running the same command resumes: ruled rows
are dropped, skipped rows come back. An `undo` is itself journaled and replayed
away — nothing is ever rewritten.

`--results results/<dir>` adds a context pane per row: how a sweep's judges
split on that pair (`flip 50% of 6 judge(s)`) and which reason codes they gave.
It reads flat and per-scenario layouts, and without it the pane is simply
absent — the ruling pass never depends on a sweep having run.

`--notes` enables free-text rubric annotations, exported to a sidecar with
`--notes-out` and deliberately kept out of the merge input.

**The journal is operator-local.** It carries vendor-derived title ids and the
operator's reasoning; `*.jsonl` is gitignored and none of it belongs in the repo.

### Note on the `unchanged` cohort

20 of the 200 rows are no-ops — `original` and `enriched` are byte-identical
(the pipeline skipped titles already containing a lowercase character). Asking
a judge to approve or reject a change that was never made is a degenerate
case; decide deliberately whether those rows belong in the calibration set or
should be scored separately.

`rule.py` flags such a row and pre-offers approve/`ok` on ⏎ so the sanity cohort
costs one keystroke, but it never rules one on the operator's behalf: `a`/`r`
still override, and an accepted offer is journaled with `"auto": true` so those
rows can be found and scored separately later. Whether a row is "unchanged" is
decided by comparing the titles, not by trusting the cohort label.
