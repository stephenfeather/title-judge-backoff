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
