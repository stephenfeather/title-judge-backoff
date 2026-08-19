# The "importable" contract — scoping proposal

**Date:** 2026-08-18
**Status:** proposal for Stephen to rule on. No code written, nothing committed.
**Measured against:** `~/Development/inventory-feeds-management/output/` — the extractor
write-side, files dated 2026-08-17 23:46 – 2026-08-18 00:17. `Extract/*/output/` and
`Transform/output/` were not used.

Scripts (scratchpad, throwaway): `importable.py`, `cssi_grain.py`, `raw_overlap.py`.

---

## 0. CSSI's image fan-out — real in the raw feed, ALREADY SOLVED in the transform

> **CORRECTION (2026-08-18, same day).** This section originally called the CSSI image
> fan-out a *blocking prerequisite*. That was wrong and is retracted. The CSSI
> transformer already collapses it correctly: `Transform/output/cssi.csv` is 48,102 rows,
> **one row per `vendor_sku`, zero duplicates**, with `images` as a JSON array (12,978
> rows carry more than one image). Nothing needs to be built here.
>
> What remains true: the fan-out is a property of the **raw** feed, so any tooling that
> reads `output/product_feed.csv` directly — including my own measurement scripts, and
> any future importable-gate that skips the transform — must collapse by SKU or it will
> double-count. The transform is the thing that knows this; work that bypasses the
> transform inherits the problem.
>
> This is a worked example of the real failure mode: the knowledge was **in the code and
> absent from the record**, so measuring raw feeds rediscovered it as if it were new.

**CSSI's raw feed is image-denormalized. It ships one row per image, not per product.**

✓ VERIFIED, not inferred: grouping `product_feed.csv` by its own `SKU` column gives
47,971 distinct SKUs across 84,380 rows. Of the 15,568 SKUs with more than one row,
**15,568 — 100.0% — vary in `Image Location` and in no other column whatsoever.**
Mean 1.76 images per SKU, max 27.

Consequences:

- CSSI's true product count is **47,971**, not 84,380. Any row-count reasoning about
  CSSI is inflated ~1.76×.
- The importable gate must collapse CSSI to product grain and fold `Image Location`
  into a gallery list, or it will emit ~36,000 duplicate products.
- This is a **silent** defect of the same family as the others: nothing errors, and a
  row-count check looks healthy.

All per-vendor figures below are stated at **product grain** (CSSI by SKU).

---

## 1. Proposed floor — four fields

Deliberately small and testable, per the "minimal contract that ships" constraint.
A record is **importable** when it has all four:

| # | Field | Why it's floor, not enrichment |
|---|-------|-------------------------------|
| 1 | `vendor_sku` | The vendor's own row identifier. Without it a record cannot be traced back, re-fetched, or de-duplicated against the next pull. |
| 2 | `name` | Akeneo will accept an empty label, but the result is a junk record a human must fix. A non-empty title is the difference between an imported product and a placeholder. |
| 3 | `brand` | Required by identity (see §3): the fallback key is `(brand, mpn)`, so brand is load-bearing, not cosmetic. |
| 4 | **identity** — a valid UPC **or** a `(brand, mpn)` pair | See §3. This is the only composite rule and the only one that rejects anything meaningful. |

**Explicitly NOT in the floor** — all of these are enrichment and, per the ruling, must
never gate import: MPN on its own, UPC on its own, category, images, description,
specifications, dimensions, weight.

**Also not in the floor: price, cost, MAP, stock.** These are commerce state and belong
on the *offer*, not the canonical product — the merge stage already treats them that way
(`COMMERCE_FIELDS` in `policy.py`, described there as "offers only, never canonical,
test-enforced invariant"). A product with no price is still importable; it just has no
sellable offer. Conflating the two would make price outages block catalog import.

---

## 2. Measured coverage — tonight's data

Field presence, per vendor, product grain:

| vendor | rows/products | vendor_sku | name | brand | mpn | upc |
|---|---|---|---|---|---|---|
| davidsons | 10,550 | 100.0% | 100.0% | 100.0% | **n/a — no column** | 99.4% |
| zanders | 35,086 | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% |
| cssi | 47,971 SKUs | 100.0% | 100.0% | 100.0% | 100.0% | 97.1% |
| rsrgroup | 30,376 | 100.0% | 100.0% | 99.1% | 100.0% | 98.0% |
| pawholesale | 174 | 100.0% | 100.0% | 100.0% | 100.0% | **0.0%** |
| **overall** | **124,157** | **100.0%** | **100.0%** | **99.8%** | 93.4% | 98.1% |

**The floor costs almost nothing.** Fields 1–3 are at or near 100% everywhere; the worst
single cell is RSR brand at 99.1%. This is the intended result — a floor that rejected
10% of the catalog would be a scoping decision masquerading as a validation rule.

Two structural gaps that are properties of the vendors, not defects:

- **Davidsons has no MPN column at all.** Header is `Item #, Item Description, MSP,
  Retail Price, Dealer Price, Sale Price, Sale Ends, Quantity, UPC Code, Manufacturer,
  Gun Type, Model Series, Caliber, Action, Capacity, Finish, Stock, Sights, Barrel
  Length, Overall Length, Features`. It is UPC-only, permanently.
- **PA Wholesale has no UPC at all** — 174/174 blank, verified by reading the column
  (digit-length histogram `{0: 174}`). It is MPN-only, permanently.

So **identity capability is per-vendor, not global.** Any rule phrased "match on UPC,
else MPN" must know which vendors can answer which question, or Davidsons will appear to
fail an MPN join it was never able to attempt.

---

## 3. Identity

### Primary key: normalized UPC. Fallback: `(brand, mpn)`.

Normalization, unchanged and sufficient as the import key: digits only, require ≥8,
strip leading zeros, left-pad to 12.

One caveat on that rule worth Stephen's eye: stripping leading zeros deliberately
collapses `0814927020368` and `814927020368` to the same key. That is correct for
UPC-A/EAN-13 of the same product, and it is why the rule works across five feeds that
zero-pad differently. It would be wrong if any vendor ever ships a genuine 13-digit
GTIN whose leading digit is significant. Not observed tonight.

### What it rejects — the number that forces the decision

| import key | products rejected | % |
|---|---|---|
| UPC only | **2,333** | 1.9% |
| UPC **or** `(brand, mpn)` | **96** | 0.08% |

Breakdown of the 2,333 that UPC-only would reject: cssi 1,410, rsrgroup 622,
pawholesale 174, zanders 68, davidsons 59.

The 96 that fail even with the fallback: davidsons 59, rsrgroup 33, zanders 4,
cssi 0, pawholesale 0. **96 records out of 124,157.** That is a quarantine bucket a
person can read in one sitting, which is the right size for a hard floor.

**Recommendation: adopt the fallback.** It is the difference between rejecting an entire
vendor (PA Wholesale, 100%) plus 1,410 CSSI products, and rejecting 96 rows total.
Stephen has already ruled the fallback is required; this quantifies what it buys.

### Ambiguity — where identity is present but not unique

| vendor | intra-vendor UPC collisions | `(brand,mpn)` collisions |
|---|---|---|
| davidsons | 30 rows / 15 UPCs | n/a |
| zanders | 0 | 55 rows / 27 keys |
| cssi | 219 SKUs / 103 UPCs | — |
| rsrgroup | 0 | 182 rows / 57 keys |
| pawholesale | 0 | 0 |

Small and bounded — roughly 486 rows across all five feeds. These need a deterministic
tiebreak, not a rejection.

**`(brand, mpn)` must never be `mpn` alone.** Bare MPNs collide across manufacturers.
Today PA Wholesale rows vanish silently and nothing is corrupted; an mpn-only join would
attach them to the *wrong* canonical product and still exit 0 — trading invisible
omission for invisible corruption.

**PA Wholesale's `sku` is a distributor SKU** and must never be promoted to a fallback
MPN on a future pull where `mpn` is missing. This repeats a previously-caught error where
a derived-identifier rule validated against the wrong authority would have promoted
distributor SKUs into canonical MPNs.

**The multi-GTIN disqualification rule is still relevant.** `identity.py build_key_index`
disqualifies a `product_key` when it sees ≥2 GTINs for that key. Feeding it a UPC-less
vendor changes clustering for *real* suppliers unless scoped. Likewise Pass A mints a
cluster for any row with a valid GTIN, so a UPC-less vendor needs an **attach-only** path
that joins an existing cluster or reports the row unmatched — never creates one from
itself.

---

## 4. Validation shape

**Three outcomes, not two: pass / quarantine / reject.**

- **pass** — meets the floor. Proceeds to import.
- **quarantine** — identity present but ambiguous (the ~486 collision rows), or a
  vendor-structural gap that a human rule could resolve. Held, listed, re-runnable.
- **reject** — no identity at all (the 96). Cannot be imported by any rule; needs the
  vendor to fix the feed.

Two-outcome pass/reject would force the 486 ambiguous rows into one of two wrong
buckets: dropped silently, or imported as duplicates.

**Reuse the existing concepts rather than inventing parallel ones:**

- The merge stage already emits `merge_quarantined.csv`, `merge_conflicts.csv`,
  `merge_unmatched.csv` and `merge_key_collisions.csv`. The importable gate should emit
  in the same shape and, ideally, the same files — a second, differently-shaped quarantine
  is how rows get lost between two systems that each think the other has them.
- `field_sources` (written into every canonical record by `canonical.py:135-141,158`) is
  already a published provenance vocabulary, with values `only_nonempty` / `precedence` /
  `quality_override`. The gate's reason codes should extend that vocabulary, not compete
  with it.

**Where the gate belongs: at the boundary that produces importable records — the end of
Transform, before Load.** Not inside the merge. The merge's job is deciding which vendor
wins a field; the gate's job is deciding whether the resulting record is fit to leave the
building. Putting the gate inside the merge conflates "which value" with "is this a
product at all."

**Non-negotiable, given tonight's findings: the gate must emit counts and fail loudly on
a shortfall.** Every defect found in this repo tonight — the stale read-side directory,
the fresh mtimes over December content, PA Wholesale's silent disappearance, the CSSI
row-count inflation — shares one root cause: *the pipeline has no notion of "I produced
less than I should have."* A gate that only filters, without asserting expected volume,
adds a sixth silent failure to the five that exist.

---

## 5. What I cannot determine from the feeds repo — open questions for Stephen

These depend on the Akeneo side. I have not assumed defaults for any of them.

1. **What is Akeneo's `identifier` attribute going to be** — vendor SKU, normalized UPC,
   or a synthesized canonical key? This decides whether the 2,333 UPC-less products can be
   imported at all, and whether cross-vendor products merge in Akeneo or in the pipeline.
2. **Which attributes are required on the target family?** If the family marks anything
   beyond identifier+label as required, the floor above is too small and the gap must be
   filled with a default rather than by blocking import.
3. **Are there mandatory locale/scope combinations?** A record complete in `en_US` may be
   incomplete under a channel that expects another locale, which changes what "importable"
   means without changing the data.
4. **Does the Akeneo API reject a partial record, or accept it and mark it incomplete?**
   If it accepts, the floor can be smaller still and Akeneo's own completeness dashboard
   becomes the enrichment worklist — which fits the ratified "Akeneo is the human editing
   surface" model better than a strict pre-import gate does.
5. **Family assignment** — is there a rule mapping vendor category to Akeneo family, and
   is family required at import? If yes, category quietly becomes a floor field for some
   vendors and I have not measured its coverage.
6. **Variants** — do any of these become Akeneo product models with variants, or is
   everything a simple product? Affects whether the CSSI image collapse in §0 is the only
   grain problem or the first of several.

Question 4 is the one I'd want answered first: it determines whether this contract is a
gate at all, or just a reporting layer over an import that would have succeeded anyway.

---

## 6. Summary for a ruling

- Floor: `vendor_sku` + `name` + `brand` + (UPC **or** `(brand, mpn)`).
- Cost of that floor on tonight's data: **96 rejected records out of 124,157 (0.08%)**.
- Cost without the MPN fallback: **2,333 rejected (1.9%)**, including all of PA Wholesale.
- Ambiguous-but-present identity needing tiebreak, not rejection: ~486 rows.
- ~~Blocking prerequisite: CSSI must be collapsed from 84,380 image rows to 47,971
  product records.~~ **RETRACTED — the transform already does this correctly** (48,102
  rows, one per `vendor_sku`, zero duplicates, `images` as a JSON array). The caveat that
  survives: anything reading the raw feed directly, bypassing the transform, must collapse
  by SKU itself.
- The gate emits pass/quarantine/reject with reasons, reuses the existing
  `merge_*.csv` outputs and the `field_sources` vocabulary, sits between Transform and
  Load, and asserts expected volume rather than filtering silently.
