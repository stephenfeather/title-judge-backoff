# DECISIONS register — entries classified by how they were established

**From:** title-judge (consulting) · **For:** pipeline-dev · **Date:** 2026-08-19

Markers: **RULING** (Stephen decided) · **VERIFIED** (read from code or measured; source
stated) · **RELAYED** (established by someone else, I only wrote it down) · **LEAD** (my
diagnosis, unmeasured).

I added **RELAYED** to the three you proposed. Several entries I would have called VERIFIED
were verified by *you or team-lead*, not by me, and collapsing that into VERIFIED is how a
register launders a second-hand claim into a fact. Treat RELAYED as "trust the person named,
not the register."

Dependency order within each section: premises before conclusions.

---

## §1 Rulings — land as-is, nothing to spot-check

| id | claim | who / when |
|---|---|---|
| **D-11** | The LLM authors titles; lexicons are constraints, not the generator | Stephen, 2026-08-04, reaffirmed 2026-08-18 in his own words |
| **D-12** | Akeneo is the human editing surface; operator edits round-trip via the harvest loop | Stephen, 2026-08-04, superseding "never edit in Akeneo" |
| **D-13** | Enrichment is not a pre-import gate | Stephen, 2026-08-18 |
| **D-06** *(requirement half)* | PA Wholesale has no UPCs, so an MPN fallback is required | Stephen, 2026-08-18 |
| **X-05** *(ruling half)* | Repair the rows and keep the specs; loud counts both ways | Stephen, 2026-08-19 |

⚠ **D-12 is ratified but NOT implemented.** The entry says so; keep that, because `operator`
appears in no precedence tuple in either tree and it would otherwise read as shipped.

---

## §2 Verified by me — each names a source you can check in seconds

**Identity (premises first):**

| id | claim | source |
|---|---|---|
| **D-01** | Pass A GTIN → Pass B strong brand+MPN → Pass C bridge (shipped disabled, D7) → Pass E enrichment attach-only | `identity.py` module docstring + `build_clusters(…, bridge=False)`, on **origin/main** |
| **D-02** | `key_strength` returns "strong" only when brand AND mpn are both non-empty | `identity.py:79-88` |
| **D-03** | `_is_low_trust` rejects `mpn == normalize_mpn(vendor_sku)` + `MIN_TRUSTED_MPN_LENGTH` | `identity.py:107-110` |
| **D-04** | product_key with ≥2 GTINs is disqualified; same `KeyIndex` feeds clustering and the collision report | `identity.py:113-142` |
| **D-17** | `fa_g_<gtin14>` / `fa_k_<16 hex>`; malformed input raises | `canonical_source.py:128-141`, `PK_TRUNCATE=16` at `:113` |

**Akeneo path:**

| id | claim | source |
|---|---|---|
| **D-18** | Empty `upc` is omitted entirely, never sent as "" | `canonical_payload.py:299-301` |
| **D-20** | The importable contract already exists; the 14 skip codes are the contract | `canonical_payload.py` — resolution helpers `:163-190`, allowlists `:83-135`, `missing_required` `:403-409`, report-not-skip rationale `:388-398` |

**Feed / merge shape:**

| id | claim | source |
|---|---|---|
| **D-08** | Commerce state is offers-only, test-enforced | `policy.py::COMMERCE_FIELDS` |
| **D-09** | `VENDOR_PRECEDENCE` is real suppliers only; `NAMING_PRECEDENCE` is name-only | `policy.py` on **origin/main** (E7, `c237cce`) |
| **D-05** | Per-vendor identity capability; Davidsons has no MPN column, PA Wholesale no UPC | measured on `output/` 2026-08-17/18 + Davidsons header |
| **D-07** | CSSI raw feed is image-denormalized; **the transform already collapses it** | raw: 47,971 SKUs / 84,380 rows, 15,568/15,568 multi-row SKUs vary only in `Image Location`. output: `Transform/output/cssi.csv`, one row per `vendor_sku`, zero dupes |
| **D-14** | Selection dominates: 80.8% of Davidsons UPCs also in CSSI or Zanders, 19.2% unique | re-derived on tonight's `output/` |

---

## §3 Relayed — real, but somebody else established them

**Land these crediting the measurer, not me.**

| id | claim | who actually established it |
|---|---|---|
| **D-10** | CSSI wins `name` 99.91% / body 99.42%; 15,531 of 15,572 contested clusters | team-lead. **I never measured this.** |
| **D-15** | Manifests, `STALE_INPUT_MAX_AGE_DAYS=7`, zero-row guard | team-lead (your PR #11). You have since confirmed the manifest fields at `base.py:328-345`, `:403`. |
| **D-16** | Akeneo contract: `sku` identifier, `upc` unique, `name` localizable, 24 families | team-lead, against the live PIM |
| **D-19** | 8,967 products total (unfiltered), `family=rifle` returns 0 | team-lead. My only contribution is the stale-comment discovery at `canonical_payload.py:388-389`, which **is** mine and is VERIFIED. |
| **X-01 / X-02** *(closures)* | Root causes and fixes | you (PRs #9, #11). X-02's float64/leading-zeros diagnosis is entirely yours. |

---

## §4 Leads — unmeasured. Do not land as fact.

| id | claim | what would settle it |
|---|---|---|
| **D-06** *(mechanism half)* | The ruling "maps to `bridge=True`, not new code" | Nobody has run the merge with `bridge=True`. Two sub-questions are pure opinion: enable globally or per-vendor, and what Pass C should do when a key maps to >1 GTIN (today it bridges only at exactly one). |
| **X-03** | "Nothing notices a run producing 65% of yesterday's rows" | An absence claim. You've now half-confirmed it via the manifest read — it records `output_rows` but nothing compares across runs. Worth re-stating from your evidence rather than my assertion. |
| **D-16** *(one clause)* | "Akeneo accepts partial records" | team-lead flagged this as **not empirically tested** — inferred from the family/attribute model and API shape. The entry reads more confident than the evidence. Several of my downstream conclusions rest on it. |
| **D-20** *(residue clause)* | "`family_unrouted` / `brand_unmapped` should import-and-report rather than skip" | This is an *argument*, not a finding. It's a policy question for Stephen. The surrounding factual table is VERIFIED; this clause is not. |

---

## §5 What I'd retract or soften today

You asked for these specifically. In rough order of how much they'd mislead.

1. **X-05, as written in closed PR #12 — retract wholesale.** Wrong break column (said
   `Web Item Description` [6], is `Specifications` [22]), wrong repair direction (right-anchor
   validates 0/1,073), and a parser warning that misdescribed this repo's validating fallback.
   Superseded by your PR #14. Nothing from my version should survive except Stephen's ruling
   and the field-width census.

2. **My "4 rows are the natural reject bucket" — retract.** They are valid products with an
   empty UPC; my `UPC ≥ 8` predicate was wrong, and applied consistently it would have
   discarded 1,267 structurally perfect rows (1.52%) on every run. Correct figure: 1,073/1,073
   recovered, reject bucket empty. Already corrected with team-lead.

3. **D-10 — soften to RELAYED or drop.** I wrote it as though measured. I never touched it.

4. **D-04a — keep the conclusion, date the counts.** "67,011 GTINs, 67,011 distinct, zero
   duplicates" came from a **December-sourced** `canonical_products.csv`. The structural
   argument (Pass A clusters on GTIN, so same-UPC rows can't become two products) is
   vintage-independent and is the real basis. Re-measure the counts post-CSSI-repair before
   quoting them.

5. **X-04 — largely superseded.** "`importable` is undefined" was true when written and is
   false now: D-20 shows `canonical_payload` defines and enforces it. What survives is only
   that `Load/` is empty and nothing calls the writer.

6. **D-05's CSSI percentages — note the grain.** Measured at row level on the raw feed, which
   is image-denormalized (D-07). Presence percentages are unaffected in practice because
   CSSI's fields are ~100% populated, but the denominators are rows, not products.

7. **All five M-notes are opinion, not fact.** M-01…M-05 are method guidance I wrote from my
   own mistakes. They've held up, but they belong in a clearly-marked advisory section rather
   than beside `file:line` claims. M-05 is team-lead's wording, not mine.

---

## §6 One thing I'd add, from your correction

Your point deserves to be the register's own rule, and it's better than my M-notes:

> **When a predicate rejects a large, structurally clean population, that is evidence about
> the predicate, not the data.** The tell was available with no extra measurement — my check
> condemned 1,267 otherwise-perfect rows.

And the meta-lesson from how it was caught: **two people measuring independently beat one
relaying to the other.** The disagreement between 1,069 and 1,073 did the work. Neither of us
assuming the other was right is what surfaced it — which is an argument for the RELAYED
marker above existing at all.
