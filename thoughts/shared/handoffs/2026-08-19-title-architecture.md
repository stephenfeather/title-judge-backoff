---
root_span_id: ba5a6e27-0fd3-4361-bb0f-3d134186f656
turn_span_id: 
session_id: ba5a6e27-0fd3-4361-bb0f-3d134186f656
---

# Handoff — title architecture

**Repo:** `title-judge-backoff` · **Date:** 2026-08-19 · **Role:** consulting

Read time ~3 min. Assumes zero context.

---

## 0. Scope boundary — read this first

**You consult. You do not work in `inventory-feeds-management` or
`featherarms-infrastructure`.** Analysis, diagnosis and argument are yours. Findings go to
the team lead; someone working in that repo acts on them.

This is the thing most likely to be forgotten. It was corrected once already, after a
durable artifact ended up owned by an author with no standing to maintain it.

Settled decisions about the feeds pipeline live in `inventory-feeds-management/DECISIONS.md`.
Read it before forming any view about that pipeline. Entries are marked by how they were
established — RULING / VERIFIED / RELAYED / LEAD — and only the LEAD class has ever been
wrong.

---

## 1. The ruling, and what it means for this repo

**RULING (Stephen, 2026-08-04, reaffirmed 2026-08-18):** the LLM **authors** titles.
Deterministic lexicons are **inputs and guardrails**, not a generator with a grader bolted
on top. His words: *"it would be much better to just have an LLM work on the titles to
begin with."*

**This repo is the grader, built for a generator that was never built.** It was created
2026-08-05 on a misreading of a 2026-08-04 quote; 55 PRs followed on that premise. The
inversion is the repo's reason for existing, so the ruling is existential rather than
directional.

### What has a role under the ruling

| survives | why |
|---|---|
| `judge/client.py`, `backends.toml`, rate limiting, retry, `--check-backends` | Provider plumbing. An authoring pass needs the same transport. |
| `judge/schema.py::Pair`, `adapt_qa_pack.py` | Corpus loading and the before/after pair shape. An eval set for a generator needs the same inputs. |
| `judge/cache_collapse.py`, `judge/deadlocks.py`, `judge/reliability.py` | Harness-integrity checks. They test the *rig*, not the rubric, so they survive a change of task. |
| `score.py` leaderboard rendering | Presentation, task-agnostic. |

### What does not

| ends | why |
|---|---|
| `judge/prompts.py` rubric (`ACTIVE_REASON_CODES`: overcorrection, meaning_change, casing_error, ok) | These are grading verdicts on a script's output. Under the ruling there is no script output to grade. |
| The approve/reject framing throughout `judge/vote.py`, `judge/flips.py`, `judge/agreement.py` | Measures agreement between judges about a change. An authoring pass produces a title, not a verdict on one. |
| Kappa as the headline metric (`judge/stats.py`) | Inter-rater agreement answers "do judges concur", which is not the question once the LLM writes. |

**The honest framing for a successor:** this is not a repo to extend. It is a repo to
harvest — take the client, the corpus loader and the integrity checks into an authoring
eval harness, and retire the rubric.

---

## 2. What the feeds pipeline actually does with titles

Everyone assumed wrongly here, repeatedly. **There is no LLM anywhere in the feeds
pipeline.** Titles are *selected* from vendor feeds and arbitrated by arithmetic.

All references `inventory-feeds-management`, `Transform/src/merge/`, on `origin/main`.

**`selectors.py::pick_name` (:98-110)** — takes candidates in vendor-precedence order, then:

1. `:103-104` — if the winning vendor is in `AUTHORITATIVE_NAME_VENDORS`, take the value
   **verbatim**, reason `authoritative`.
2. `:105-106` — else if `name_quality(value) >= NAME_QUALITY_THRESHOLD`, take it.
3. `:107-110` — else take the highest-scoring candidate instead, reason `quality_override`.

**`selectors.py::name_quality` (:41-49)** — the three-term heuristic, in full:

```python
length_score = min(len(text), 120) / 120 * 0.5   # longer is better, capped at 120 chars
case_score   = 0.3 if any(c.islower() for c in text) else 0.0
words_score  = 0.2 if len(text.split()) >= 5 else 0.0
```

`NAME_QUALITY_THRESHOLD = 0.45` (`policy.py:83`). That is the entirety of the pipeline's
notion of a good title: **length, presence of a lowercase letter, and at least five words.**

### The deletion set, under the ruling

- `name_quality` — `selectors.py:41-49`
- the override branch of `pick_name` — `selectors.py:107-110`
- `NAME_QUALITY_THRESHOLD` — `policy.py:83`
- probably `pick_description`'s ratio override — `selectors.py:121-125`, with
  `DESCRIPTION_OVERRIDE_RATIO = 2.0` and `DESCRIPTION_OVERRIDE_MIN_CHARS = 120`
  (`policy.py:86,88`). Same shape: arithmetic standing in for judgement. Flagged as
  *probably* because nobody has argued it explicitly.

### The seam that survives — and it already exists

Do not build a new insertion point. `AUTHORITATIVE_NAME_VENDORS = frozenset(ENRICHMENT_VENDORS)`
(`policy.py`) means a title from the `enriched` vendor is **taken verbatim, bypassing the
quality gate** — E8's rationale being that quality is the emitter's responsibility. Paired
with `NAMING_PRECEDENCE = ENRICHMENT_VENDORS + VENDOR_PRECEDENCE`, which puts `enriched`
above every real vendor for `name` only.

**That is the LLM's socket.** The mechanism for an authoritative external title already
ships. What changes is who emits it and when — see §3.

### Public vocabulary — a compatibility constraint

`field_sources` is written into every canonical record (`canonical.py:135-141,158`) and
publishes `quality_override` on **785 records** (name reason distribution:
`only_nonempty` 63,780 / `precedence` 18,925 / `quality_override` 785).
`canonical_products.csv/.ndjson` are consumed downstream. Deleting the heuristic removes a
value from a published vocabulary — coordinate, don't just delete.

*(RELAYED — the 785 figure and the distribution are team-lead's measurement, not
re-derived here.)*

---

## 3. The sequencing constraint

**Titling cannot stay in the merge layer.**

`pick_name` works by ranking a **slate of competing vendor rows** for one product. That
slate exists only inside Stage 2.5. Under the standing ruling that enrichment is
post-import and in-place (D-13), by the time an LLM would author a title the product is a
single Akeneo record — **there is no slate left to rank.**

So authoring moves to a **post-import stage that does not exist**, downstream of Load.
Load itself is greenfield: `Load/` is an empty directory and `run.py`'s load domain is a
placeholder with `script: None`.

Consequence: **the authoring pass is blocked on Load**, not on prompt design or model
choice. Anyone starting this by writing prompts is starting at the wrong end.

The `enriched` seam in §2 still matters — it is how an authored title re-enters a *merge*
if a re-merge ever happens — but the primary write path under D-13 is Akeneo-side.

---

## 4. First real worklist: archery

**1,320 products in Akeneo, with a visible good/bad title split.** The best available
starting corpus: real records, already imported, in a family that is populated.

⚠ **The `value_count` 6-vs-7 correlation with title quality is SAMPLED, not confirmed
across all 1,320.** Do not build a selection rule on it before checking it holds. If it
does, `value_count` is a free triage signal; if it doesn't, it is coincidence in a small
sample.

*(RELAYED — the 1,320 count and the split are team-lead's observation against the live PIM.
Not verified here.)*

Context: the PIM holds ~8,967 products total, skewed to non-firearm families
(`family=rifle` returns 0). Archery is one of the populated ones.

---

## 5. This repo's own open items

**5.1 The judge asks for something its inputs never supply.**
`judge/prompts.py:54` rubric requires "correct brand casing". `:80,:82` make brand/MPN a
*conditional* branch. `judge/schema.py:66-69` relaxed both to `str | None` with the comment
"brand/mpn are absent from some sources (e.g. the E10 QA pack)". Result: the brand/MPN
branch **never executed in any of the 4,800 v3 votes**, while the rubric kept asking for it.
Git archaeology puts the seam on day one — `911c152` scaffolded them required, `7f785cb`
relaxed them the same day.

Related: the judge is never told the pipeline expands abbreviations, so it **cannot report
undercorrection even in principle**. That kills any v4 undercorrection reason code more
decisively than corpus analysis would.

**5.2 Kappa cannot estimate production accuracy.**
The E10 pack self-documents as stratified and seeded (`seed=20260804`), *"deliberately NOT
a uniform random sample — a pack of only easy wins is a sales pitch, not a review."*
The caveat belongs in `leaderboard.md` and is currently absent.

**5.3 The 82-skipped-titles proxy bug — fix the proxy, keep the guard.**
`infrastructure/enrichment/scripts/lexicon.py` refuses to touch any title containing a
lowercase letter. **That guard is sound and load-bearing:** it protects clean CSSI titles,
and it forces any mixed-case-introducing stage to run last. Do not relax it.

What is wrong is only its *proxy*. "Contains a lowercase character" is standing in for
"already cased by a good vendor", and these false-positive:

- metric cartridge/dimension notation — `5.56x45`, `7.62x51`, `7.62x39`, `.578x28`,
  `5/8x24`, `4x32`, `12"x17`
- model designators — `GEN3i`, `X1i`, `LCRx`, `43x`
- ordinary text — `25th`, `w/`

Verified against the 200-row QA corpus: 19 of 20 unchanged-cohort rows contain a lowercase
character, 0 of the other 180. 82 titles catalog-wide are skipped.

**Fix:** exclude a lowercase `x`/`i` between digits or inside a known dimension/designator
pattern before deciding a title is already cased. Narrow the proxy; keep the guard.

**5.4 Corpus location.** Not in this repo — `*.jsonl` and `results/` are gitignored and the
source pack lives at
`featherarms-infrastructure/infrastructure/enrichment/qa/title-qa-pack.md`.
Rebuild: `uv run adapt_qa_pack.py template --pack <path> --out rulings-template.jsonl`
(200 rows: 40 most-changed, 20 quarantined-brand, 120 representative, 20 unchanged).
`score.py` additionally needs operator-ruled `pairs.jsonl` via `adapt_qa_pack.py merge`.

⚠ Pair ids are content-derived (`e10-<sha256 of before+after>`) and vendor feeds drift, so
a corpus can **silently stop joining** to source data and lose rows without erroring. Verify
by intersecting regenerated ids against the `pair_id` values in any older results directory.

---

## 6. Tooling trap that has caused three false claims

**An unqualified `fd` or `rg` result is not a search of these repos — it is a search of the
tracked subset.** `output/`, `Transform/output/`, `Extract/<vendor>/output/`, `results/` and
`*.jsonl` are gitignored, and that is where the data lives. Use `fd -I` / `rg -uu`, or name
the path directly.

This produced three false absence claims across two agents and two sessions. An empty result
presents as fact and invites no inspection — absence has no surface to check.

---

## 7. Leftover state awaiting disposal

Factual, no recommendation. Stephen has not ruled on these.

| what | where | status |
|---|---|---|
| branch `agent-decisions-pointer`, commit `1183083` | `featherarms-infrastructure` | pushed, **no PR**. A CLAUDE.md pointer block that was told not to land. |
| worktree `.claude/worktrees/agent-decisions-pointer` | `featherarms-infrastructure` | gitignored |
| worktree `.claude/worktrees/agent-decisions-register` | `inventory-feeds-management` | gitignored; holds a staged, uncommitted `DECISIONS.md` |
| uncommitted `CLAUDE.md` edit | both repos | pointer/READ-FIRST blocks |
| untracked `DECISIONS.md` | `inventory-feeds-management` main checkout | superseded by the landed version |

PR #12 (`agent-decisions-register`) was **closed** by team-lead. Do not build on that branch.
