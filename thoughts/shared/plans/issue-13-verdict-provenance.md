# Issue #13 — Verdict record carries no provenance (design proposal, NOT implemented)

Status: **SHIPPED**. Merged 2026-08-08 as PR #28 (merge commit `65a11fc`), closing #13.
Revision 2 folded in the Codex adversarial review (v1 verdict: needs-attention).
Revision 3 (below) folds in the two P2 findings from the bot review ON PR #28.

Commits: `fbd06ed` provenance module · `4a6ea26` record fields · `3e4f4a0` guard and CLI ·
`60508c1` audit counts malformed rows · `02da3b4` partial rows unproven + default wiring.
408 tests passing on main.

## Revision 3 — what the PR review changed

Two P2 findings, both real, both fixed before merge:

1. **Partially stamped rows passed the guard.** `run_backend(code_version=X)` with the
   default `client_factory` built `JudgeClient(backend)` with no code_version, so rows
   carried base_url and config_digest but a null code_version. The unproven test required
   ALL fields absent, and `_check_provenance` skips nulls — so those rows resumed silently
   under different code. Fixed at both levels: any expected field the row cannot answer now
   marks it unproven, AND `client_factory` defaults to None so `run_backend` builds the
   client itself and cannot be constructed unwired. `main()` no longer needs its lambda, so
   the guard and the stamp read the same argument and cannot diverge.
2. **Malformed rows vanished from the audit.** A corrupt row was counted in `total_rows`
   then dropped, so a damaged file with uniform provenance reported `is_mixed=False`. Added
   `malformed_rows`; `is_mixed` is true when unreadable rows sit beside known provenance.
   Deliberately counts rather than raises — an audit that refuses to run on a damaged file
   is useless exactly when it is needed; failing closed here means never reporting *coherent*.

**Pattern worth carrying forward:** the same shape — provenance that looks present but is
only half-populated — was caught in the rev 1 review, and again by the PR bot. The weakness
of an identity guard is rarely the missing-everything case; it is the partial record.
Enumerate partial states explicitly when designing this kind of check.

## Problem

A verdict row records *what was decided* and *part of* the config it was decided under
(`model_id`, `prompt_version`, `temperature`, `reasoning_effort`) but not *which host
served the call*, not *which code produced it*, and not the rest of the backend
configuration that shapes the request.

`already_judged_ids()` (`run_bakeoff.py:57-98`) builds its resume guard from exactly the
four config fields the row carries, compared as a 4-tuple. Anything not on the row cannot
be guarded.

### Failure 1 — two providers, one file

Two hosts serving the same `model_id` string let a resume append the second provider's
verdicts to the first's. Nothing raises; downstream kappa / flip-rate treat it as one
coherent run. Prevented today only by a capitalization accident in DeepInfra's ID
(`deepseek-ai/DeepSeek-V4-Flash` vs NVIDIA's lowercase) — luck, not a guard, and it expires
as backends consolidate onto DeepInfra after NVIDIA's free tier ends.

### Failure 2 — two code versions, one file

Nothing records which code produced which row. This nearly happened during #10: the shared
checkout was left on a feature branch; a sweep launched from it would have run unmerged
judging code against paid endpoints and appended those rows into S1 files produced under
`main`. Caught by human inspection, not by any guard, and *undetectable after the fact* —
there is no field to compare, so existing `results/` cannot be audited to find out whether
it already happened. Requires no unusual action, only a branch left checked out.

### Failure 3 — same model, same host, different request (added in rev 2)

`Backend` carries `api` (openai vs anthropic wire protocol) and `structured_output`
(strict json_schema on/off). Both change the request body, the endpoint path, and the
response-extraction path in `JudgeClient.judge()`. Neither is on the verdict row. A run
with the same `model_id`, `prompt_version`, `temperature`, `reasoning_effort`, host and git
sha but a different `--backends` file mixes materially different judgments and passes every
guard proposed in rev 1. This generalizes: any future `Backend` field that shapes the
request reintroduces the same hole.

### Migration constraint

Every existing row in `results/` lacks provenance. A guard that reads absent-as-mismatch
refuses to resume anything already collected, invalidating the corpus to fix a hypothetical.

## Proposed design (revision 2)

### 1. Classify every `Backend` field, and enforce the classification

Split `Backend`'s fields into two named sets in `judge/client.py`:

- **verdict-affecting** — `base_url`, `model_id`, `api`, `temperature`,
  `reasoning_effort`, `structured_output`. These change what the model is asked or how the
  answer is read.
- **operational** — `name`, `rpm`, `eval_only`, `api_key_env`, `role`, `timeout_s`. These
  change scheduling, credentials, or analysis labelling, not the judgment.

A unit test asserts that the union of the two sets equals `Backend`'s actual field names.
Adding a field to `Backend` then fails that test until someone classifies it deliberately.
This is the structural answer to the review's "another ad hoc field each time" objection —
the classification cannot silently fall behind the dataclass.

### 2. Three new optional fields on `Verdict` (`judge/schema.py:162`)

All default to `None` so legacy rows load:

- `base_url: str | None` — the full normalized `backend.base_url`, already `rstrip("/")`'d
  at load (`judge/client.py:95`). Human-readable audit trail and the host guard.
- `config_digest: str | None` — 12 hex chars of a sha256 over the canonical JSON of the
  verdict-affecting field set plus `prompt_version`. Catches `api` / `structured_output`
  and everything added later.
- `code_version: str | None` — see §4.

The four existing config fields stay explicit rather than being folded into the digest:
they are read directly elsewhere (`judge/flips.py`, `score.py`, `judge/agreement.py`), and
keeping them named is what lets the mismatch error say *which* field changed instead of
"digest differs".

### 3. Compare the full base_url, not `limiter_host()` — review finding accepted

Rev 1 proposed comparing `limiter_host()` to tolerate path tweaks. Dropped. `limiter_host`
strips scheme *and* path (`judge/client.py:236-247`), so it equates `http://` with
`https://` and equates two gateway paths that can route to different upstreams or tenants.
It is the rate-limit key; it was never meant to be an identity. `base_url` comes from a
checked-in TOML file and is normalized at load — it is stable, so the brittleness rev 1
worried about was hypothetical. Compare it exactly.

*Deferred, not folded in:* the review also noted that an `http://` base_url would send
bearer credentials in plaintext. That is a real but separate defect — it belongs in its own
issue, not smuggled into the resume guard. Flagging, not fixing here.

### 4. `code_version` fails closed — review finding accepted

Rev 1's `<sha>-dirty` collapsed every distinct dirty tree at one commit into a single
value, and its fallback to `None` meant two git-unavailable runs got no protection at all.
Both recreate the failure this is meant to prevent. Replaced with:

- **Clean tree:** `code_version` is the 12-char sha. Normal path.
- **Dirty tree:** refuse to start a run. A paid calibration sweep should not run from an
  uncommitted tree. Escape hatch `--allow-dirty` records
  `<sha>-dirty-<digest12>`, where the digest is sha256 over `git diff HEAD` plus the
  contents of untracked non-ignored `.py` files — deterministic, and distinct trees get
  distinct values.
- **Cannot resolve git state at all** (not a repo, git missing): refuse to start. Writing a
  row whose provenance is unknowable is exactly the thing being fixed.

So `None` means only one thing: *a legacy row, written before this change*. It is never
produced going forward. That asymmetry — unknown is tolerated on read, impossible on
write — is what makes the migration safe without leaving a permanent bypass.

### 5. Guard becomes per-field comparison with a manifest fallback

`already_judged_ids()` compares 4-tuples today (`run_bakeoff.py:81-88`), which cannot
express "absent means skip". New rule, per field:

- found value is set and differs from expected → raise `ValueError`
- found value is `None` → consult the sidecar manifest (below) before giving up
- error names only the fields that actually mismatched, in the voice of the existing
  temperature error
- a `config_digest` mismatch where every named field matches reports: the backend
  configuration differs in a field not recorded per row, compare `<backend>.manifest.json`

`code_version` mismatch is a hard raise, same as `temperature`. Confirmed with the user:
this means any merge to `main` mid-sweep forces a fresh `--out`, and that is the correct
tradeoff for paid calibration runs in a measurement harness.

**Manifest fallback — review finding accepted.** `run_manifest()` already writes `base_url`,
`api`, `temperature`, `reasoning_effort`, `structured_output` and `model_id` to
`<backend>.manifest.json` next to the results file (`run_bakeoff.py:253-268`). Rev 1 threw
that evidence away. For a legacy row with `base_url=None`, read the sidecar manifest and
compare against it. Most existing result directories have one, so most legacy files get a
real host check rather than a free pass.

**Residual unknowns fail closed.** If a results file has rows with no provenance *and* no
readable manifest, refuse to resume unless `--allow-unknown-provenance` is passed. The
manifest fallback keeps this flag rare, so it stays a deliberate act rather than something
pasted reflexively.

### 6. Manifest gains what the row gains

Add `code_version` and `config_digest` to `run_manifest()`. The manifest is the
human-readable expansion the digest points at.

### 7. After-the-fact detection covers unknown-vs-known

Each row carries its own `code_version`, so a mixed file is found by collecting distinct
values across the file. Per the review, the audit treats `None` as its own bucket: a file
mixing legacy rows with new rows is reported as mixed, not silently collapsed to one known
version. Reported per file: distinct code versions, distinct config digests, distinct
base_urls, and a count of provenance-less rows. Wired into the existing results reporting
rather than left as a helper nobody calls.

## Tests

Guard:
- legacy row (no provenance) + matching manifest → resumes
- legacy row + manifest with a *different* base_url → raises
- legacy row + no manifest → raises without `--allow-unknown-provenance`, resumes with it
- same host, same sha, same digest → resumes
- different base_url → raises naming base_url
- different sha → raises naming code_version
- same named fields but `api` flipped → raises on config_digest, message points at manifest
- same named fields but `structured_output` flipped → raises on config_digest

Version resolution:
- clean tree → bare sha
- dirty tree → refuses; with `--allow-dirty` → `<sha>-dirty-<digest>`
- two *different* dirty trees at the same commit → different digests (the rev 1 hole)
- git unavailable → refuses to start

Classification:
- the verdict-affecting ∪ operational sets equal `Backend`'s field names (fails when a
  field is added without classification)

Audit:
- file with two code versions → detected
- file mixing legacy and new rows → detected as mixed

## Acceptance (from the issue)

- Verdict rows carry host and code version. — §2
- Resuming across a different host raises, with the same clarity as the temperature error. — §5
- Existing files with no provenance field still resume. — §5, via manifest fallback
- A file containing rows from two different code versions is detectable after the fact. — §7

## What changed from revision 1

| Review finding | Disposition |
|---|---|
| Host-only comparison is not a provider identity | Accepted — compare full `base_url`; `limiter_host` dropped |
| `api` / `structured_output` outside the identity | Accepted — `config_digest` + enforced field classification |
| `HEAD-dirty` collapses distinct trees; `None` bypasses | Accepted — fail closed on dirty and on unresolvable git; content digest under `--allow-dirty` |
| Legacy compatibility disables the guard | Accepted — manifest fallback first, `--allow-unknown-provenance` only when there is no evidence at all |
| Audit misses unknown-vs-known mixtures | Accepted — `None` is its own bucket |
| `http://` leaks bearer credentials | Deferred to its own issue — real, but not this guard's job |
