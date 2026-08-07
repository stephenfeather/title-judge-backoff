# Agent instructions

## Running tests

```sh
uv run pytest
```

That's the whole suite (~1s). Run it from the repo root; `uv` resolves the
environment from `pyproject.toml` (pytest is in the `dev` dependency group)
and `pythonpath = ["."]` makes `judge/` and the root scripts importable.

Narrower runs:

```sh
uv run pytest tests/test_schema.py          # one module
uv run pytest -k kappa                      # by keyword
uv run pytest -x -q                         # stop on first failure, quiet
```

## Test rules

- **No live API calls in tests — ever.** HTTP behavior is tested with
  `httpx.MockTransport`; time/rate limiting with injected `now`/`sleep`
  callables. A test must pass with no network and no API keys set.
- **Synthetic fixtures only.** Invented titles (e.g. "Acme Widget 3000") and
  invented pair ids (e.g. `e10-aaa`). Never put vendor-derived titles, real
  pair ids, or their measured vote distributions in tests — **this repo IS
  public**, so anything committed is published the moment it lands. `*.jsonl`
  is gitignored for that reason.

  This line used to read "may go public", and both a human and an agent read
  it as a future concern while copying real pair ids into fixtures. It is not
  a future concern. The same rule applies to anything else that gets published
  from here — issue bodies, PR descriptions, commit messages.
- **TDD is house policy.** Write the failing test first, watch it fail,
  then implement. New behavior in `judge/`, `run_bakeoff.py`, or `score.py`
  needs a test in `tests/`.
- Keep tests hermetic: use `tmp_path` for files, `monkeypatch` for env vars
  (e.g. `NVIDIA_API_KEY`), no shared state between tests.

## Layout

| Path | Tested by |
|---|---|
| `judge/schema.py` | `tests/test_schema.py` |
| `judge/prompts.py` | `tests/test_prompts.py` |
| `judge/client.py` | `tests/test_client.py` |
| `judge/vote.py` (majority, flip rates) | `tests/test_vote.py` |
| `judge/rulings.py` (ruling journal) | `tests/test_rulings.py` |
| `judge/flips.py` (per-pair flip stats) | `tests/test_flips.py` |
| `rule.py` (ruling TUI) | `tests/test_rule.py` |
| `judge/stats.py` (spread, bootstrap) | `tests/test_stats.py` |
| `judge/check.py` | `tests/test_check_backends.py` |
| `backends.toml` itself | `tests/test_backends_config.py` |
| `run_bakeoff.py` (pure helpers, concurrency) | `tests/test_run_bakeoff.py` |
| `score.py` (metrics, leaderboard) | `tests/test_score.py` |

## Launching anything that hits a backend

Always source both env files in the same shell — see the README section
"Launching a run". Sourcing `042_env_ai_tokens` alone blanks every key, because
`get_secret()` is defined in `002_functions`:

```sh
zsh -c 'source ~/.zsh/zshrc.d/002_functions && source ~/.zsh/zshrc.d/042_env_ai_tokens && <command>'
```

A missing key skips a backend silently rather than failing, so `run_bakeoff.py`
refuses to start when any backend lacks one (`--allow-skipped` to override).
Do not weaken that guard — it exists because a long unattended sweep would
otherwise produce a quietly incomplete report.

## Sampling parameters

**Never add a harness-wide temperature constant back.** `temperature` is opt-in
per backend and omitted by default — see `TEMPERATURE_OMITTED` in
`judge/client.py`. Sending a value to a model that rejects it fails the whole
run; recording 0.0 for a call that never sent the field mislabels a sampled
verdict as deterministic. Same for `reasoning_effort`: it changes which reason
code the model returns, so it is pinned per backend and recorded on every
verdict, never left to a provider default.

Stability comes from `--votes` (majority-of-N, default 3), not from sampling
parameters. Any new metric must be reported with its spread.

## No majority is a state, not a value

`VoteResult.verdict` and `VoteResult.reason` are **`None`** when the votes
reached no majority — when two or more values shared the top count, so any
winner would be the tie-break's invention. Use `settled_majority()` for
rulings; `majority()` still returns a modal value and exists only for
`flip_rate`, which is invariant to which tied candidate wins.

Rules for any new consumer:

- **Decide explicitly.** `None` is deliberate so that ignoring it raises
  `AttributeError` rather than producing a plausible wrong number.
- **Verdict and reason settle independently.** The common shape is a settled
  verdict with an unsettled reason — three votes agreeing on "reject" while
  splitting 1-1-1 on why. That verdict is real evidence and must still score.
- **Exclude from metrics, and COUNT the exclusion.** `score.py` reports
  `n_unsettled_verdict` and `n_unsettled_reason`; the scenario report
  enumerates the pairs. A metric over 189 of 200 is honest; a silent exclusion
  is its own kind of lie.
- **Never compare two unsettled values.** `None == None` would score two
  models that both failed to decide as agreeing.
- **Flip rates always survive.** An unsettled pair is contested, not missing,
  and the ruling queue ranks it up for exactly that reason.

Verdict ties are not hypothetical: a backend that stops short leaves pairs
with two votes, and inkling has one such 1-1 split in the 2026-08-06 run.
"Verdict is binary so it cannot tie" holds only for a complete run.

## Concurrency

`--concurrency N` runs N calls per backend and runs backends in parallel; the
default of 1 is fully serial on both axes and must stay byte-identical to
pre-concurrency runs. Three invariants hold this together — do not weaken them:

- **Rate limits key on HOST, never on backend.** Six `nvidia-*` backends share
  one endpoint and one quota. `LimiterRegistry` hands every backend on a host
  the same `RateLimiter`, at the lowest `rpm` any of them declares. Keying per
  backend is what produced the 2026-08-06 503 storm.
- **`RateLimiter.wait()` sleeps while holding its lock.** That is deliberate:
  releasing first lets every waiting thread read the same `_last` and fire
  together. Only admission is serialized; the HTTP call runs outside the lock.
- **One writer per results file.** Workers hand lines to `ResultWriter`, whose
  single thread owns the handle. The append-and-flush-per-line contract is what
  makes a killed run resumable, and it only holds with one writer.
- **A 429 backs off the HOST, not the worker.** `RateLimiter.penalize()` holds
  every caller on that host — including other backends sharing it. Backing off
  only the worker that caught the 429 leaves the other N-1 hammering an
  overloaded host, which is the concurrency version of the same bug. The call
  itself is not retried in-run; resume covers it like any other failed vote.

Never infer rate-limit headroom from response headers. DeepInfra — the
successor host for deepseek after NVIDIA's free tier ends — sends no
`x-ratelimit-*` and no `Retry-After` at all (verified live 2026-08-07). Treat a
declared `rpm` as a request, not a guarantee; a 429 is the host telling you the
bucket was too generous.

The todo list is partitioned ONCE, before any worker starts. A worker that
re-read the results file could hand the same `(pair_id, run_index)` to two
threads. Nothing may depend on jsonl line order — `tally_votes` sorts each
pair's votes by `run_index` and `score_model` orders by the pairs file so that
tie-breaks and the bootstrap CI stay reproducible.
