# title-judge-bakeoff

A bake-off harness for selecting an LLM to act as an automated judge of
product-title enrichment. Each candidate model reviews a before→after title
change (plus known brand and MPN) and rules **approve** or **reject** with a
reason code. Verdicts are scored against a calibration set of operator-ruled
pairs, and models are ranked on a leaderboard.

## Protocol

1. **Backends** — candidate judge models, declared in `backends.toml`
   (OpenAI-compatible chat endpoints).
2. **Calibration pairs** — operator-ruled `pairs.jsonl`, supplied via
   `--data` (never committed; see `data/README.md` for the schema).
3. **Verdicts** — every backend judges every pair **N times** (`--votes`,
   default 3) with the versioned prompt in `judge/prompts.py`. Results land in
   `results/<date>/<backend>.jsonl`, one verdict per line, tagged with
   `model_id`, `prompt_version`, `temperature`, `reasoning_effort`, and
   `run_index`. Each run also writes `<backend>.manifest.json` recording the
   exact request payload and every model snapshot that answered.
4. **Leaderboard** — `score.py` collapses the N votes per pair to a majority
   ruling, then computes accuracy, Cohen's kappa, false-approve rate, and
   per-reason confusion. It emits `leaderboard.md` grouped into separability
   tiers, with per-run kappa spread, a bootstrap CI, and per-item flip rates.

## Why votes instead of temperature 0

The harness used to pin every call to temperature 0. That is no longer
available, and pretending otherwise would have quietly corrupted the results.

Probing `gpt-5.6-luna` on 2026-08-06 (both `chat/completions` and `responses`):
`temperature=0` returns **400** at reasoning effort `medium`, and **200** at
effort `none`. The GPT-5.x family gates sampling parameters on reasoning being
off, and Anthropic has deprecated the knob outright on recent models. Buying
temperature 0 by setting `effort="none"` is not free either — on a borderline
pair that switched the reason code from `truncation_worse` to `casing_error`
while spending 0 reasoning tokens, and reason codes are exactly what the
per-reason confusion matrix scores.

So determinism comes from **repetition**, not sampling parameters:

- `temperature` is omitted unless a backend proves the endpoint accepts it, and
  verdicts record `null` rather than claiming 0.0.
- Each pair is judged N=3 times and the majority verdict is scored. That cuts
  the effective flip rate from roughly 13% to 5%, and judge-noise sd(kappa)
  from ~±0.048 to ~±0.030 — under the ±0.057 sampling floor of a 200-pair set.
- Every metric is reported with its spread, and **the leaderboard refuses to
  order models whose kappa intervals overlap**. At n=200 that means kappa
  differences below roughly 0.15 are not resolvable; they show up as a shared
  "not separable" tier instead of a fake 1st/2nd.
- Per-item flip rates are a first-class output. Flipping items are the
  borderline set worth sharpening the rubric on, and a rising flip rate is the
  earliest signal a provider changed the model under us.

## Required environment variables

Keys are read from the environment only — never from config or code. A
backend whose key is absent is **skipped**, not fatal.

| Variable | Backends |
|---|---|
| `NVIDIA_API_KEY` | all `nvidia-*` (one key covers every model on the endpoint) |
| `GEMINI_API_KEY` | `gemini-2.5-pro` |
| `XAI_API_KEY` | `xai-grok-4.1-fast` |
| `ANTHROPIC_API_KEY` | `anthropic-haiku-4.5` |
| `OPENAI_API_KEY` | `openai-gpt-5.6-luna` |

### Launching a run — always source both env files

Keys are fetched from the macOS Keychain by `get_secret()`, which is defined in
`002_functions` and *consumed* by `042_env_ai_tokens`. **Every command that
talks to a backend must source both, in this order, in the same shell:**

```sh
zsh -c 'source ~/.zsh/zshrc.d/002_functions && source ~/.zsh/zshrc.d/042_env_ai_tokens && <command>'
```

Two traps this avoids:

- **Sourcing `042_env_ai_tokens` alone sets every key to empty**, because the
  `get_secret` helper lives in the other file. That is worse than not sourcing
  it at all.
- **A missing key is not an error.** The backend is silently skipped and simply
  vanishes from the results — on a multi-hour sweep that is only noticed when
  the report comes back short. `run_bakeoff.py` therefore **refuses to start**
  if any backend lacks a key; pass `--allow-skipped` when the omission is
  deliberate.

Validate the slate before spending anything:

```sh
uv run run_bakeoff.py --check-backends          # key presence only, no network
uv run run_bakeoff.py --check-backends --ping   # also sends ONE real request per backend
```

`--check-backends` reports key presence; `--ping` proves reachability. They are
different questions — a key can be valid while the model id is retired, and a
model can be listed in a provider's catalogue and still 404 on invoke.

`--ping` spends real tokens on paid backends — use it deliberately.

## Usage

```sh
uv run run_bakeoff.py --backends backends.toml --data /path/to/pairs.jsonl --out results/2026-08-05/
uv run score.py --data /path/to/pairs.jsonl --results results/2026-08-05/ --out leaderboard.md
```

`--votes N` sets how many times each pair is judged (default 3). `--votes 1`
disables voting and costs 1×, at the price of an unmeasurable kappa spread.

Runs are resumable: re-running `run_bakeoff.py` skips the individual
`(pair, vote)` calls a backend has already made, so a run interrupted after 2
of 3 votes resumes at the third rather than re-judging or skipping the pair.
Resuming across a changed `temperature`, `reasoning_effort`, `model_id`, or
`prompt_version` is refused outright — use a fresh `--out` directory. Calls are
rate-limited per backend (`rpm` in `backends.toml`).

## Backend roles and protocols

`role = "floor"` marks a baseline that is not a contender — currently
`meta/llama-3.1-8b-instruct`, which judged 5/5 smoke pairs as reject
(accuracy 0.800, **kappa 0.000**), demonstrating that kappa catches a
constant-reject predictor that accuracy alone would flatter.

`api` selects the wire protocol: `openai` (OpenAI-compatible chat
completions) or `anthropic` (the messages API, which takes the system
prompt as a top-level field and requires `max_tokens`). Both send
temperature 0 and the same versioned judge prompt.

## Eval-only backends

Backends marked `eval_only = true` (e.g. NVIDIA build.nvidia.com free tier)
are for **evaluation/testing only** per their terms of service. They must
never serve production traffic; the winning judge runs on a paid host.

## Repo hygiene

- No API keys anywhere in the repo — environment variables only.
- No vendor-derived titles committed — calibration data stays outside the
  repo and is passed in via `--data`.

## Development

```sh
uv run pytest
```

Tests run on tiny synthetic fixtures; no live API calls.
