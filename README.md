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
3. **Verdicts** — every backend judges every pair at temperature 0 with the
   versioned prompt in `judge/prompts.py`. Results land in
   `results/<date>/<backend>.jsonl`, one verdict per line, tagged with
   `model_id`, `prompt_version`, and `temperature`.
4. **Leaderboard** — `score.py` computes accuracy, Cohen's kappa,
   false-approve rate, and per-reason confusion, then emits `leaderboard.md`
   sorted by kappa.

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

Validate the slate before spending anything:

```sh
uv run run_bakeoff.py --check-backends          # key presence only, no network
uv run run_bakeoff.py --check-backends --ping   # also sends ONE real request per backend
```

`--ping` spends real tokens on paid backends — use it deliberately.

## Usage

```sh
uv run run_bakeoff.py --backends backends.toml --data /path/to/pairs.jsonl --out results/2026-08-05/
uv run score.py --data /path/to/pairs.jsonl --results results/2026-08-05/ --out leaderboard.md
```

Runs are resumable: re-running `run_bakeoff.py` skips pairs a backend has
already judged. Calls are rate-limited per backend (`rpm` in
`backends.toml`).

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
