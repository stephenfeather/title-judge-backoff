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
- **Synthetic fixtures only.** Invented titles (e.g. "Acme Widget 3000").
  Never put vendor-derived titles or real calibration pairs in tests —
  `*.jsonl` is gitignored for a reason, and this repo may go public.
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
| `run_bakeoff.py` (pure helpers) | `tests/test_run_bakeoff.py` |
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
