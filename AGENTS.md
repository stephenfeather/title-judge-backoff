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
| `run_bakeoff.py` (pure helpers) | `tests/test_run_bakeoff.py` |
| `score.py` (metrics, leaderboard) | `tests/test_score.py` |
