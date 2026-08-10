# Codex Adversarial Review

Target: working tree diff
Verdict: needs-attention

Do not ship yet. The suite passes, but the implementation has a crash-resume write-loss window, lazy host-limit registration that can exceed the declared quota, and unbounded Retry-After parsing that can stall a host indefinitely.

Findings:
- [high] ResultWriter acknowledges before the result is flushed (run_bakeoff.py:260-267)
  `ResultWriter.write()` only queues the line and returns; the worker immediately proceeds to the next API call. At `--concurrency 1`, this no longer matches the old serial `write` + `flush` before the next request. If the process is killed after `client.judge()` succeeds but before the writer drains the queue item, the billed verdict is absent from the JSONL file and resume re-judges the same `(pair_id, run_index)`. Higher concurrency can lose several queued results in the same way.
  Recommendation: Make `write()` wait for an acknowledgement emitted only after the writer has written and flushed the line, or otherwise enforce a completion barrier before a worker task finishes. Add an interruption/resume regression test.
- [high] Lazy limiter registration permits an early host-rate burst (run_bakeoff.py:437-457)
  `run_slate()` starts backend tasks before all backends have registered with `LimiterRegistry`. A higher-RPM backend can register and send calls before a lower-RPM backend on the same host reaches `for_backend()`, so those calls use the higher interval even though the lower rate is declared in the slate. The same occurs at `--concurrency 1`: the first backend can complete before a later backend tightens the shared limiter. This can exceed provider quota and cause 429s and incomplete results.
  Recommendation: Register every runnable backend with the shared limiter before starting any backend work, or construct and validate the registry from the complete slate before submitting workers.
- [medium] Malformed Retry-After values can permanently stall a host (judge/client.py:192-200)
  `rate_limit_penalty()` accepts any value that `float()` parses, without checking finiteness or sign. A malformed `Retry-After: inf` sets the limiter deadline to infinity; subsequent `wait()` calls calculate an infinite delay while holding the shared limiter lock, blocking every worker and backend using that host indefinitely. `nan` and negative values also violate the intended backoff semantics.
  Recommendation: Accept only finite, non-negative delta-seconds, reject malformed values, and apply a bounded maximum before calling `penalize()`.

Next steps:
- Add a flushed-write acknowledgement and test kill/resume behavior.
- Pre-register all host limiters before any backend starts.
- Harden Retry-After parsing with finite, non-negative, bounded validation.
