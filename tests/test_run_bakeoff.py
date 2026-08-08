import json
import random
import threading
import time
from dataclasses import replace

import httpx
import pytest

from judge.client import RETRY_AFTER_DEFAULT_S, Backend, LimiterRegistry
from judge.prompts import PROMPT_VERSION
from judge.schema import Pair, ReasonCode, Usage, Verdict, verdict_to_json_line
from run_bakeoff import (
    ResultWriter,
    already_judged_ids,
    pending_votes,
    render_skip_warning,
    run_backend,
    run_manifest,
    run_slate,
    skipped_backend_names,
)


def make_pair(pair_id):
    return Pair(
        id=pair_id,
        original=f"orig {pair_id}",
        enriched=f"enriched {pair_id}",
        brand="Acme",
        mpn=f"MPN-{pair_id}",
        ground_truth="approve",
        reason=ReasonCode.OK,
    )


def make_verdict(
    pair_id,
    model_id="m",
    prompt_version="v1",
    temperature=0.0,
    run_index=0,
    reasoning_effort=None,
    base_url=None,
    config_digest=None,
    code_version=None,
):
    return Verdict(
        pair_id=pair_id,
        verdict="approve",
        reason=ReasonCode.OK,
        model_id=model_id,
        prompt_version=prompt_version,
        temperature=temperature,
        run_index=run_index,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        config_digest=config_digest,
        code_version=code_version,
    )


CONFIG = dict(model_id="m", prompt_version="v1", temperature=0.0, reasoning_effort=None)

# What a provenance-aware caller (run_backend, post-#13) passes down.
PROVENANCE = dict(
    base_url="https://host-a.test/v1",
    config_digest="dig000000001",
    code_version="9b0d01a1c2d3",
)
FULL_CONFIG = {**CONFIG, **PROVENANCE}


def write_manifest(results_path, **overrides):
    """The sidecar run_manifest already written next to every results file."""
    manifest = {"backend": "backend", "model_id": "m", "base_url": "https://host-a.test/v1"}
    manifest.update(overrides)
    results_path.with_name(results_path.stem + ".manifest.json").write_text(json.dumps(manifest))


def test_already_judged_ids_resumes_when_provenance_matches(tmp_path):
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1", **PROVENANCE)) + "\n")
    assert already_judged_ids(out, **FULL_CONFIG) == {("p1", 0)}


def test_already_judged_ids_rejects_a_different_host(tmp_path):
    # Failure 1: two providers serving one model_id string, mixed into one file.
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1", **PROVENANCE)) + "\n")
    with pytest.raises(ValueError, match="base_url"):
        already_judged_ids(out, **{**FULL_CONFIG, "base_url": "https://host-b.test/v1"})


def test_already_judged_ids_rejects_a_different_code_version(tmp_path):
    # Failure 2: the #10 near-miss — unmerged code appending into a main-built file.
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1", **PROVENANCE)) + "\n")
    with pytest.raises(ValueError, match="code_version"):
        already_judged_ids(out, **{**FULL_CONFIG, "code_version": "deadbeef1234"})


def test_already_judged_ids_rejects_a_different_request_config(tmp_path):
    # Failure 3: every named field matches, but `api` or `structured_output`
    # differs. The row cannot name what changed, so it points at the manifest.
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1", **PROVENANCE)) + "\n")
    with pytest.raises(ValueError, match="manifest"):
        already_judged_ids(out, **{**FULL_CONFIG, "config_digest": "dig000000002"})


def test_already_judged_ids_uses_the_manifest_for_legacy_rows(tmp_path):
    # Rows predating #13 carry no host, but the sidecar manifest does. Using it
    # is the difference between a real check and a free pass.
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1")) + "\n")
    write_manifest(out, base_url="https://host-a.test/v1")
    assert already_judged_ids(out, **FULL_CONFIG) == {("p1", 0)}


def test_already_judged_ids_rejects_legacy_rows_whose_manifest_names_another_host(tmp_path):
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1")) + "\n")
    write_manifest(out, base_url="https://host-b.test/v1")
    with pytest.raises(ValueError, match="base_url"):
        already_judged_ids(out, **FULL_CONFIG)


def test_already_judged_ids_finds_the_manifest_for_a_dotted_backend_name(tmp_path):
    # Real slate names carry dots: anthropic-haiku-4.5, gemini-3.6-flash. If the
    # manifest path were derived by stripping at the first dot, every legacy file
    # for those backends would lose its evidence and be refused.
    out = tmp_path / "anthropic-haiku-4.5.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1")) + "\n")
    (tmp_path / "anthropic-haiku-4.5.manifest.json").write_text(
        json.dumps({"base_url": "https://host-a.test/v1"})
    )
    assert already_judged_ids(out, **FULL_CONFIG) == {("p1", 0)}


def test_already_judged_ids_refuses_unprovable_files(tmp_path):
    # No provenance on the rows AND no manifest: nothing can establish that this
    # file is compatible. Refuse rather than resume on hope.
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1")) + "\n")
    with pytest.raises(ValueError, match="allow-unknown-provenance"):
        already_judged_ids(out, **FULL_CONFIG)


def test_already_judged_ids_resumes_unprovable_files_when_explicitly_allowed(tmp_path):
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1")) + "\n")
    assert already_judged_ids(out, **FULL_CONFIG, allow_unknown_provenance=True) == {("p1", 0)}


def test_already_judged_ids_keys_on_pair_and_run_index(tmp_path):
    # Resume identity is (pair_id, run_index), not pair_id: under majority
    # voting the same pair is judged N times on purpose, so keying on pair_id
    # alone would treat vote 1 as "already done" and silently run N=1.
    out = tmp_path / "backend.jsonl"
    out.write_text(
        verdict_to_json_line(make_verdict("p1", run_index=0))
        + "\n"
        + verdict_to_json_line(make_verdict("p1", run_index=1))
        + "\n"
        + verdict_to_json_line(make_verdict("p3", run_index=0))
        + "\n"
    )
    assert already_judged_ids(out, **CONFIG) == {("p1", 0), ("p1", 1), ("p3", 0)}


def test_run_backend_refuses_to_resume_a_file_from_another_host(tmp_path):
    # The guard is only worth anything if run_backend actually passes provenance
    # down; without the wiring every unit test above passes and the sweep still
    # mixes providers.
    backend = Backend(
        name="backend",
        base_url="https://host-b.test/v1",
        model_id="m",
        rpm=6000,
        eval_only=False,
        api_key_env="NVIDIA_API_KEY",
        temperature=0.0,
    )
    results = tmp_path / "backend.jsonl"
    results.write_text(
        verdict_to_json_line(
            make_verdict("p1", prompt_version=PROMPT_VERSION, base_url="https://host-a.test/v1")
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="base_url"):
        run_backend(
            backend,
            [make_pair("p1")],
            tmp_path,
            votes=1,
            code_version="9b0d01a1c2d3",
            client_factory=lambda b: FakeClient(b),
        )


def test_run_backend_records_the_code_version_in_the_manifest(tmp_path):
    backend = Backend(
        name="backend",
        base_url="https://host-a.test/v1",
        model_id="m",
        rpm=6000,
        eval_only=False,
        api_key_env="NVIDIA_API_KEY",
        temperature=0.0,
    )
    run_backend(
        backend,
        [make_pair("p1")],
        tmp_path,
        votes=1,
        code_version="9b0d01a1c2d3",
        client_factory=lambda b: FakeClient(b),
    )
    manifest = json.loads((tmp_path / "backend.manifest.json").read_text())
    assert manifest["code_version"] == "9b0d01a1c2d3"


def test_already_judged_ids_missing_file_is_empty(tmp_path):
    assert already_judged_ids(tmp_path / "nope.jsonl", **CONFIG) == set()


def test_already_judged_ids_rejects_stale_prompt_version(tmp_path):
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1", prompt_version="v1")) + "\n")
    with pytest.raises(ValueError, match="prompt_version"):
        already_judged_ids(out, **{**CONFIG, "prompt_version": "v2"})


def test_already_judged_ids_rejects_stale_model_or_temperature(tmp_path):
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1")) + "\n")
    with pytest.raises(ValueError, match="different run config"):
        already_judged_ids(out, **{**CONFIG, "model_id": "other-model"})
    with pytest.raises(ValueError, match="different run config"):
        already_judged_ids(out, **{**CONFIG, "temperature": 0.7})


def test_already_judged_ids_rejects_stale_reasoning_effort(tmp_path):
    # effort=none and effort=medium gave different reason codes on the same
    # borderline pair, so verdicts from two efforts are not interchangeable.
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1", reasoning_effort="none")) + "\n")
    with pytest.raises(ValueError, match="different run config"):
        already_judged_ids(out, **{**CONFIG, "reasoning_effort": "medium"})


def test_already_judged_ids_distinguishes_omitted_temperature_from_zero(tmp_path):
    # temperature=None (field omitted) and temperature=0.0 (field sent) are
    # different run configs; merging them would mix sampled with greedy.
    out = tmp_path / "backend.jsonl"
    out.write_text(verdict_to_json_line(make_verdict("p1", temperature=None)) + "\n")
    with pytest.raises(ValueError, match="different run config"):
        already_judged_ids(out, **{**CONFIG, "temperature": 0.0})
    assert already_judged_ids(out, **{**CONFIG, "temperature": None}) == {("p1", 0)}


def test_pending_votes_expands_each_pair_to_n_votes():
    pairs = [make_pair("p1"), make_pair("p2")]
    assert [(p.id, r) for p, r in pending_votes(pairs, set(), votes=3)] == [
        ("p1", 0),
        ("p1", 1),
        ("p1", 2),
        ("p2", 0),
        ("p2", 1),
        ("p2", 2),
    ]


def test_pending_votes_skips_only_the_votes_already_done():
    # A run interrupted after 2 of 3 votes on p1 resumes at vote 2, rather than
    # re-judging p1 from scratch or skipping it entirely.
    pairs = [make_pair("p1"), make_pair("p2")]
    done = {("p1", 0), ("p1", 1), ("p2", 0)}
    assert [(p.id, r) for p, r in pending_votes(pairs, done, votes=3)] == [
        ("p1", 2),
        ("p2", 1),
        ("p2", 2),
    ]


def test_run_manifest_records_provenance():
    # The manifest is the human-readable expansion the row's config_digest
    # points at, so a digest mismatch can be diagnosed instead of just refused.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend,
        votes=3,
        prompt_version="v1",
        n_pairs=200,
        sample_payload={"model": "m"},
        observed_models=set(),
        code_version="9b0d01a1c2d3",
        config_digest="dig000000001",
    )
    assert manifest["code_version"] == "9b0d01a1c2d3"
    assert manifest["config_digest"] == "dig000000001"
    # base_url was already recorded; the guard's legacy fallback depends on it.
    assert manifest["base_url"] == "https://example.test/v1"


def test_run_manifest_records_payload_effort_and_snapshots():
    # R7: Responses has no system_fingerprint, so the manifest is the only
    # record of what was actually sent and which snapshot answered.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
        reasoning_effort="medium",
    )
    manifest = run_manifest(
        backend,
        votes=3,
        prompt_version="v1",
        n_pairs=200,
        sample_payload={"model": "m", "reasoning_effort": "medium"},
        observed_models={"m-2026-06-30"},
    )
    assert manifest["backend"] == "nv"
    assert manifest["model_id"] == "m"
    assert manifest["reasoning_effort"] == "medium"
    assert manifest["temperature"] is None
    assert manifest["votes"] == 3
    assert manifest["prompt_version"] == "v1"
    assert manifest["n_pairs"] == 200
    assert manifest["request_payload"] == {"model": "m", "reasoning_effort": "medium"}
    assert manifest["observed_models"] == ["m-2026-06-30"]


def test_skipped_backend_names_lists_only_those_missing_keys(monkeypatch):
    monkeypatch.setenv("HAVE_KEY", "x")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    backends = [
        Backend(
            name="has-key",
            base_url="https://example.test/v1",
            model_id="m1",
            rpm=40,
            eval_only=True,
            api_key_env="HAVE_KEY",
        ),
        Backend(
            name="no-key",
            base_url="https://example.test/v1",
            model_id="m2",
            rpm=40,
            eval_only=True,
            api_key_env="MISSING_KEY",
        ),
    ]
    assert skipped_backend_names(backends) == ["no-key"]


def test_skip_warning_names_the_backends_and_the_env_prefix():
    # The silent-skip trap: a missing key does not fail, the backend just
    # vanishes from the run. On a multi-hour unattended sweep that is only
    # discovered at the report, so the warning has to say what to DO.
    warning = render_skip_warning(["anthropic-haiku-4.5", "gemini-3.6-flash"])
    assert "anthropic-haiku-4.5" in warning
    assert "gemini-3.6-flash" in warning
    assert "042_env_ai_tokens" in warning
    assert "002_functions" in warning
    assert "--allow-skipped" in warning
    # S4 operational health: which backends are slow, which are flaky. A ping
    # gives one sample; a 600-call run gives a distribution, and that is what
    # decides whether a backend is usable at full-pack scale.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend,
        votes=3,
        prompt_version="v1",
        n_pairs=200,
        sample_payload={},
        observed_models=set(),
        latencies=[1.0, 2.0, 3.0, 100.0],
        errors=["timeout", "timeout", "500"],
    )
    health = manifest["health"]
    assert health["calls_ok"] == 4
    assert health["calls_failed"] == 3
    assert health["latency_min"] == 1.0
    assert health["latency_max"] == 100.0
    assert health["latency_median"] == pytest.approx(2.5)
    assert health["error_kinds"] == {"timeout": 2, "500": 1}


def test_health_reports_failure_latencies_separately_from_success():
    # Failed calls are typically the SLOWEST — a 180s timeout is the whole
    # reason we care. Recording elapsed only on success made the worst latencies
    # vanish from the health block, so a backend that timed out on every call
    # looked instantaneous. They are kept separate so a slow failure cannot be
    # mistaken for a slow success.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend,
        votes=1,
        prompt_version="v1",
        n_pairs=3,
        sample_payload={},
        observed_models=set(),
        latencies=[1.0, 2.0],
        errors=["ReadTimeout", "ReadTimeout"],
        failed_latencies=[180.0, 180.0],
    )
    health = manifest["health"]
    assert health["latency_max"] == 2.0, "success latencies must not absorb failures"
    assert health["failed_latency_median"] == 180.0
    assert health["failed_latency_max"] == 180.0


def test_health_failure_latencies_are_none_when_nothing_failed():
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    health = run_manifest(
        backend,
        votes=1,
        prompt_version="v1",
        n_pairs=1,
        sample_payload={},
        observed_models=set(),
        latencies=[1.0],
        errors=[],
    )["health"]
    assert health["failed_latency_median"] is None
    assert health["failed_latency_max"] is None


def test_run_manifest_handles_a_backend_that_never_succeeded():
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend,
        votes=1,
        prompt_version="v1",
        n_pairs=5,
        sample_payload={},
        observed_models=set(),
        latencies=[],
        errors=["timeout"],
    )
    health = manifest["health"]
    assert health["calls_ok"] == 0
    assert health["latency_median"] is None


def test_manifest_totals_the_tokens_the_run_actually_spent():
    # Issue #11: cost modelling had to ESTIMATE prompt tokens from an unrelated
    # metering call because no run recorded a single count. The manifest is
    # where a cost model looks, so the per-call captures are summed here.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend,
        votes=1,
        prompt_version="v1",
        n_pairs=2,
        sample_payload={},
        observed_models=set(),
        usages=[
            Usage(prompt_tokens=400, completion_tokens=20, total_tokens=420, reasoning_tokens=120),
            Usage(prompt_tokens=410, completion_tokens=30, total_tokens=440, reasoning_tokens=100),
        ],
    )
    tokens = manifest["usage"]
    assert tokens["prompt_tokens"] == 810
    assert tokens["completion_tokens"] == 50
    assert tokens["total_tokens"] == 860
    # Reasoning bills as output and roughly doubles a reasoning backend's cost,
    # so it is reported separately rather than buried in completion_tokens.
    assert tokens["reasoning_tokens"] == 220
    assert tokens["calls_measured"] == 2
    assert tokens["calls_unmeasured"] == 0


def test_manifest_counts_calls_the_host_reported_nothing_for():
    # A partly-instrumented host must not look fully measured — a total summed
    # over 2 of 600 calls is worse than no total, because it looks like one.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend,
        votes=1,
        prompt_version="v1",
        n_pairs=3,
        sample_payload={},
        observed_models=set(),
        usages=[Usage(prompt_tokens=400, completion_tokens=20, total_tokens=420), None, None],
    )
    assert manifest["usage"]["calls_measured"] == 1
    assert manifest["usage"]["calls_unmeasured"] == 2


def test_manifest_surfaces_cache_hits_because_they_would_fake_stability():
    # The three votes are byte-identical requests. A host serving them from
    # cache collapses majority-of-3 to n=1 and drives flip rate to a spurious
    # 0.0 — metrics that IMPROVE while measuring nothing. cached_tokens is the
    # only in-band tell, so the manifest counts the calls that had one.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend,
        votes=1,
        prompt_version="v1",
        n_pairs=3,
        sample_payload={},
        observed_models=set(),
        usages=[
            Usage(prompt_tokens=400, completion_tokens=20, total_tokens=420, cached_tokens=384),
            Usage(prompt_tokens=400, completion_tokens=20, total_tokens=420, cached_tokens=0),
            Usage(prompt_tokens=400, completion_tokens=20, total_tokens=420),
        ],
    )
    assert manifest["usage"]["cached_tokens"] == 384
    assert manifest["usage"]["calls_with_cache_hit"] == 1


def test_manifest_usage_is_absent_rather_than_zeroed_when_nothing_reported():
    # Zeros would read as a measured free run.
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
    )
    manifest = run_manifest(
        backend, votes=1, prompt_version="v1", n_pairs=1, sample_payload={}, observed_models=set()
    )
    assert manifest["usage"]["total_tokens"] is None
    assert manifest["usage"]["calls_measured"] == 0


def test_run_manifest_omits_nothing_when_temperature_is_set():
    backend = Backend(
        name="nv",
        base_url="https://example.test/v1",
        model_id="m",
        rpm=40,
        eval_only=True,
        api_key_env="NVIDIA_API_KEY",
        temperature=0.0,
    )
    manifest = run_manifest(
        backend, votes=1, prompt_version="v1", n_pairs=5, sample_payload={}, observed_models=set()
    )
    assert manifest["temperature"] == 0.0
    assert manifest["observed_models"] == []


# --------------------------------------------------------------------------
# Concurrency (issue #9)
# --------------------------------------------------------------------------


def concurrency_backend(name="nv", base_url="https://integrate.api.nvidia.com/v1", rpm=6000):
    # rpm high enough that the limiter is never the thing under test here;
    # spacing itself is covered in tests/test_client.py.
    return Backend(
        name=name,
        base_url=base_url,
        model_id=f"model/{name}",
        rpm=rpm,
        eval_only=False,
        api_key_env="NVIDIA_API_KEY",
    )


class FakeClient:
    """Stands in for JudgeClient: records calls, optionally slow, optionally fails."""

    def __init__(self, backend, *, delay=0.0, fail_pair_ids=()):
        self.backend = backend
        self.observed_models = {backend.model_id}
        self.delay = delay
        self.fail_pair_ids = set(fail_pair_ids)
        self.calls = []
        self.closed = False
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def request_body(self, pair):
        return {"model": self.backend.model_id}

    def judge(self, pair, run_index=0):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.calls.append((pair.id, run_index))
        try:
            if self.delay:
                time.sleep(self.delay)
            if pair.id in self.fail_pair_ids:
                raise RuntimeError(f"backend exploded on {pair.id}")
            return Verdict(
                pair_id=pair.id,
                verdict="approve",
                reason=ReasonCode.OK,
                model_id=self.backend.model_id,
                prompt_version=PROMPT_VERSION,
                temperature=self.backend.temperature,
                run_index=run_index,
                reasoning_effort=self.backend.reasoning_effort,
            )
        finally:
            with self._lock:
                self._in_flight -= 1

    def close(self):
        self.closed = True


class RateLimitedClient(FakeClient):
    """Always 429s, the way an over-subscribed shared host does."""

    def judge(self, pair, run_index=0):
        request = httpx.Request("POST", f"{self.backend.base_url}/chat/completions")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)


class MixedFailureClient(FakeClient):
    """Fails with a different exception type per pair, to exercise error_kinds."""

    _KINDS = (ValueError, RuntimeError, TypeError, KeyError, ArithmeticError, OSError)

    def judge(self, pair, run_index=0):
        raise self._KINDS[int(pair.id.removeprefix("p")) % len(self._KINDS)](pair.id)


def no_sleep_limiters():
    return LimiterRegistry(sleep=lambda _: None)


def read_verdicts(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def vote_keys(path):
    return {(v["pair_id"], v["run_index"]) for v in read_verdicts(path)}


def test_result_writer_keeps_lines_intact_under_concurrent_writers(tmp_path):
    # The append+flush contract is what makes resume work. Many threads, ONE
    # handle: no interleaved or partial lines, nothing dropped.
    path = tmp_path / "out.jsonl"
    n_threads, per_thread = 8, 50
    with ResultWriter(path) as writer:
        def worker(w):
            for i in range(per_thread):
                writer.write(f"worker-{w}-line-{i}")

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    lines = path.read_text().splitlines()
    assert len(lines) == n_threads * per_thread
    assert set(lines) == {
        f"worker-{w}-line-{i}" for w in range(n_threads) for i in range(per_thread)
    }


def test_result_writer_does_not_acknowledge_a_write_before_it_is_flushed(tmp_path):
    # #10's central claim is that --concurrency 1 reproduces prior runs
    # exactly. The old serial loop flushed before issuing the next request, so
    # a kill could lose at most the call in flight. An async queue widens that
    # to "whatever is still queued" — verdicts already PAID FOR that resume
    # then re-judges. Small, but it undercuts the reproducibility claim that
    # makes this PR safe to merge, so write() waits for the flush.
    path = tmp_path / "out.jsonl"
    with ResultWriter(path) as writer:
        writer.write("first")
        # No join and no sleep: if write() acknowledged early, this read would
        # see an empty file.
        assert path.read_text() == "first\n"
        writer.write("second")
        assert path.read_text() == "first\nsecond\n"

        # And in bulk, where an async writer would fall behind for certain.
        for i in range(200):
            writer.write(f"bulk-{i}")
        assert len(path.read_text().splitlines()) == 202


def test_result_writer_appends_rather_than_truncating(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text("pre-existing\n")
    with ResultWriter(path) as writer:
        writer.write("new")
    assert path.read_text().splitlines() == ["pre-existing", "new"]


class _FailingFile:
    """A file handle whose writes start failing partway through — a full disk."""

    def __init__(self, real, fail_after):
        self._real = real
        self._fail_after = fail_after
        self._writes = 0

    def write(self, data):
        self._writes += 1
        if self._writes > self._fail_after:
            raise OSError(28, "No space left on device")
        return self._real.write(data)

    def flush(self):
        return self._real.flush()

    def close(self):
        return self._real.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._real.close()
        return False


class FullDiskWriter(ResultWriter):
    FAIL_AFTER = 2

    def _open(self):
        return _FailingFile(open(self._path, "a"), self.FAIL_AFTER)


def test_a_dead_writer_stops_the_workers_instead_of_burning_the_rest_of_the_run(tmp_path):
    # P1. If persistence dies mid-run, continuing to call the API spends real
    # money producing verdicts that are then discarded. Surfacing the error at
    # the END of a multi-hour sweep is barely better than losing it silently,
    # because the quota is already gone. Workers must notice and stop.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i:03d}") for i in range(60)]
    # A per-call delay is not padding — it is the realistic case. A judge call
    # takes seconds; the writer's local write+flush takes microseconds, so the
    # writer is always far ahead and the guard sees the failure on the very
    # next call. Without a delay the fake outruns its own writer by ~40 calls,
    # which measures the fake rather than the guard.
    client = FakeClient(backend, delay=0.002)

    with pytest.raises(OSError):
        run_backend(
            backend,
            pairs,
            tmp_path,
            votes=1,
            concurrency=1,
            limiters=no_sleep_limiters(),
            client_factory=lambda b: client,
            writer_factory=FullDiskWriter,
        )

    # The writer dies on write 3, so calls 1-3 are already paid for and one or
    # two may be in flight. Anything approaching 60 means the guard does nothing.
    assert len(client.calls) <= 8, f"kept calling the API after the writer died: {len(client.calls)}"


def test_result_writer_reraises_a_failure_from_its_own_thread(tmp_path):
    # A writer that dies on a full disk would otherwise drop every verdict
    # after it in silence, and a multi-hour run would look like it simply
    # judged fewer pairs. The failure has to reach the caller's thread.
    unwritable = tmp_path / "nope" / "out.jsonl"  # parent does not exist
    with pytest.raises(FileNotFoundError), ResultWriter(unwritable) as writer:
        writer.write("lost")


def test_run_slate_with_no_runnable_backends_is_a_no_op(tmp_path):
    # Every backend skipped for a missing key is a normal outcome under
    # --allow-skipped; it must not trip the thread pool.
    run_slate([], [make_pair("p0")], tmp_path, votes=1, concurrency=4)
    assert list(tmp_path.iterdir()) == []


def test_run_backend_at_concurrency_1_writes_in_pair_then_vote_order(tmp_path):
    # Decision 6: the default must reproduce prior runs exactly, so line order
    # at --concurrency 1 stays pair-then-vote, byte for byte.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(4)]
    client = FakeClient(backend)
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=2,
        concurrency=1,
        limiters=no_sleep_limiters(),
        client_factory=lambda b: client,
    )
    written = [(v["pair_id"], v["run_index"]) for v in read_verdicts(tmp_path / "nv.jsonl")]
    assert written == [(p.id, r) for p in pairs for r in range(2)]
    assert client.max_in_flight == 1


def test_run_backend_concurrent_produces_the_same_verdict_set(tmp_path):
    # Acceptance 1: identical verdict SETS at concurrency 1 vs N, modulo order.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(20)]

    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()

    for out_dir, concurrency in ((serial_dir, 1), (parallel_dir, 8)):
        run_backend(
            backend,
            pairs,
            out_dir,
            votes=3,
            concurrency=concurrency,
            limiters=no_sleep_limiters(),
            client_factory=lambda b: FakeClient(b, delay=0.001),
        )

    assert vote_keys(serial_dir / "nv.jsonl") == vote_keys(parallel_dir / "nv.jsonl")
    assert len(read_verdicts(parallel_dir / "nv.jsonl")) == 60


def test_run_backend_actually_runs_workers_in_parallel(tmp_path):
    # A latency-bound backend must spend its rate budget: with 8 workers and a
    # per-call delay, calls overlap instead of queueing one at a time.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(16)]
    client = FakeClient(backend, delay=0.02)
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=1,
        concurrency=8,
        limiters=no_sleep_limiters(),
        client_factory=lambda b: client,
    )
    assert client.max_in_flight > 1, "workers never overlapped — still serial"


def test_run_backend_spends_the_rate_budget_when_latency_bound(tmp_path):
    # Acceptance 3. Serial throughput is 1/max(60/rpm, latency), so a backend
    # whose latency exceeds its rate interval never spends the budget it was
    # already granted — grok ran at 14% of its declared rpm this way. With the
    # limiter out of the way, N workers must cut wall clock by roughly N.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(24)]

    def elapsed_at(concurrency, out_dir):
        out_dir.mkdir()
        started = time.monotonic()
        run_backend(
            backend,
            pairs,
            out_dir,
            votes=1,
            concurrency=concurrency,
            limiters=no_sleep_limiters(),
            client_factory=lambda b: FakeClient(b, delay=0.02),
        )
        return time.monotonic() - started

    serial = elapsed_at(1, tmp_path / "serial")
    parallel = elapsed_at(8, tmp_path / "parallel")
    assert parallel < serial / 3, f"8 workers took {parallel:.2f}s vs {serial:.2f}s serial"


def test_run_backend_still_cannot_outrun_the_host_rate_limit(tmp_path):
    # The safety property that makes raising --concurrency sane. Workers only
    # overlap the WAITING; departures stay one interval apart, so 8 threads on a
    # 40 rpm host still send 40 rpm. Real limiter, real sleeps, on purpose.
    interval = 0.025
    backend = concurrency_backend(rpm=int(60 / interval))
    pairs = [make_pair(f"p{i}") for i in range(8)]
    started = time.monotonic()
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=1,
        concurrency=8,
        limiters=LimiterRegistry(),
        client_factory=FakeClient,
    )
    elapsed = time.monotonic() - started
    # 8 calls one interval apart occupy at least 7 intervals however many
    # threads are waiting on them.
    assert elapsed >= 7 * interval * 0.9, f"8 calls left in {elapsed:.3f}s — rate limit bypassed"


def test_a_429_from_one_worker_holds_back_the_whole_host(tmp_path):
    # The concurrency-specific failure: with per-worker backoff the other seven
    # workers keep hammering a host that just said stop. The penalty has to
    # land on the shared bucket, so it must be visible to a DIFFERENT backend
    # on the same host.
    penalties = []
    backend = concurrency_backend("nv-a", base_url="https://integrate.api.nvidia.com/v1")
    other = concurrency_backend("nv-b", base_url="https://integrate.api.nvidia.com/v1")
    registry = no_sleep_limiters()
    shared = registry.for_backend(other)
    shared.penalize = lambda seconds: penalties.append(seconds)  # type: ignore[method-assign]

    pairs = [make_pair("p0")]
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=1,
        concurrency=1,
        limiters=registry,
        client_factory=lambda b: RateLimitedClient(b),
    )
    assert penalties == [RETRY_AFTER_DEFAULT_S], "the shared host bucket was never penalized"


def test_a_non_429_failure_does_not_hold_back_the_host(tmp_path):
    # A 500 says "this call broke", not "you are going too fast". Backing the
    # whole host off for every transient error would throttle the run to
    # nothing on a flaky backend.
    penalties = []
    backend = concurrency_backend()
    registry = no_sleep_limiters()
    registry.for_backend(backend).penalize = lambda s: penalties.append(s)  # type: ignore[method-assign]
    run_backend(
        backend,
        [make_pair("p0")],
        tmp_path,
        votes=1,
        concurrency=1,
        limiters=registry,
        client_factory=lambda b: FakeClient(b, fail_pair_ids=["p0"]),
    )
    assert penalties == []


def test_health_error_kinds_are_reported_in_a_stable_order(tmp_path):
    # Concurrency makes the ARRIVAL order of errors arbitrary, and
    # dict(Counter(...)) preserves insertion order — so the manifest's
    # error_kinds keys would shuffle between otherwise identical runs.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(6)]
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=1,
        concurrency=6,
        limiters=no_sleep_limiters(),
        client_factory=lambda b: MixedFailureClient(b),
    )
    error_kinds = json.loads((tmp_path / "nv.manifest.json").read_text())["health"]["error_kinds"]
    assert list(error_kinds) == sorted(error_kinds)


def test_an_interrupt_stops_scheduling_rather_than_draining_the_queue(tmp_path):
    # Every vote is submitted up front, so a plain shutdown(wait=True) would
    # keep sending the remaining PAID requests — potentially for hours — before
    # the interrupt propagated. Two runs were killed by hand on 2026-08-06; an
    # operator pressing Ctrl-C expects the spend to stop.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i:03d}") for i in range(80)]

    class InterruptingClient(FakeClient):
        def judge(self, pair, run_index=0):
            # super() records the call, so counting here too would double it.
            result = super().judge(pair, run_index)
            if len(self.calls) == 3:
                raise KeyboardInterrupt("operator pressed ctrl-c")
            return result

    # As above, a per-call delay is the realistic shape: cancellation only has
    # to outrun the NEXT call, which in production is seconds away.
    client = InterruptingClient(backend, delay=0.002)
    with pytest.raises(KeyboardInterrupt):
        run_backend(
            backend,
            pairs,
            tmp_path,
            votes=1,
            concurrency=1,
            limiters=no_sleep_limiters(),
            client_factory=lambda b: client,
        )
    # KeyboardInterrupt is a BaseException, so judge_one's `except Exception`
    # does not swallow it — it must reach the pool and cancel what is pending.
    assert len(client.calls) < 20, f"kept spending after the interrupt: {len(client.calls)}"


def test_run_backend_resume_writes_no_duplicate_votes(tmp_path):
    # Acceptance 2: resume after a mid-run kill yields zero duplicate
    # (pair_id, run_index). The todo list is partitioned ONCE up front; a worker
    # that re-scanned the results file could hand the same vote to two threads.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(10)]
    results_path = tmp_path / "nv.jsonl"

    # Simulate a run killed partway: the first 12 of 30 votes landed.
    already = [(p.id, r) for p in pairs for r in range(3)][:12]
    results_path.write_text(
        "".join(
            verdict_to_json_line(
                Verdict(
                    pair_id=pair_id,
                    verdict="approve",
                    reason=ReasonCode.OK,
                    model_id=backend.model_id,
                    prompt_version=PROMPT_VERSION,
                    temperature=backend.temperature,
                    run_index=run_index,
                    reasoning_effort=backend.reasoning_effort,
                )
            )
            + "\n"
            for pair_id, run_index in already
        )
    )

    client = FakeClient(backend, delay=0.001)
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=3,
        concurrency=8,
        limiters=no_sleep_limiters(),
        # The interrupted rows above carry no provenance and have no manifest
        # beside them, which is exactly the file #13 refuses by default. This
        # test is about duplicate votes, so it accepts the file explicitly.
        allow_unknown_provenance=True,
        client_factory=lambda b: client,
    )

    keys = [(v["pair_id"], v["run_index"]) for v in read_verdicts(results_path)]
    assert len(keys) == len(set(keys)) == 30
    # Resumed votes are judged exactly once, and the completed ones not at all.
    owed = {(p.id, r) for p in pairs for r in range(3)} - set(already)
    assert sorted(client.calls) == sorted(owed)


def test_run_backend_records_every_latency_under_concurrency(tmp_path):
    # Decision 7: per-call latencies are the evidence base for issue #9, so a
    # concurrent run must not drop samples through an unguarded list.append.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(25)]
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=2,
        concurrency=8,
        limiters=no_sleep_limiters(),
        client_factory=lambda b: FakeClient(b, delay=0.001),
    )
    health = json.loads((tmp_path / "nv.manifest.json").read_text())["health"]
    assert health["calls_ok"] == 50
    assert health["calls_failed"] == 0
    assert health["latency_median"] is not None


def test_a_run_writes_usage_to_both_the_row_and_the_manifest(tmp_path):
    # Wiring check. Every piece can be individually correct and still not be
    # connected: the point of #11 is that a real run leaves measured tokens on
    # disk, so this asserts against what actually lands there.
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(3)]

    class MeteredClient(FakeClient):
        def judge(self, pair, run_index=0):
            verdict = super().judge(pair, run_index)
            return replace(
                verdict,
                usage=Usage(
                    prompt_tokens=400,
                    completion_tokens=20,
                    total_tokens=420,
                    reasoning_tokens=120,
                ),
            )

    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=1,
        concurrency=2,
        limiters=no_sleep_limiters(),
        client_factory=MeteredClient,
    )

    rows = read_verdicts(tmp_path / "nv.jsonl")
    assert all(r["usage"]["prompt_tokens"] == 400 for r in rows)

    usage = json.loads((tmp_path / "nv.manifest.json").read_text())["usage"]
    assert usage["calls_measured"] == 3
    assert usage["calls_unmeasured"] == 0
    assert usage["total_tokens"] == 1260
    assert usage["reasoning_tokens"] == 360


def test_a_run_against_a_silent_host_records_no_tokens_rather_than_zero(tmp_path):
    # FakeClient reports no usage, like a host that omits the block.
    backend = concurrency_backend()
    run_backend(
        backend,
        [make_pair("p0")],
        tmp_path,
        votes=1,
        concurrency=1,
        limiters=no_sleep_limiters(),
        client_factory=FakeClient,
    )
    usage = json.loads((tmp_path / "nv.manifest.json").read_text())["usage"]
    assert usage["calls_measured"] == 0
    assert usage["calls_unmeasured"] == 1
    assert usage["total_tokens"] is None


def test_run_backend_records_failures_from_every_worker(tmp_path):
    backend = concurrency_backend()
    pairs = [make_pair(f"p{i}") for i in range(10)]
    doomed = [p.id for p in pairs[:4]]
    run_backend(
        backend,
        pairs,
        tmp_path,
        votes=2,
        concurrency=8,
        limiters=no_sleep_limiters(),
        client_factory=lambda b: FakeClient(b, delay=0.001, fail_pair_ids=doomed),
    )
    health = json.loads((tmp_path / "nv.manifest.json").read_text())["health"]
    assert health["calls_failed"] == 8
    assert health["calls_ok"] == 12
    assert health["error_kinds"] == {"RuntimeError": 8}
    # Failed votes are simply absent; resume covers them on the next launch.
    assert len(read_verdicts(tmp_path / "nv.jsonl")) == 12


def test_run_slate_at_concurrency_1_runs_backends_in_slate_order(tmp_path):
    # Decision 6 again, on the other axis: N=1 keeps backends serial.
    backends = [concurrency_backend(f"b{i}", rpm=6000) for i in range(3)]
    order = []
    clients = {}

    def factory(backend):
        order.append(("start", backend.name))
        client = FakeClient(backend, delay=0.005)
        clients[backend.name] = client
        return client

    run_slate(
        backends,
        [make_pair("p0")],
        tmp_path,
        votes=1,
        concurrency=1,
        limiters=no_sleep_limiters(),
        client_factory=factory,
    )
    assert order == [("start", "b0"), ("start", "b1"), ("start", "b2")]


def test_run_slate_registers_every_host_limiter_before_any_call_departs(tmp_path):
    # Raised by the adversarial review, and real. The registry only ever
    # TIGHTENS, but tightening is useless if it happens after the calls have
    # already left: for_backend() is called inside each backend's own task, so
    # a high-rpm backend can run at its own rate before a low-rpm sibling on
    # the same host has registered at all. At concurrency 1 the first backend
    # can finish its entire leg first. That is an over-rate burst on a shared
    # host — the exact 503-storm class of failure host-keying exists to stop.
    host = "https://integrate.api.nvidia.com/v1"
    fast = concurrency_backend("fast", base_url=host, rpm=6000)  # interval 0.01
    slow = concurrency_backend("slow", base_url=host, rpm=60)  # interval 1.0
    registry = no_sleep_limiters()
    intervals_in_force = []

    class IntervalSpyClient(FakeClient):
        def judge(self, pair, run_index=0):
            # White-box on purpose: the effective rate at the moment a call
            # departs is the property under test, and asserting it on wall
            # clock would be flaky.
            intervals_in_force.append(registry.for_backend(self.backend)._interval)
            return super().judge(pair, run_index)

    run_slate(
        [fast, slow],
        [make_pair("p0")],
        tmp_path,
        votes=1,
        concurrency=1,
        limiters=registry,
        client_factory=IntervalSpyClient,
    )
    # Both backends share one host, so both must depart at the tightest
    # declared rate — including the one that got there first.
    assert intervals_in_force == [1.0, 1.0]


def test_run_slate_does_not_let_one_slow_backend_delay_the_others(tmp_path):
    # Acceptance 4: a backend returning continuous 5xx must not hold up any
    # other backend's completion. The slow one is failing AND slow, which is
    # exactly the nemotron 503 case.
    slow = concurrency_backend("slow", base_url="https://slow.test/v1")
    fast_a = concurrency_backend("fast-a", base_url="https://fast-a.test/v1")
    fast_b = concurrency_backend("fast-b", base_url="https://fast-b.test/v1")
    pairs = [make_pair(f"p{i}") for i in range(4)]

    finished = []
    lock = threading.Lock()

    def factory(backend):
        delay = 0.05 if backend.name == "slow" else 0.0
        fail = [p.id for p in pairs] if backend.name == "slow" else []
        return FakeClient(backend, delay=delay, fail_pair_ids=fail)

    def record(name):
        with lock:
            finished.append(name)

    run_slate(
        backends=[slow, fast_a, fast_b],
        pairs=pairs,
        out_dir=tmp_path,
        votes=1,
        concurrency=2,
        limiters=no_sleep_limiters(),
        client_factory=factory,
        on_backend_done=record,
    )

    assert finished[-1] == "slow", f"fast backends waited on the slow one: {finished}"
    assert set(finished) == {"slow", "fast-a", "fast-b"}


def test_run_slate_isolates_a_backend_that_raises(tmp_path):
    # A stale-config ValueError in one backend must not cost the whole slate the
    # work the other backends already did — but it must still be reported.
    good = concurrency_backend("good", base_url="https://good.test/v1")
    bad = concurrency_backend("bad", base_url="https://bad.test/v1")
    pairs = [make_pair("p0")]

    def factory(backend):
        if backend.name == "bad":
            raise ValueError("stale run config")
        return FakeClient(backend)

    with pytest.raises(RuntimeError, match="bad"):
        run_slate(
            backends=[bad, good],
            pairs=pairs,
            out_dir=tmp_path,
            votes=1,
            concurrency=2,
            limiters=no_sleep_limiters(),
            client_factory=factory,
        )
    # The healthy backend still finished and its verdicts are on disk.
    assert len(read_verdicts(tmp_path / "good.jsonl")) == 1


def test_run_backend_shares_one_limiter_across_backends_on_a_host(tmp_path):
    # The registry is threaded through run_backend, not built per backend —
    # otherwise two NVIDIA legs would each get their own 40 rpm budget.
    registry = no_sleep_limiters()
    a = concurrency_backend("nv-a", base_url="https://integrate.api.nvidia.com/v1", rpm=6000)
    b = concurrency_backend("nv-b", base_url="https://integrate.api.nvidia.com/v1", rpm=6000)
    pairs = [make_pair("p0")]
    for backend in (a, b):
        run_backend(
            backend,
            pairs,
            tmp_path,
            votes=1,
            concurrency=1,
            limiters=registry,
            client_factory=FakeClient,
        )
    assert registry.for_backend(a) is registry.for_backend(b)


def test_already_judged_ids_is_independent_of_line_order(tmp_path):
    # Acceptance 5: concurrency makes line order arbitrary, so resume must not
    # care about it.
    verdicts = [
        make_verdict(f"p{i}", run_index=r, temperature=None) for i in range(6) for r in range(3)
    ]
    lines = [verdict_to_json_line(v) for v in verdicts]

    ordered = tmp_path / "ordered.jsonl"
    shuffled = tmp_path / "shuffled.jsonl"
    ordered.write_text("".join(line + "\n" for line in lines))
    scrambled = list(lines)
    random.Random(1).shuffle(scrambled)
    shuffled.write_text("".join(line + "\n" for line in scrambled))

    config = {"model_id": "m", "prompt_version": "v1", "temperature": None, "reasoning_effort": None}
    assert already_judged_ids(ordered, **config) == already_judged_ids(shuffled, **config)


def test_already_judged_ids_treats_a_partially_stamped_row_as_unproven(tmp_path):
    # PR #28 review: a row can carry base_url and config_digest but no
    # code_version — a caller that passed code_version to run_backend while the
    # client went unwired. Requiring BOTH to be absent missed that row, and
    # _check_provenance skips a null field, so a later run under different code
    # resumed it silently. That is the exact defect #13 exists to close.
    out = tmp_path / "backend.jsonl"
    out.write_text(
        verdict_to_json_line(
            make_verdict("p1", base_url="https://host-a.test/v1", config_digest="dig000000001")
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="allow-unknown-provenance"):
        already_judged_ids(out, **FULL_CONFIG)


def test_run_backend_wires_the_code_version_into_the_default_client(tmp_path):
    # The root cause of the above: run_backend built its default client without
    # the code_version it had been given, so the value never reached the rows.
    seen = {}

    class SpyClient(FakeClient):
        def __init__(self, backend, *, code_version=None):
            super().__init__(backend)
            seen["code_version"] = code_version

    backend = Backend(
        name="backend",
        base_url="https://host-a.test/v1",
        model_id="m",
        rpm=6000,
        eval_only=False,
        api_key_env="NVIDIA_API_KEY",
        temperature=0.0,
    )
    run_backend(
        backend,
        [make_pair("p1")],
        tmp_path,
        votes=1,
        code_version="9b0d01a1c2d3",
        judge_client=SpyClient,
    )
    assert seen["code_version"] == "9b0d01a1c2d3"
