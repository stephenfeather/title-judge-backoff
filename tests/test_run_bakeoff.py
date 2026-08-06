import pytest

from judge.client import Backend
from judge.schema import Pair, ReasonCode, Verdict, verdict_to_json_line
from run_bakeoff import already_judged_ids, pending_votes, run_manifest


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
    pair_id, model_id="m", prompt_version="v1", temperature=0.0, run_index=0, reasoning_effort=None
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
    )


CONFIG = dict(model_id="m", prompt_version="v1", temperature=0.0, reasoning_effort=None)


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


def test_run_manifest_summarizes_latency_and_errors():
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
