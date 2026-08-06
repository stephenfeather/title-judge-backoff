"""Per-pair disagreement stats read from a sweep's results directory.

This context is a nicety: when no sweep has run, the ruling pass must still
work, so every loader here degrades to "no data" rather than raising.
"""

from __future__ import annotations

import json

from judge import flips
from judge.schema import ReasonCode, Verdict


def verdict(pair_id: str, value: str, reason: ReasonCode, model_id: str) -> Verdict:
    return Verdict(
        pair_id=pair_id,
        verdict=value,
        reason=reason,
        model_id=model_id,
        prompt_version="v1",
        temperature=0.0,
    )


def write_results(directory, name: str, verdicts: list[Verdict]) -> None:
    from judge.schema import verdict_to_json_line

    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        "".join(verdict_to_json_line(v) + "\n" for v in verdicts), encoding="utf-8"
    )


# --- flip rate (pure) ---------------------------------------------------------


def test_unanimous_pair_has_zero_flip_rate():
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "approve", ReasonCode.OK, "m2"),
        ]
    )
    assert stats["a"].flip_rate == 0.0
    assert stats["a"].majority == "approve"
    assert stats["a"].n == 2


def test_evenly_split_pair_has_maximal_flip_rate():
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m2"),
        ]
    )
    assert stats["a"].flip_rate == 0.5


def test_flip_rate_is_one_minus_modal_share():
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "approve", ReasonCode.OK, "m2"),
            verdict("a", "approve", ReasonCode.OK, "m3"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m4"),
        ]
    )
    assert stats["a"].flip_rate == 0.25
    assert stats["a"].verdicts == {"approve": 3, "reject": 1}


def test_stats_are_grouped_per_pair():
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("b", "reject", ReasonCode.MEANING_CHANGE, "m1"),
        ]
    )
    assert set(stats) == {"a", "b"}


def test_reason_codes_are_counted():
    stats = flips.flip_stats(
        [
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m1"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m2"),
            verdict("a", "reject", ReasonCode.OVERCORRECTION, "m3"),
        ]
    )
    assert stats["a"].reasons == {"casing_error": 2, "overcorrection": 1}


def test_models_are_listed_sorted():
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "zeta"),
            verdict("a", "approve", ReasonCode.OK, "alpha"),
        ]
    )
    assert stats["a"].models == ["alpha", "zeta"]


def test_no_verdicts_yields_no_stats():
    assert flips.flip_stats([]) == {}


# --- loading (tolerant I/O) ---------------------------------------------------


def test_load_results_reads_every_backend_file(tmp_path):
    results = tmp_path / "2026-08-06"
    write_results(results, "m1.jsonl", [verdict("a", "approve", ReasonCode.OK, "m1")])
    write_results(results, "m2.jsonl", [verdict("a", "reject", ReasonCode.CASING_ERROR, "m2")])
    assert len(flips.load_results(results)) == 2


def test_load_results_descends_into_scenario_subdirectories(tmp_path):
    """A scenario sweep nests one directory per scenario; both layouts work."""
    root = tmp_path / "results"
    write_results(root / "baseline", "m1.jsonl", [verdict("a", "approve", ReasonCode.OK, "m1")])
    write_results(root / "terse", "m1.jsonl", [verdict("a", "reject", ReasonCode.CASING_ERROR, "m1")])
    assert len(flips.load_results(root)) == 2


def test_load_results_on_missing_directory_is_empty(tmp_path):
    assert flips.load_results(tmp_path / "never-ran") == []


def test_load_results_when_no_directory_requested_is_empty():
    assert flips.load_results(None) == []


def test_load_results_skips_unparseable_lines(tmp_path):
    """A sweep still in flight can leave a torn line; context must not crash."""
    results = tmp_path / "2026-08-06"
    write_results(results, "m1.jsonl", [verdict("a", "approve", ReasonCode.OK, "m1")])
    with (results / "m1.jsonl").open("a") as fh:
        fh.write('{"pair_id": "b", "verdi')
    assert len(flips.load_results(results)) == 1


def test_load_results_skips_files_that_are_not_verdicts(tmp_path):
    results = tmp_path / "2026-08-06"
    results.mkdir(parents=True)
    (results / "notes.jsonl").write_text(json.dumps({"hello": "world"}) + "\n")
    assert flips.load_results(results) == []


def test_load_flip_stats_is_the_one_call_the_tui_makes(tmp_path):
    results = tmp_path / "2026-08-06"
    write_results(results, "m1.jsonl", [verdict("a", "approve", ReasonCode.OK, "m1")])
    assert flips.load_flip_stats(results)["a"].flip_rate == 0.0
    assert flips.load_flip_stats(None) == {}


# --- rendering ----------------------------------------------------------------


def test_render_flip_pane_reports_split_and_reasons():
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m2"),
        ]
    )
    pane = flips.render_flip_pane(stats["a"])
    assert "50%" in pane
    assert "approve 1" in pane and "reject 1" in pane
    assert "casing_error" in pane


def test_render_flip_pane_without_stats_is_empty():
    assert flips.render_flip_pane(None) == ""
