"""Per-pair disagreement stats read from a sweep's results directory.

This context is a nicety: when no sweep has run, the ruling pass must still
work, so every loader here degrades to "no data" rather than raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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


def test_pane_rate_is_the_same_function_the_vote_tally_uses():
    """One metric, one implementation.

    The operator card and the leaderboard must never describe the same pair
    differently, so this asserts agreement with `judge.vote` rather than
    re-stating the arithmetic.
    """
    from judge.vote import flip_rate

    values = ["approve", "approve", "reject", "approve"]
    stats = flips.flip_stats(
        [
            verdict("a", value, ReasonCode.OK if value == "approve" else ReasonCode.CASING_ERROR, f"m{i}")
            for i, value in enumerate(values)
        ]
    )
    assert stats["a"].flip_rate == flip_rate(values)


def test_a_tied_pair_has_no_majority_rather_than_a_tie_broken_one():
    """Issue #21. This card feeds the pass where GROUND TRUTH is created, so a
    fabricated value here anchors a label nothing can later recompute."""
    stats = flips.flip_stats(
        [
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m1"),
            verdict("a", "approve", ReasonCode.OK, "m2"),
        ]
    )
    assert stats["a"].majority is None
    assert stats["a"].settled is False


def test_a_clear_pair_still_reports_its_majority():
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "approve", ReasonCode.OK, "m2"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m3"),
        ]
    )
    assert stats["a"].majority == "approve"
    assert stats["a"].settled is True


def test_repeat_verdicts_from_one_model_count_as_repeats():
    """Majority-of-N runs the same model several times; each vote is a vote."""
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m1"),
        ]
    )
    assert stats["a"].n == 3
    assert stats["a"].verdicts == {"approve": 2, "reject": 1}
    assert stats["a"].models == ["m1"]


def test_grouping_is_by_pair_id_alone():
    """Never by (pair, model, run) — the pane answers "did this pair flip"."""
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m2"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m2"),
        ]
    )
    assert set(stats) == {"a"}
    assert stats["a"].n == 3


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


def test_load_results_ignores_fields_it_does_not_know(tmp_path):
    """The sweep's verdict record grows (run index, vote index, effort). Extra
    keys are the sweep's business, not this pane's — never a reason to drop a
    line, and never a reason to widen the grouping key."""
    results = tmp_path / "2026-08-06"
    results.mkdir(parents=True)
    (results / "m1.jsonl").write_text(
        json.dumps(
            {
                "pair_id": "a",
                "verdict": "approve",
                "reason": "ok",
                "model_id": "m1",
                "prompt_version": "v1",
                "temperature": 0.0,
                "run_index": 2,
                "vote_index": 1,
                "effort": "high",
                "something_invented_next_week": {"nested": True},
            }
        )
        + "\n"
    )
    loaded = flips.load_results(results)
    assert len(loaded) == 1
    assert loaded[0].pair_id == "a"
    assert flips.flip_stats(loaded)["a"].flip_rate == 0.0


def test_load_results_needs_only_the_four_fields_it_reads(tmp_path):
    """Loading must not depend on the rest of the Verdict schema holding still."""
    results = tmp_path / "2026-08-06"
    results.mkdir(parents=True)
    (results / "m1.jsonl").write_text(
        json.dumps({"pair_id": "a", "verdict": "reject", "reason": "casing_error",
                    "model_id": "m1"}) + "\n"
    )
    assert len(flips.load_results(results)) == 1


def test_load_results_keeps_repeat_votes_from_the_same_model(tmp_path):
    """Majority-of-N writes several lines per (pair, model). None may be lost."""
    results = tmp_path / "2026-08-06"
    write_results(
        results,
        "m1.jsonl",
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m1"),
        ],
    )
    assert flips.flip_stats(flips.load_results(results))["a"].n == 3


def test_loading_does_not_construct_the_schema_verdict(tmp_path):
    """Decoupled on purpose.

    If this built `schema.Verdict` and that record gained a required field, every
    line would raise and be dropped — the pane would silently go blank instead of
    failing loudly. So loading yields its own four-field record.
    """
    from judge.schema import Verdict as SchemaVerdict

    results = tmp_path / "2026-08-06"
    write_results(results, "m1.jsonl", [verdict("a", "approve", ReasonCode.OK, "m1")])
    loaded = flips.load_results(results)[0]
    assert not isinstance(loaded, SchemaVerdict)
    assert (loaded.pair_id, loaded.verdict, loaded.model_id) == ("a", "approve", "m1")


def test_stats_accept_any_record_carrying_the_four_fields():
    """Structural, not nominal — a richer sweep record works unchanged."""

    @dataclass(frozen=True)
    class RicherVerdict:
        pair_id: str
        verdict: str
        reason: ReasonCode
        model_id: str
        run_index: int
        effort: str

    stats = flips.flip_stats(
        [
            RicherVerdict("a", "approve", ReasonCode.OK, "m1", 0, "high"),
            RicherVerdict("a", "reject", ReasonCode.CASING_ERROR, "m1", 1, "high"),
        ]
    )
    assert stats["a"].flip_rate == 0.5


def test_an_unknown_reason_code_still_counts_toward_the_flip_rate(tmp_path):
    """A reason vocabulary this branch hasn't seen must not erase the verdict —
    the flip rate is about approve/reject, not about the reason enum."""
    results = tmp_path / "2026-08-06"
    results.mkdir(parents=True)
    (results / "m1.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"pair_id": "a", "verdict": "approve", "reason": "ok",
                            "model_id": "m1"}),
                json.dumps({"pair_id": "a", "verdict": "reject",
                            "reason": "invented_next_week", "model_id": "m2"}),
            ]
        )
        + "\n"
    )
    stats = flips.flip_stats(flips.load_results(results))
    assert stats["a"].n == 2
    assert stats["a"].flip_rate == 0.5
    assert stats["a"].reasons["invented_next_week"] == 1


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


def test_render_flip_pane_counts_votes_and_models_separately():
    """With majority-of-N, votes outnumber models — saying "judges" would lie."""
    stats = flips.flip_stats(
        [
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "approve", ReasonCode.OK, "m1"),
            verdict("a", "reject", ReasonCode.CASING_ERROR, "m2"),
        ]
    )
    pane = flips.render_flip_pane(stats["a"])
    assert "3 votes" in pane
    assert "2 models" in pane


def test_render_flip_pane_without_stats_is_empty():
    assert flips.render_flip_pane(None) == ""


# --- undecided must be unmissable (issue #21) -------------------------------
#
# Fixtures are the three real ties measured across the pooled 2026-08-06 run:
#   e10-4f92576a25bf  verdict  approve 16 / reject 16          n=32
#   e10-816b0c7d2b94  reason   ok 14 / casing_error 14         n=33
#   e10-8a1d3065c6e8  reason   overcorrection 12 / ok 12       n=33


def split_verdicts(pair_id, approve, reject):
    """`approve` + `reject` judgments spread across distinct models."""
    return [
        verdict(pair_id, "approve", ReasonCode.OK, f"m{i}") for i in range(approve)
    ] + [
        verdict(pair_id, "reject", ReasonCode.CASING_ERROR, f"m{approve + i}")
        for i in range(reject)
    ]


def test_pane_says_undecided_when_the_verdict_is_tied():
    # e10-4f92576a25bf: 16-16 across 32 judgments from eight models. Not a coin
    # flip on thin data — deep, well-sampled disagreement, and the single pair
    # an operator most needs flagged rather than summarised.
    stats = flips.flip_stats(split_verdicts("e10-4f92576a25bf", 16, 16))
    pane = flips.render_flip_pane(stats["e10-4f92576a25bf"])
    assert "UNDECIDED" in pane
    # The counts must still be there — the flag explains them, it does not
    # replace them.
    assert "approve 16" in pane and "reject 16" in pane


def test_pane_does_not_cry_undecided_when_the_judges_agreed():
    # A 17-15 split is contested but decided. Flagging it too would make the
    # marker noise, and a marker that fires on everything flags nothing.
    stats = flips.flip_stats(split_verdicts("clear", 17, 15))
    pane = flips.render_flip_pane(stats["clear"])
    assert "UNDECIDED" not in pane


def test_pane_flags_a_tied_reason_even_when_the_verdict_settled():
    # e10-816b0c7d2b94: every judge said reject, but ok 14 / casing_error 14 on
    # WHY. The verdict is real; the reason is not, and the reason is what the
    # per-reason confusion matrix will score once rulings exist.
    verdicts = (
        [verdict("e10-816b0c7d2b94", "reject", ReasonCode.OK, f"a{i}") for i in range(14)]
        + [
            verdict("e10-816b0c7d2b94", "reject", ReasonCode.CASING_ERROR, f"b{i}")
            for i in range(14)
        ]
        + [
            verdict("e10-816b0c7d2b94", "reject", ReasonCode.OVERCORRECTION, f"c{i}")
            for i in range(5)
        ]
    )
    pane = flips.render_flip_pane(flips.flip_stats(verdicts)["e10-816b0c7d2b94"])
    banner = pane.splitlines()[0]
    assert "UNDECIDED" in banner
    # Specifically the REASON, not the verdict — asserting the bare word
    # "reason" would pass off the reasons breakdown line and prove nothing.
    assert "reason (" in banner
    assert "verdict (" not in banner
    assert "ok 14 = casing_error 14" in banner or "casing_error 14 = ok 14" in banner


def test_undecided_marker_is_on_the_headline_line():
    # A marker buried under the reason breakdown is a marker an operator
    # scanning a 200-row pass will miss, which is the whole failure mode.
    stats = flips.flip_stats(split_verdicts("tied", 16, 16))
    pane = flips.render_flip_pane(stats["tied"])
    assert "UNDECIDED" in pane.splitlines()[0]
