import json
from dataclasses import replace

import pytest

from judge.prompts import PROMPT_VERSION
from judge.schema import ReasonCode, Usage, Verdict, verdict_to_json_line
from scenario_report import (
    _health_table,
    _reason_cross_tab_sections,
    _stability_table,
    contention_ranking,
    load_results,
    render_scenario_report,
    unsettled_reason_pairs,
)


def make_verdict(pair_id, verdict, reason, model_id="m", run_index=0):
    return Verdict(
        pair_id=pair_id,
        verdict=verdict,
        reason=ReasonCode(reason),
        model_id=model_id,
        prompt_version="v1",
        temperature=None,
        run_index=run_index,
    )


def votes(model_id, pair_id, rulings):
    return [
        make_verdict(pair_id, v, r, model_id=model_id, run_index=i)
        for i, (v, r) in enumerate(rulings)
    ]


def test_load_results_reads_each_backend_and_skips_manifests(tmp_path):
    (tmp_path / "backend-a.jsonl").write_text(
        verdict_to_json_line(make_verdict("p1", "approve", "ok")) + "\n"
    )
    (tmp_path / "backend-b.jsonl").write_text(
        verdict_to_json_line(make_verdict("p1", "reject", "ok")) + "\n"
    )
    (tmp_path / "backend-a.manifest.json").write_text(json.dumps({"backend": "backend-a"}))

    loaded = load_results(tmp_path)
    assert set(loaded) == {"backend-a", "backend-b"}
    assert len(loaded["backend-a"]) == 1


def test_contention_ranking_puts_the_most_contested_item_first():
    by_model = {
        # p_split: models disagree with each other. p_agree: everyone agrees.
        "a": votes("a", "p_split", [("approve", "ok")]) + votes("a", "p_agree", [("approve", "ok")]),
        "b": votes("b", "p_split", [("reject", "casing_error")]) + votes("b", "p_agree", [("approve", "ok")]),
    }
    ranking = contention_ranking(by_model)
    assert ranking[0].pair_id == "p_split"
    assert ranking[0].cross_model_disagreement == pytest.approx(0.5)
    assert ranking[-1].pair_id == "p_agree"
    assert ranking[-1].cross_model_disagreement == 0.0


def test_contention_ranking_counts_within_model_flips_too():
    # An item every model calls "reject" but that flips between votes is still
    # unstable, and still worth an operator ruling.
    by_model = {
        "a": votes("a", "p_flip", [("reject", "ok"), ("approve", "ok"), ("reject", "ok")]),
        "b": votes("b", "p_flip", [("reject", "ok"), ("reject", "ok"), ("reject", "ok")]),
    }
    row = contention_ranking(by_model)[0]
    assert row.cross_model_disagreement == 0.0
    assert row.mean_verdict_flip_rate == pytest.approx((1 / 3) / 2)


def test_contention_ranking_surfaces_reason_only_disagreement():
    # Both models say reject, for different reasons. Invisible in the binary
    # rate — and this is the exact shape reasoning effort was shown to produce.
    by_model = {
        "a": votes("a", "p1", [("reject", "casing_error")]),
        "b": votes("b", "p1", [("reject", "truncation_worse")]),
    }
    row = contention_ranking(by_model)[0]
    assert row.cross_model_disagreement == 0.0
    assert row.cross_model_reason_disagreement > 0.0


def test_contention_row_records_how_many_models_judged_it():
    by_model = {
        "a": votes("a", "p1", [("approve", "ok")]),
        "b": votes("b", "p1", [("approve", "ok")]),
        "c": votes("c", "p2", [("approve", "ok")]),
    }
    rows = {r.pair_id: r for r in contention_ranking(by_model)}
    assert rows["p1"].n_models == 2
    assert rows["p2"].n_models == 1


def test_render_scenario_report_has_every_required_section():
    by_model = {
        "a": votes("a", "p1", [("approve", "ok"), ("reject", "ok"), ("approve", "ok")]),
        "b": votes("b", "p1", [("reject", "casing_error")] * 3),
    }
    manifests = {
        "a": {
            "model_id": "model-a",
            "votes": 3,
            "reasoning_effort": None,
            "temperature": None,
            "observed_models": ["model-a"],
            "health": {
                "calls_ok": 3,
                "calls_failed": 0,
                "latency_min": 1.0,
                "latency_median": 1.5,
                "latency_max": 2.0,
                "error_kinds": {},
            },
        },
        "b": {
            "model_id": "model-b",
            "votes": 3,
            "reasoning_effort": "medium",
            "temperature": None,
            "observed_models": ["model-b"],
            "health": {
                "calls_ok": 2,
                "calls_failed": 1,
                "latency_min": 3.0,
                "latency_median": 4.0,
                "latency_max": 5.0,
                "error_kinds": {"ReadTimeout": 1},
            },
        },
    }
    md = render_scenario_report(by_model, manifests)
    assert "# Scenario report" in md
    assert "flip" in md.lower()
    assert "agreement" in md.lower()
    assert "health" in md.lower()
    assert "ruling queue" in md.lower()
    assert "## Reliability" in md
    # Health must surface the failing backend, not silently average it away.
    assert "ReadTimeout" in md


def test_a_retired_reason_code_in_old_rows_is_labelled_as_retired():
    # Issue #44. A run judged under v2 keeps its truncation_worse rows forever.
    # Without the note there is no way to tell a code models DECLINED to use
    # from one they were never offered, and a reader would compare its count
    # against a v3 run as though both had the same menu.
    by_model = {"a": votes("a", "p1", [("reject", "truncation_worse")] * 3)}
    md = render_scenario_report(by_model, {})
    assert "Retired code(s) present" in md
    assert "`truncation_worse`" in md
    assert "appeared in votes" in md


def test_no_retirement_note_when_every_code_is_current():
    by_model = {"a": votes("a", "p1", [("reject", "casing_error")] * 3)}
    assert "Retired code" not in render_scenario_report(by_model, {})


def test_the_retirement_note_holds_when_the_code_only_ever_appears_in_unsettled_votes():
    # The note scans raw votes while the distribution table above it counts
    # settled majorities, so a scattered retired vote is exactly the case the
    # wording must survive: the code fired, the table shows nothing, and the
    # reader must not be pointed at counts that were never rendered.
    by_model = {
        "a": votes("a", "p1", [("reject", "truncation_worse"), ("reject", "casing_error"), ("reject", "ok")])
    }
    md = render_scenario_report(by_model, {})
    assert "Retired code(s) present" in md
    assert "appeared in votes" in md


def test_a_retired_code_on_a_current_version_row_is_flagged_as_a_contract_breach():
    # v3 never offers the code, so a v3 row carrying it is not history — it is
    # a model that ignored the menu (a parse-path JudgeResponseError should have
    # refused it). The note must not claim these rows predate the retirement.
    row = replace(make_verdict("p1", "reject", "truncation_worse"), prompt_version=PROMPT_VERSION)
    md = render_scenario_report({"a": [row]}, {})
    assert "still offered the code" not in md
    assert "never offered" in md


def test_historical_and_current_retired_rows_get_separate_paragraphs():
    old = votes("a", "p1", [("reject", "truncation_worse")])
    current = [replace(old[0], prompt_version=PROMPT_VERSION)]
    md = render_scenario_report({"a": old + current}, {})
    assert "still offered the code" in md
    assert "never offered" in md
    assert "different menus.\n\n**Retired code(s) on current-version" in md


def test_reliability_is_reported_before_any_quality_table():
    # Issue #43. Reliability can disqualify a backend outright, so a reader who
    # meets kappa or flip rates first has already formed a view of a judge that
    # cannot produce enough judgments to hold one. Order is the whole point of
    # the section — the same reasoning that put the cache warning up top (#15).
    by_model = {"a": votes("a", "p1", [("approve", "ok")] * 3)}
    md = render_scenario_report(by_model, {"a": {"health": {"calls_ok": 3, "calls_failed": 0}}})
    assert md.index("## Reliability") < md.index("## Stability")
    assert md.index("## Reliability") < md.index("## Cross-model verdict agreement")


def test_a_partial_backend_is_flagged_before_its_quality_numbers_are_shown():
    # "Say so at the point its quality numbers are shown" — a caveat printed
    # after the tables arrives once the reader has already believed them.
    by_model = {
        "full": votes("full", "p1", [("approve", "ok")] * 3)
        + votes("full", "p2", [("approve", "ok")] * 3),
        "partial": votes("partial", "p1", [("approve", "ok")] * 3),
    }
    md = render_scenario_report(by_model, {})
    assert "biased subset" in md
    assert md.index("biased subset") < md.index("## Stability")


def cached_votes(model_id, pair_id, rulings, cached=90):
    """Votes carrying per-call usage — where the cache evidence now comes from.

    The manifest is rewritten per launch while verdict rows accumulate across
    resumes, so the check reads the rows (issue #15 review).
    """
    return [
        replace(
            v,
            usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110, cached_tokens=cached),
        )
        for v in votes(model_id, pair_id, rulings)
    ]


def test_report_warns_when_a_cache_hit_coincides_with_a_flat_flip_rate():
    # Issue #15 acceptance: the harness correlates cache hits with a flat flip
    # rate itself, rather than leaving a reader to spot it across two sections.
    md = render_scenario_report({"nv": cached_votes("nv", "p1", [("approve", "ok")] * 3)}, {})
    assert "Vote independence" in md
    assert "collapsed majority" in md.lower()


def test_the_cache_warning_precedes_the_stability_table_it_undermines():
    # A warning printed after the numbers has already let them be believed.
    md = render_scenario_report({"nv": cached_votes("nv", "p1", [("approve", "ok")] * 3)}, {})
    assert md.index("Vote independence") < md.index("## Stability")


def test_report_raises_no_alarm_when_flips_prove_the_responses_differed():
    flipping = [("approve", "ok"), ("reject", "ok"), ("approve", "ok")]
    md = render_scenario_report({"nv": cached_votes("nv", "p1", flipping)}, {})
    # The section still reports that it checked (silence would be indistinguishable
    # from the check not existing), but raises no alarm.
    assert "collapsed majority" not in md.lower()
    assert "clear" in md.lower()


def test_report_flags_a_flat_backend_with_no_usage_data_as_unverifiable():
    by_model = {"nv": votes("nv", "p1", [("approve", "ok")] * 3)}
    md = render_scenario_report(by_model, {})
    assert "Vote independence" in md
    assert "not checkable" in md.lower()


def test_report_surfaces_a_backend_that_deadlocked_alone(tmp_path):
    # #23: one backend ties 1-1 while two others agree. Pooled by pair_id the
    # tie vanishes into a settled majority; the report must still show it.
    by_model = {
        "tied": votes("tied", "p1", [("approve", "ok"), ("reject", "ok")]),
        "agree-a": votes("agree-a", "p1", [("approve", "ok")] * 3),
        "agree-b": votes("agree-b", "p1", [("approve", "ok")] * 3),
    }
    md = render_scenario_report(by_model, {}, leg="s1")
    assert "Backend-level deadlocks" in md
    assert "tied" in md


def test_report_deadlock_section_is_absent_when_every_backend_settled():
    by_model = {"nv": votes("nv", "p1", [("approve", "ok")] * 3)}
    assert "Backend-level deadlocks" not in render_scenario_report(by_model, {}, leg="s1")


def test_deadlock_section_does_not_claim_the_slate_deadlocked():
    # UNDECIDED on the card means the slate deadlocked, and its worth is that it
    # is rare. This section must read as a different, quieter claim.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok")])}
    md = render_scenario_report(by_model, {}, leg="s1").lower()
    assert "one backend" in md


def test_report_separates_model_failures_from_transport_failures():
    # Issue #41: a model that cannot follow the output contract must not be
    # indistinguishable from a connection that dropped. Coverage merges them;
    # this section must not.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok")] * 3)}
    manifests = {
        "nv": {
            "base_url": "https://api.deepinfra.com/v1/openai",
            "health": {
                "calls_ok": 90,
                "calls_failed": 10,
                "error_kinds": {"JudgeResponseError": 7, "ReadTimeout": 3},
            },
        }
    }
    md = render_scenario_report(by_model, manifests)
    assert "Output-contract compliance" in md
    # The finding is the 7 the model got wrong, not the 3 the network lost.
    assert "7" in md
    assert "contract" in md.lower()


def test_compliance_prefers_cumulative_counts_over_the_last_launch():
    # #40: the last launch of a resumed run is typically small and clean, so
    # reading `health` alone reports a flawless backend that failed hundreds of
    # calls earlier. The cumulative block is what survives.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok")] * 3)}
    manifests = {
        "nv": {
            "base_url": "https://api.openai.com/v1",
            "health": {"calls_ok": 47, "calls_failed": 0, "error_kinds": {}},
            "cumulative": {
                "launches": 2,
                "calls_ok": 550,
                "calls_failed": 50,
                "error_kinds": {"JudgeResponseError": 50},
            },
        }
    }
    md = render_scenario_report(by_model, manifests)
    assert "Output-contract compliance" in md
    assert "50" in md


def test_compliance_section_is_absent_when_no_backend_violated_the_contract():
    by_model = {"nv": votes("nv", "p1", [("approve", "ok")] * 3)}
    manifests = {
        "nv": {
            "base_url": "https://api.openai.com/v1",
            "health": {"calls_ok": 100, "calls_failed": 3, "error_kinds": {"ReadTimeout": 3}},
        }
    }
    assert "Output-contract compliance" not in render_scenario_report(by_model, manifests)


def test_host_concentration_is_read_from_the_slate_not_hardcoded():
    # The prose version of this caveat named six backends on integrate.api.nvidia.com
    # long after the slate had moved off that host entirely. A caveat that needs
    # hand-editing to stay true eventually reads as false.
    by_model = {"a": votes("a", "p1", [("approve", "ok")]), "b": votes("b", "p1", [("approve", "ok")]),
                "c": votes("c", "p1", [("approve", "ok")])}
    manifests = {
        "a": {"base_url": "https://api.deepinfra.com/v1/openai"},
        "b": {"base_url": "https://api.deepinfra.com/v1/openai"},
        "c": {"base_url": "https://api.openai.com/v1"},
    }
    md = render_scenario_report(by_model, manifests)
    assert "2 of 3 share `api.deepinfra.com`" in md
    assert "nvidia" not in md.lower()


def test_host_concentration_counts_a_backend_with_no_manifest():
    # A killed backend writes no manifest (#40), so a manifest-only count makes
    # it vanish. Its verdict rows carry base_url and survive, so it must still
    # be counted — this undercounted the real 2026-08-11 run by one backend.
    rows = [replace(v, base_url="https://api.deepinfra.com/v1/openai")
            for v in votes("killed", "p1", [("approve", "ok")])]
    by_model = {"killed": rows, "finished": votes("finished", "p1", [("approve", "ok")])}
    manifests = {"finished": {"base_url": "https://api.deepinfra.com/v1/openai"}}
    md = render_scenario_report(by_model, manifests)
    assert "2 of 2 share `api.deepinfra.com`" in md


def test_host_concentration_says_so_when_nothing_is_shared():
    by_model = {"a": votes("a", "p1", [("approve", "ok")]), "b": votes("b", "p1", [("approve", "ok")])}
    manifests = {
        "a": {"base_url": "https://api.openai.com/v1"},
        "b": {"base_url": "https://api.anthropic.com/v1"},
    }
    md = render_scenario_report(by_model, manifests)
    assert "No two backends in this run shared a host." in md


def test_cross_tab_rows_are_totally_ordered_so_two_runs_agree():
    """The report must be diffable. Found while verifying #18.

    Rows were sorted by count alone, so ties fell back to dict insertion order
    — which comes from iterating a SET intersection, and Python randomizes str
    hashing per process. Two runs of identical code over identical data
    produced 60 differing lines, all of them tied-count rows.

    That is worse than untidy: "regenerate the report and diff it" is the
    verification this project keeps relying on, and it was returning false
    alarms. The sort needs a tiebreaker so the order is a property of the data
    rather than of the process.
    """
    # Eight pairs, every cross-tab cell tied at 1, so ordering is decided
    # entirely by the tiebreaker. Eight rather than four on purpose: with the
    # bug, the row order is whatever the hash seed produces, and a small
    # fixture passes by coincidence whenever that order happens to be sorted.
    # At four rows that is 1 in 24 — measured, one seed in five slipped through.
    # At eight it is 1 in 40320.
    codes = ["casing_error", "meaning_change", "overcorrection", "truncation_worse"]
    pairings = [(codes[i % 4], codes[(i + 1 + i // 4) % 4]) for i in range(8)]
    left, right = [], []
    for index, (lreason, rreason) in enumerate(pairings):
        pair_id = f"p{index}"
        left += votes("a", pair_id, [("reject", lreason)])
        right += votes("b", pair_id, [("reject", rreason)])
    lines = _reason_cross_tab_sections({"a": left, "b": right})

    # Row shape is "| left | right | count[ ⟵ divergence] |"; the marker rides
    # on the count cell.
    parsed = []
    for line in lines:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) != 5 or cells[0] or cells[4]:
            continue
        count = cells[3].replace("⟵ divergence", "").strip()
        if not count.isdigit():
            continue  # header row
        parsed.append((-int(count), cells[1], cells[2]))

    assert len(parsed) >= 6, f"expected the tied cross-tab rows, got {parsed}"
    assert parsed == sorted(parsed), (
        "cross-tab rows are not totally ordered; ties fall back to dict "
        f"insertion order and will differ between processes: {parsed}"
    )


def test_health_table_gives_each_backend_its_own_numbers():
    """Issue #18, and the guard the late-binding closure needed.

    `fmt` used to be defined inside the loop, closing over the rebound `h`.
    It was accidentally correct because every call happened eagerly in the same
    iteration — so this test PASSES against the old code too, and there was no
    honest way to write a failing one. Its job is to make a future regression
    loud: the moment any of these values is formatted lazily, or the helper is
    reused after the loop, two backends start reporting the same latencies and
    this fails.

    Two backends with deliberately disjoint numbers, so a leak is unmissable.
    """
    lines = _health_table(
        {
            "alpha": {
                "health": {"calls_ok": 1, "latency_min": 1.0, "latency_median": 1.5, "latency_max": 2.0},
                "observed_models": ["a"],
            },
            "zulu": {
                "health": {"calls_ok": 2, "latency_min": 9.0, "latency_median": 9.5, "latency_max": 9.9},
                "observed_models": ["z"],
            },
        }
    )
    alpha = next(line for line in lines if "| alpha |" in line)
    zulu = next(line for line in lines if "| zulu |" in line)
    assert "1.0 / 1.5 / 2.0" in alpha
    assert "9.0 / 9.5 / 9.9" in zulu
    # The specific failure a loop-variable closure produces: the last
    # iteration's values on every row.
    assert "9.0 / 9.5 / 9.9" not in alpha


def test_stability_table_survives_a_backend_that_judged_nothing():
    # A backend whose every call errored leaves an empty verdict list. Dividing
    # by n crashed the WHOLE report render — so one dead backend destroyed the
    # output for every healthy one.
    lines = _stability_table({"dead": [], "alive": votes("alive", "p1", [("approve", "ok")])})
    rendered = "\n".join(lines)
    assert "dead" in rendered
    assert "alive" in rendered


def test_report_renders_with_an_all_errors_backend_present():
    md = render_scenario_report({"dead": [], "alive": votes("alive", "p1", [("approve", "ok")])}, {})
    assert "dead" in md
    assert "alive" in md


def test_report_includes_a_reason_cross_tab_between_models():
    # The PR promised inter-model reason cross-tabs and the report only rendered
    # per-backend distributions. The cross-tab is the part that shows WHERE two
    # models diverge — both call it reject, for different reasons.
    by_model = {
        "a": votes("a", "p1", [("reject", "casing_error")]),
        "b": votes("b", "p1", [("reject", "truncation_worse")]),
    }
    md = render_scenario_report(by_model, {})
    assert "cross-tab" in md.lower()
    # The off-diagonal cell is the finding: same verdict, different reason.
    assert "casing_error" in md and "truncation_worse" in md


def test_cross_tab_section_is_skipped_for_a_single_backend():
    # Nothing to cross-tabulate against; an empty section would be noise.
    md = render_scenario_report({"only": votes("only", "p1", [("approve", "ok")])}, {})
    assert "cross-tab" not in md.lower()


def test_render_scenario_report_states_that_kappa_is_not_computable():
    # The report is produced BEFORE rulings exist. It must say so, or a reader
    # will assume the absence of kappa is an oversight.
    by_model = {"a": votes("a", "p1", [("approve", "ok")])}
    md = render_scenario_report(by_model, {})
    assert "no operator rulings" in md.lower()
    assert "kappa" in md.lower()


def unsettled_votes(model_id, pair_id):
    """Three votes, three reasons: settled verdict, no majority reason."""
    return votes(
        model_id,
        pair_id,
        [("reject", "meaning_change"), ("reject", "ok"), ("reject", "overcorrection")],
    )


def test_unsettled_reason_pairs_catches_a_two_two_split_not_just_all_distinct():
    # The version shipped in PR #19 looked for "every reason different", which
    # misses a 2-2 tie on four votes — equally undecided, and reachable when a
    # pair is re-judged after a resume. Deriving from tally_votes instead of
    # re-implementing the rule removes the second definition entirely.
    two_two = votes(
        "m",
        "split-2-2",
        [
            ("reject", "casing_error"),
            ("reject", "casing_error"),
            ("reject", "overcorrection"),
            ("reject", "overcorrection"),
        ],
    )
    clear = votes("m", "clear", [("reject", "ok"), ("reject", "ok"), ("reject", "casing_error")])
    assert unsettled_reason_pairs(two_two + clear) == ["split-2-2"]


def test_contention_counts_models_that_failed_to_settle():
    # Requirement 3 of issue #12. Previously an unsettled pair was ranked on
    # whatever the tie-break invented, which could push it up OR bury it. It
    # must rank BECAUSE it is unsettled: a pair its own judges could not decide
    # is the definition of an ambiguous rubric, which is what the queue is for.
    by_model = {
        "a": unsettled_votes("a", "cannot-decide") + votes("a", "easy", [("approve", "ok")] * 3),
        "b": unsettled_votes("b", "cannot-decide") + votes("b", "easy", [("approve", "ok")] * 3),
    }
    rows = {r.pair_id: r for r in contention_ranking(by_model)}
    assert rows["cannot-decide"].n_unsettled == 2
    assert rows["easy"].n_unsettled == 0
    assert rows["cannot-decide"].contention > rows["easy"].contention


def test_an_unsettled_pair_outranks_one_its_judges_merely_disagreed_on():
    # Two models disagreeing is evidence. A model unable to decide at all is
    # stronger evidence, and previously scored lower because a tie-broken value
    # can happen to match the other model's.
    by_model = {
        "a": unsettled_votes("a", "undecidable") + votes("a", "disputed", [("approve", "ok")] * 3),
        "b": unsettled_votes("b", "undecidable")
        + votes("b", "disputed", [("reject", "casing_error")] * 3),
    }
    rows = {r.pair_id: r for r in contention_ranking(by_model)}
    assert rows["undecidable"].contention >= rows["disputed"].contention


def test_contention_ranking_does_not_crash_on_an_unsettled_pair():
    # The regression that motivated all of this: `r.reason.value` on a None.
    by_model = {"a": unsettled_votes("a", "p1")}
    assert contention_ranking(by_model)[0].pair_id == "p1"


def test_completion_is_counted_from_rows_not_from_manifest_health():
    # health.calls_ok covers only the LAST launch segment, so on any resumed
    # backend it under-reports — four of seven manifests in the 2026-08-06 run
    # misreport completion this way. Completion is rows on disk and distinct
    # pair_ids, and the report has to say so where a reader will see it.
    by_model = {
        "resumed": votes("resumed", "p1", [("approve", "ok")] * 3)
        + votes("resumed", "p2", [("approve", "ok")] * 3)
    }
    manifests = {"resumed": {"health": {"calls_ok": 1, "calls_failed": 0}}}
    md = render_scenario_report(by_model, manifests)

    assert "6" in md  # six rows on disk, not the 1 the health block claims
    assert "calls_ok" in md
    assert "last launch" in md.lower()


def test_report_names_the_pairs_whose_reason_has_no_majority():
    # A 1-1-1 reason split has no majority. tally_votes returns one anyway, and
    # it is recorded indistinguishably from a 3-0 consensus (issue #12). Until
    # that is representable, the report must at least name them, or a reader
    # takes a fabricated tie-break for a settled ruling.
    by_model = {
        "m": votes(
            "m",
            "three-way-tie",
            [("reject", "meaning_change"), ("reject", "ok"), ("reject", "overcorrection")],
        )
        + votes("m", "unanimous-pair", [("approve", "ok")] * 3)
    }
    md = render_scenario_report(by_model, {})
    assert "no majority" in md.lower()

    # Scoped to the caveats block: every pair also appears in the ruling queue,
    # so a whole-document check would pass for the wrong reason.
    fabricated_block = md.split("### Reason codes with no majority")[1].split("###")[0]
    assert "three-way-tie" in fabricated_block
    assert "unanimous-pair" not in fabricated_block
