import json

import pytest

from judge.schema import ReasonCode, Verdict, verdict_to_json_line
from scenario_report import (
    ContentionRow,
    _stability_table,
    contention_ranking,
    load_results,
    render_scenario_report,
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
    # Health must surface the failing backend, not silently average it away.
    assert "ReadTimeout" in md


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
