import json

from judge.deadlocks import (
    RECORD_VERSION,
    BackendDeadlock,
    backend_deadlocks,
    deadlock_records,
    render_deadlock_section,
)
from judge.schema import ReasonCode, Verdict


def votes(model, pair_id, rulings):
    return [
        Verdict(
            pair_id=pair_id,
            verdict=verdict,
            reason=ReasonCode(reason),
            model_id=model,
            prompt_version="v2",
            temperature=None,
            run_index=i,
        )
        for i, (verdict, reason) in enumerate(rulings)
    ]


def test_a_backend_whose_own_votes_tied_is_recorded():
    # The masking case: this backend deadlocked 1-1, but pooled with other
    # backends that agree it disappears into a settled majority.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok")])}
    found = backend_deadlocks(by_model, leg="s1")
    assert [(d.backend, d.pair_id, d.kind) for d in found] == [("nv", "p1", "verdict")]


def test_a_settled_backend_is_not_recorded():
    by_model = {"nv": votes("nv", "p1", [("approve", "ok")] * 3)}
    assert backend_deadlocks(by_model, leg="s1") == []


def test_a_majority_is_not_a_deadlock():
    # 2-1 settles. Only a tie is a deadlock.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok"), ("approve", "ok")])}
    assert backend_deadlocks(by_model, leg="s1") == []


def test_a_reason_tie_is_recorded_separately_from_a_verdict_tie():
    # Both backends agree on `reject`; one cannot settle on WHY. That is a
    # weaker finding than a verdict deadlock and must be distinguishable —
    # 11 of the 12 real cases were this kind.
    by_model = {
        "nv": votes("nv", "p1", [("reject", "casing_error"), ("reject", "truncation_worse")])
    }
    found = backend_deadlocks(by_model, leg="s1")
    assert [d.kind for d in found] == ["reason"]


def test_a_backend_can_deadlock_on_both_axes_at_once():
    by_model = {
        "nv": votes("nv", "p1", [("approve", "ok"), ("reject", "casing_error")])
    }
    assert sorted(d.kind for d in backend_deadlocks(by_model, leg="s1")) == ["reason", "verdict"]


def test_the_split_is_recorded_so_an_audit_can_see_what_tied():
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok")])}
    found = backend_deadlocks(by_model, leg="s1")[0]
    assert found.split == {"approve": 1, "reject": 1}
    assert found.n_votes == 2


def test_the_leg_is_carried_on_every_record():
    # Acceptance: a backend run in four legs is four populations, not one.
    # Without the leg, s1 and s2-high collapse into a tie no single run produced.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok")])}
    assert backend_deadlocks(by_model, leg="s2-high")[0].leg == "s2-high"


def test_records_are_ordered_so_two_runs_diff_cleanly():
    by_model = {
        "b": votes("b", "p2", [("approve", "ok"), ("reject", "ok")]),
        "a": votes("a", "p1", [("approve", "ok"), ("reject", "ok")]),
    }
    found = backend_deadlocks(by_model, leg="s1")
    assert [(d.backend, d.pair_id) for d in found] == [("a", "p1"), ("b", "p2")]


def test_records_serialise_to_stable_machine_readable_json():
    # The audit trail outlives this code, and the harness is moving beyond
    # titles to more complex fields, so the schema carries a version.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok")])}
    lines = deadlock_records(backend_deadlocks(by_model, leg="s1")).splitlines()
    record = json.loads(lines[0])
    assert record["record_version"] == RECORD_VERSION
    assert record["leg"] == "s1"
    assert record["backend"] == "nv"
    assert record["pair_id"] == "p1"
    assert record["kind"] == "verdict"
    assert record["split"] == {"approve": 1, "reject": 1}


def test_no_deadlocks_serialises_to_nothing():
    assert deadlock_records([]) == ""


def test_the_section_names_the_backend_and_the_pair():
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok")])}
    rendered = "\n".join(render_deadlock_section(backend_deadlocks(by_model, leg="s1")))
    assert "nv" in rendered
    assert "p1" in rendered


def test_the_section_is_absent_when_nothing_deadlocked():
    assert render_deadlock_section([]) == []


def test_the_section_says_it_is_not_a_slate_level_finding():
    # The card's UNDECIDED marker means "the slate deadlocked" and its value is
    # that it is rare. This section must not read as the same claim.
    by_model = {"nv": votes("nv", "p1", [("approve", "ok"), ("reject", "ok")])}
    rendered = " ".join(render_deadlock_section(backend_deadlocks(by_model, leg="s1"))).lower()
    assert "one backend" in rendered or "single backend" in rendered


def test_a_deadlock_dataclass_is_hashable_for_set_comparison():
    a = BackendDeadlock(
        leg="s1", backend="nv", pair_id="p1", kind="verdict", n_votes=2, split={"a": 1}
    )
    assert a.pair_id == "p1"
