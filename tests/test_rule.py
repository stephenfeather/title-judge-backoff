"""The ruling TUI: rendering and the keystroke loop.

The loop's key reader, line reader, output sink and clock are all injected, so
a whole operator session is driven here by a string of keys — no TTY, no
sleeping, no vendor data. Titles are synthetic throughout.
"""

from __future__ import annotations

import io

import rule
from judge import flips, rulings
from judge.schema import ReasonCode, Verdict


def row(pair_id: str, *, before="acme widget 3000 blk", after="Acme Widget 3000, Black",
        cohort="casing", stages=("casing",)) -> dict:
    return {
        "id": pair_id,
        "original": before,
        "enriched": after,
        "cohort": cohort,
        "source": "synthetic",
        "stages": list(stages),
        "ground_truth": None,
        "reason": None,
    }


def unchanged_row(pair_id: str) -> dict:
    title = "GAMO SWARM VIPER 10X GEN3i"
    return row(pair_id, before=title, after=title, cohort="unchanged", stages=())


def drive(keys: str, rows, journal_path, *, lines=(), **kwargs):
    """Run one session over `rows`, feeding `keys`. Returns (result, output)."""
    out = io.StringIO()
    result = rule.run_session(
        rows,
        journal_path,
        read_key=rule.keys_from_string(keys),
        read_line=rule.lines_from_iterable(lines),
        out=out,
        **kwargs,
    )
    return result, out.getvalue()


# --- diff rendering -----------------------------------------------------------


def test_word_diff_marks_added_and_removed_tokens():
    diff = rule.word_diff("acme widget blk", "Acme Widget Black")
    assert "acme" in diff and "Acme" in diff


def test_word_diff_on_identical_titles_says_no_change():
    assert "no change" in rule.word_diff("Same Title", "Same Title").lower()


def test_word_diff_aligns_tokens_that_differ_only_in_case():
    """Casing is the biggest cohort — a recased token must not read as a rewrite."""
    diff = rule.word_diff("acme widget blk", "Acme Widget Black")
    assert "[blk]" in diff and "[Black]" in diff
    assert "[acme widget]" not in diff


def test_word_diff_marks_a_casing_only_change_visibly():
    diff = rule.word_diff("acme widget", "Acme Widget")
    assert "~acme~" in diff and "~Acme~" in diff


def test_word_diff_leaves_untouched_tokens_unmarked():
    diff = rule.word_diff("Acme Widget blk", "Acme Widget Black")
    assert "~" not in diff
    assert "[Acme]" not in diff


def test_render_card_shows_cohort_stages_and_position():
    card = rule.render_card(row("e10-aaa"), position=3, total=200)
    assert "casing" in card
    assert "3/200" in card
    assert "e10-aaa" in card


def test_render_card_names_the_stages_that_fired():
    card = rule.render_card(row("e10-aaa", stages=("casing", "brand_prefix")), 1, 1)
    assert "brand_prefix" in card


def test_render_card_says_so_when_no_stage_fired():
    card = rule.render_card(unchanged_row("e10-aaa"), 1, 1)
    assert "no stages" in card.lower()


def test_render_card_flags_the_unchanged_cohort_with_an_auto_offer():
    card = rule.render_card(unchanged_row("e10-aaa"), 1, 1)
    assert "unchanged" in card.lower()
    assert "enter" in card.lower()


def test_render_card_includes_flip_context_when_present():
    stats = flips.flip_stats(
        [
            Verdict("e10-aaa", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("e10-aaa", "reject", ReasonCode.CASING_ERROR, "m2", "v1", 0.0),
        ]
    )
    card = rule.render_card(row("e10-aaa"), 1, 1, stats=stats.get("e10-aaa"))
    assert "flip 50%" in card


def test_render_card_omits_flip_context_when_absent():
    card = rule.render_card(row("e10-aaa"), 1, 1, stats=None)
    assert "flip" not in card.lower()


# --- the keystroke loop -------------------------------------------------------


def test_approve_journals_approve_with_ok_reason(tmp_path):
    journal = tmp_path / "j.jsonl"
    result, _ = drive("a", [row("a")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings["a"]["ground_truth"] == "approve"
    assert state.rulings["a"]["reason"] == "ok"
    assert result.approved == 1


def test_reject_then_reason_key_journals_that_code(tmp_path):
    journal = tmp_path / "j.jsonl"
    key, code = next(iter(rulings.reject_reason_keymap().items()))
    drive(f"r{key}", [row("a")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings["a"]["ground_truth"] == "reject"
    assert state.rulings["a"]["reason"] == code.value


def test_no_reason_key_collides_with_a_verdict_action_key():
    """`judge/rulings.py` reserves these letters; this is what keeps it honest."""
    action_keys = {
        rule.KEY_APPROVE, rule.KEY_REJECT, rule.KEY_SKIP,
        rule.KEY_UNDO, rule.KEY_NOTE, rule.KEY_QUIT,
    }
    assert not action_keys & set(rulings.reason_keymap())
    assert not action_keys & set(rulings.KEY_POOL)


def test_reject_shows_the_reason_menu(tmp_path):
    key = next(iter(rulings.reject_reason_keymap()))
    _, output = drive(f"r{key}", [row("a")], tmp_path / "j.jsonl")
    assert "overcorrection" in output


def test_unknown_reason_key_aborts_the_rejection(tmp_path):
    """A fat-fingered reason must not invent ground truth — re-present the row."""
    journal = tmp_path / "j.jsonl"
    result, _ = drive("rZa", [row("a")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings["a"]["ground_truth"] == "approve"
    assert result.ruled == 1


def test_skip_leaves_the_pair_unruled(tmp_path):
    journal = tmp_path / "j.jsonl"
    result, _ = drive("s", [row("a")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings == {}
    assert state.skipped == ["a"]
    assert result.skipped == 1


def test_undo_retracts_the_last_ruling_and_re_presents_it(tmp_path):
    journal = tmp_path / "j.jsonl"
    result, _ = drive("au", [row("a"), row("b")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings == {}
    assert result.undos == 1


def test_undo_then_reruling_lands_on_the_original_pair(tmp_path):
    journal = tmp_path / "j.jsonl"
    key = next(iter(rulings.reject_reason_keymap()))
    drive(f"au r{key}".replace(" ", ""), [row("a"), row("b")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings["a"]["ground_truth"] == "reject"
    assert "b" not in state.rulings


def test_undo_after_a_skip_does_not_retract_an_earlier_ruling(tmp_path):
    """Undo steps back one row; it must not reach past a skip into a ruling."""
    journal = tmp_path / "j.jsonl"
    drive("asu", [row("a"), row("b"), row("c")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings["a"]["ground_truth"] == "approve"


def test_undo_reverses_the_approve_tally(tmp_path):
    result, _ = drive("au", [row("a"), row("b")], tmp_path / "j.jsonl")
    assert (result.ruled, result.approved) == (0, 0)
    assert result.reasons == {}


def test_undo_reverses_the_reject_tally_and_its_reason(tmp_path):
    key, code = next(iter(rulings.reject_reason_keymap().items()))
    result, _ = drive(f"r{key}u", [row("a"), row("b")], tmp_path / "j.jsonl")
    assert (result.ruled, result.rejected) == (0, 0)
    assert code.value not in result.reasons


def test_tallies_stay_consistent_with_the_ruled_count_after_undo(tmp_path):
    """`ruled` and approved+rejected are the same decisions counted twice —
    they must never disagree, or the summary lies about the pass."""
    key = next(iter(rulings.reject_reason_keymap()))
    result, _ = drive(f"ar{key}ua", [row("a"), row("b"), row("c")], tmp_path / "j.jsonl")
    assert result.ruled == result.approved + result.rejected
    assert sum(result.reasons.values()) == result.ruled


def test_undo_then_reruling_tallies_only_the_final_verdict(tmp_path):
    key, code = next(iter(rulings.reject_reason_keymap().items()))
    result, _ = drive(f"au r{key}".replace(" ", ""), [row("a"), row("b")],
                      tmp_path / "j.jsonl")
    assert (result.approved, result.rejected) == (0, 1)
    assert result.reasons == {code.value: 1}


def test_repeated_undo_unwinds_each_ruling_in_turn(tmp_path):
    result, _ = drive("aauu", [row("a"), row("b"), row("c")], tmp_path / "j.jsonl")
    assert (result.ruled, result.approved, result.undos) == (0, 0, 2)


def test_undoing_a_skip_does_not_touch_the_verdict_tallies(tmp_path):
    result, _ = drive("asu", [row("a"), row("b"), row("c")], tmp_path / "j.jsonl")
    assert (result.approved, result.skipped) == (1, 0)


def test_undo_at_the_start_of_a_session_is_a_noop(tmp_path):
    journal = tmp_path / "j.jsonl"
    result, _ = drive("ua", [row("a")], journal)
    assert result.undos == 0
    assert rulings.ruled_ids(journal) == {"a"}


def test_quit_ends_the_session_early(tmp_path):
    journal = tmp_path / "j.jsonl"
    result, _ = drive("aq", [row("a"), row("b")], journal)
    assert result.quit_early
    assert rulings.ruled_ids(journal) == {"a"}


def test_exhausted_keys_end_the_session(tmp_path):
    result, _ = drive("a", [row("a"), row("b")], tmp_path / "j.jsonl")
    assert not result.quit_early
    assert result.ruled == 1


def test_unrecognized_key_is_ignored_and_the_row_stays(tmp_path):
    journal = tmp_path / "j.jsonl"
    drive("Za", [row("a")], journal)
    assert rulings.ruled_ids(journal) == {"a"}


# --- unchanged cohort ---------------------------------------------------------


def test_enter_accepts_the_auto_offer_on_an_unchanged_row(tmp_path):
    journal = tmp_path / "j.jsonl"
    drive("\r", [unchanged_row("a")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings["a"]["ground_truth"] == "approve"
    assert state.rulings["a"]["auto"] is True


def test_operator_can_override_the_auto_offer(tmp_path):
    journal = tmp_path / "j.jsonl"
    key, code = next(iter(rulings.reject_reason_keymap().items()))
    drive(f"r{key}", [unchanged_row("a")], journal)
    state = rulings.replay(rulings.read_records(journal))
    assert state.rulings["a"]["ground_truth"] == "reject"
    assert state.rulings["a"]["auto"] is False
    assert state.rulings["a"]["reason"] == code.value


def test_enter_on_a_changed_row_does_nothing(tmp_path):
    """No auto offer outside the sanity cohort — Enter must not approve blind."""
    journal = tmp_path / "j.jsonl"
    drive("\r", [row("a")], journal)
    assert rulings.read_records(journal) == []


# --- notes --------------------------------------------------------------------


def test_notes_key_journals_an_annotation_when_enabled(tmp_path):
    journal = tmp_path / "j.jsonl"
    drive("na", [row("a")], journal, lines=["brand casing edge case"], notes_enabled=True)
    state = rulings.replay(rulings.read_records(journal))
    assert state.notes == {"a": "brand casing edge case"}
    assert state.rulings["a"]["ground_truth"] == "approve"


def test_empty_note_is_not_journaled(tmp_path):
    journal = tmp_path / "j.jsonl"
    drive("na", [row("a")], journal, lines=[""], notes_enabled=True)
    assert rulings.replay(rulings.read_records(journal)).notes == {}


def test_notes_key_is_inert_without_the_flag(tmp_path):
    journal = tmp_path / "j.jsonl"
    drive("na", [row("a")], journal, lines=["ignored"], notes_enabled=False)
    assert rulings.replay(rulings.read_records(journal)).notes == {}


# --- progress -----------------------------------------------------------------


def test_session_reports_throughput(tmp_path):
    clock = iter([0.0, 60.0, 60.0, 60.0])
    result, _ = drive("aa", [row("a"), row("b")], tmp_path / "j.jsonl",
                      now=lambda: next(clock, 60.0))
    assert result.ruled == 2
    assert result.decisions_per_minute == 2.0


def test_render_stats_lists_the_tally():
    result = rule.SessionResult(total=2, ruled=1, approved=1, skipped=1, elapsed_s=30.0)
    rendered = rule.render_stats(result)
    assert "1" in rendered and "skipped" in rendered.lower()


def test_zero_elapsed_time_does_not_divide_by_zero():
    assert rule.SessionResult(ruled=1, elapsed_s=0.0).decisions_per_minute == 0.0


def test_rate_is_withheld_when_the_session_was_too_short_to_measure():
    """A sub-second scripted run must not report 181,000 decisions/min."""
    rendered = rule.render_stats(rule.SessionResult(total=1, ruled=1, elapsed_s=0.02))
    assert "decisions/min" not in rendered


def test_rate_is_reported_once_there_is_enough_elapsed_time():
    rendered = rule.render_stats(rule.SessionResult(total=2, ruled=2, elapsed_s=60.0))
    assert "2.0 decisions/min" in rendered


# --- ordering -----------------------------------------------------------------


def stats_for(**by_id) -> dict:
    """Flip stats hand-built from {id: flip_rate}, via real verdict records."""
    verdicts = []
    for pair_id, rate in by_id.items():
        # rate is achieved as (1 - modal share) over four votes
        minority = round(rate * 4)
        for index in range(4):
            value = "reject" if index < minority else "approve"
            reason = ReasonCode.CASING_ERROR if value == "reject" else ReasonCode.OK
            verdicts.append(Verdict(pair_id, value, reason, f"m{index}", "v1", 0.0))
    return flips.flip_stats(verdicts)


def test_contested_rows_come_first_by_default():
    rows = [row("calm"), row("split"), row("mild")]
    stats = stats_for(calm=0.0, split=0.5, mild=0.25)
    ordered = rule.order_rows(rows, stats)
    assert [r["id"] for r in ordered] == ["split", "mild", "calm"]


def test_ties_keep_pack_order():
    rows = [row("a"), row("b"), row("c")]
    ordered = rule.order_rows(rows, stats_for(a=0.5, b=0.5, c=0.5))
    assert [r["id"] for r in ordered] == ["a", "b", "c"]


def test_rows_the_sweep_never_judged_go_last_in_pack_order():
    """No verdicts is not the same as no disagreement — don't rank it as calm."""
    rows = [row("unjudged1"), row("calm"), row("unjudged2"), row("split")]
    ordered = rule.order_rows(rows, stats_for(calm=0.0, split=0.5))
    assert [r["id"] for r in ordered] == ["split", "calm", "unjudged1", "unjudged2"]


def test_a_pair_the_judges_could_not_decide_comes_first():
    # Issue #21, the ordering half. A card that shouts UNDECIDED is worthless
    # if the row sits at position 180 of a 200-row pass. The pair the judges
    # deadlocked on is the pair most in need of a human, so it leads.
    verdicts = (
        # e10-tie2 shape: unanimous reject, reason dead even 14/14. Its
        # VERDICT flip rate is 0.0, so ranking on flip rate alone filed it with
        # the pairs everyone agreed on — behind a unanimous row, in fact.
        [Verdict("tied-reason", "reject", ReasonCode.OK, f"a{i}", "v1", 0.0) for i in range(14)]
        + [
            Verdict("tied-reason", "reject", ReasonCode.CASING_ERROR, f"b{i}", "v1", 0.0)
            for i in range(14)
        ]
        + [Verdict("calm", "approve", ReasonCode.OK, f"c{i}", "v1", 0.0) for i in range(4)]
        + [
            Verdict("wobbly", "approve", ReasonCode.OK, "d0", "v1", 0.0),
            Verdict("wobbly", "reject", ReasonCode.CASING_ERROR, "d1", "v1", 0.0),
            Verdict("wobbly", "approve", ReasonCode.OK, "d2", "v1", 0.0),
        ]
    )
    stats = flips.flip_stats(verdicts)
    rows = [row("calm"), row("tied-reason"), row("wobbly")]
    ordered = [r["id"] for r in rule.order_rows(rows, stats)]
    assert ordered[0] == "tied-reason"
    # A merely-wobbly pair still beats a calm one.
    assert ordered == ["tied-reason", "wobbly", "calm"]


def test_a_tied_verdict_also_leads():
    verdicts = [
        Verdict("deadlock", "approve", ReasonCode.OK, f"a{i}", "v1", 0.0) for i in range(16)
    ] + [
        Verdict("deadlock", "reject", ReasonCode.CASING_ERROR, f"b{i}", "v1", 0.0)
        for i in range(16)
    ]
    verdicts += [
        Verdict("wobbly", "approve", ReasonCode.OK, "d0", "v1", 0.0),
        Verdict("wobbly", "reject", ReasonCode.CASING_ERROR, "d1", "v1", 0.0),
        Verdict("wobbly", "approve", ReasonCode.OK, "d2", "v1", 0.0),
    ]
    stats = flips.flip_stats(verdicts)
    ordered = [r["id"] for r in rule.order_rows([row("wobbly"), row("deadlock")], stats)]
    assert ordered[0] == "deadlock"


def test_without_sweep_data_pack_order_is_preserved():
    rows = [row("a"), row("b"), row("c")]
    assert [r["id"] for r in rule.order_rows(rows, {})] == ["a", "b", "c"]


def test_pack_order_can_be_asked_for_explicitly():
    rows = [row("calm"), row("split")]
    stats = stats_for(calm=0.0, split=0.5)
    ordered = rule.order_rows(rows, stats, order=rule.ORDER_PACK)
    assert [r["id"] for r in ordered] == ["calm", "split"]


def test_ordering_does_not_mutate_the_caller_s_list():
    rows = [row("calm"), row("split")]
    rule.order_rows(rows, stats_for(calm=0.0, split=0.5))
    assert [r["id"] for r in rows] == ["calm", "split"]


def test_ordering_drops_no_rows():
    rows = [row(str(n)) for n in range(6)]
    ordered = rule.order_rows(rows, stats_for(**{"1": 0.5, "4": 0.25}))
    assert sorted(r["id"] for r in ordered) == sorted(r["id"] for r in rows)


# --- CLI ----------------------------------------------------------------------


def test_run_command_resumes_past_already_ruled_rows(tmp_path):
    template = tmp_path / "template.jsonl"
    rule.write_jsonl(template, [row("a"), row("b")])
    journal = tmp_path / "j.jsonl"
    rulings.append(journal, rulings.rule_record("a", "approve", ReasonCode.OK))

    out = io.StringIO()
    rule.main(
        ["run", "--template", str(template), "--journal", str(journal)],
        read_key=rule.keys_from_string("s"),
        out=out,
    )
    state = rulings.replay(rulings.read_records(journal))
    assert state.skipped == ["b"]


def test_run_command_reports_when_everything_is_ruled(tmp_path):
    template = tmp_path / "template.jsonl"
    rule.write_jsonl(template, [row("a")])
    journal = tmp_path / "j.jsonl"
    rulings.append(journal, rulings.rule_record("a", "approve", ReasonCode.OK))

    out = io.StringIO()
    rule.main(["run", "--template", str(template), "--journal", str(journal)],
              read_key=rule.keys_from_string(""), out=out)
    assert "nothing left" in out.getvalue().lower()


def write_results_dir(directory, verdicts) -> None:
    from judge.schema import verdict_to_json_line

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "m1.jsonl").write_text(
        "".join(verdict_to_json_line(v) + "\n" for v in verdicts), encoding="utf-8"
    )


def test_run_presents_the_most_contested_row_first(tmp_path):
    """The default with sweep data present: contested-first, no flag needed."""
    template = tmp_path / "template.jsonl"
    rule.write_jsonl(template, [row("calm"), row("split")])
    results = tmp_path / "results"
    write_results_dir(
        results,
        [
            Verdict("calm", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("calm", "approve", ReasonCode.OK, "m2", "v1", 0.0),
            Verdict("split", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("split", "reject", ReasonCode.CASING_ERROR, "m2", "v1", 0.0),
        ],
    )
    journal = tmp_path / "j.jsonl"

    out = io.StringIO()
    rule.main(
        ["run", "--template", str(template), "--journal", str(journal),
         "--results", str(results)],
        read_key=rule.keys_from_string("a"),
        out=out,
    )
    assert rulings.ruled_ids(journal) == {"split"}


def test_order_pack_flag_restores_pack_order(tmp_path):
    template = tmp_path / "template.jsonl"
    rule.write_jsonl(template, [row("calm"), row("split")])
    results = tmp_path / "results"
    write_results_dir(
        results,
        [
            Verdict("calm", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("split", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("split", "reject", ReasonCode.CASING_ERROR, "m2", "v1", 0.0),
        ],
    )
    journal = tmp_path / "j.jsonl"

    rule.main(
        ["run", "--template", str(template), "--journal", str(journal),
         "--results", str(results), "--order", "pack"],
        read_key=rule.keys_from_string("a"),
        out=io.StringIO(),
    )
    assert rulings.ruled_ids(journal) == {"calm"}


def test_run_without_results_keeps_pack_order(tmp_path):
    template = tmp_path / "template.jsonl"
    rule.write_jsonl(template, [row("first"), row("second")])
    journal = tmp_path / "j.jsonl"

    rule.main(
        ["run", "--template", str(template), "--journal", str(journal)],
        read_key=rule.keys_from_string("a"),
        out=io.StringIO(),
    )
    assert rulings.ruled_ids(journal) == {"first"}


def test_ordering_applies_to_what_resume_left_pending(tmp_path):
    """Rank the rows still to do — a ruled row must not reclaim the front."""
    template = tmp_path / "template.jsonl"
    rule.write_jsonl(template, [row("calm"), row("done"), row("split")])
    results = tmp_path / "results"
    write_results_dir(
        results,
        [
            Verdict("calm", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("calm", "approve", ReasonCode.OK, "m2", "v1", 0.0),
            Verdict("done", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("done", "reject", ReasonCode.CASING_ERROR, "m2", "v1", 0.0),
            Verdict("split", "approve", ReasonCode.OK, "m1", "v1", 0.0),
            Verdict("split", "reject", ReasonCode.CASING_ERROR, "m2", "v1", 0.0),
        ],
    )
    journal = tmp_path / "j.jsonl"
    rulings.append(journal, rulings.rule_record("done", "approve", ReasonCode.OK))

    rule.main(
        ["run", "--template", str(template), "--journal", str(journal),
         "--results", str(results)],
        read_key=rule.keys_from_string("a"),
        out=io.StringIO(),
    )
    assert rulings.ruled_ids(journal) == {"done", "split"}


def test_run_says_which_order_it_used(tmp_path):
    """The operator must know why row 1 is row 1."""
    template = tmp_path / "template.jsonl"
    rule.write_jsonl(template, [row("a")])
    results = tmp_path / "results"
    write_results_dir(results, [Verdict("a", "approve", ReasonCode.OK, "m1", "v1", 0.0)])

    out = io.StringIO()
    rule.main(
        ["run", "--template", str(template), "--journal", str(tmp_path / "j.jsonl"),
         "--results", str(results)],
        read_key=rule.keys_from_string("q"),
        out=out,
    )
    assert "most contested first" in out.getvalue().lower()


def test_export_command_writes_the_merge_ready_rulings(tmp_path):
    journal = tmp_path / "j.jsonl"
    rulings.append(journal, rulings.rule_record("a", "approve", ReasonCode.OK, notes="n"))
    out_path = tmp_path / "rulings.jsonl"

    rule.main(["export", "--journal", str(journal), "--out", str(out_path)], out=io.StringIO())
    assert '"ground_truth": "approve"' in out_path.read_text()
    assert "notes" not in out_path.read_text()


def test_export_command_writes_notes_to_a_sidecar(tmp_path):
    journal = tmp_path / "j.jsonl"
    rulings.append(journal, rulings.rule_record("a", "approve", ReasonCode.OK, notes="why"))
    notes_path = tmp_path / "notes.jsonl"

    rule.main(
        ["export", "--journal", str(journal), "--out", str(tmp_path / "r.jsonl"),
         "--notes-out", str(notes_path)],
        out=io.StringIO(),
    )
    assert "why" in notes_path.read_text()
