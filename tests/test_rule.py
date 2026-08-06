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
