import json

import pytest

from judge.schema import (
    Pair,
    ReasonCode,
    Usage,
    Verdict,
    pair_from_dict,
    usage_from_payload,
    verdict_from_json_line,
    verdict_to_json_line,
)


def make_verdict(**overrides):
    base = dict(
        pair_id="p1",
        verdict="approve",
        reason=ReasonCode.OK,
        model_id="meta/llama-3.1-8b-instruct",
        prompt_version="v1",
        temperature=0.0,
    )
    base.update(overrides)
    return Verdict(**base)


def test_verdict_records_omitted_temperature_as_none():
    # gpt-5.6 rejects temperature at any reasoning effort above `none`, so the
    # harness omits the field. Recording 0.0 in that case would label a sampled
    # verdict as deterministic.
    v = make_verdict(temperature=None)
    assert v.temperature is None
    assert json.loads(verdict_to_json_line(v))["temperature"] is None
    assert verdict_from_json_line(verdict_to_json_line(v)) == v


def test_verdict_carries_run_index_and_reasoning_effort():
    v = make_verdict(run_index=2, reasoning_effort="medium")
    assert v.run_index == 2
    assert v.reasoning_effort == "medium"
    assert verdict_from_json_line(verdict_to_json_line(v)) == v


def test_verdict_defaults_run_index_to_zero_and_effort_to_none():
    v = make_verdict()
    assert v.run_index == 0
    assert v.reasoning_effort is None


def test_verdict_from_json_line_tolerates_pre_voting_records():
    # Results written before majority voting existed have no run_index or
    # reasoning_effort; they read back as run 0 with unrecorded effort rather
    # than exploding, so old result dirs stay inspectable.
    legacy = json.dumps(
        {
            "pair_id": "p1",
            "verdict": "approve",
            "reason": "ok",
            "model_id": "m",
            "prompt_version": "v1",
            "temperature": 0.0,
        }
    )
    v = verdict_from_json_line(legacy)
    assert v.run_index == 0
    assert v.reasoning_effort is None


def test_verdict_roundtrips_through_jsonl():
    v = make_verdict()
    line = verdict_to_json_line(v)
    assert verdict_from_json_line(line) == v


def test_verdict_json_line_is_single_line_with_required_fields():
    line = verdict_to_json_line(make_verdict(reason=ReasonCode.MEANING_CHANGE, verdict="reject"))
    assert "\n" not in line
    record = json.loads(line)
    assert record["pair_id"] == "p1"
    assert record["verdict"] == "reject"
    assert record["reason"] == "meaning_change"
    assert record["model_id"] == "meta/llama-3.1-8b-instruct"
    assert record["prompt_version"] == "v1"
    assert record["temperature"] == 0.0


def test_verdict_rejects_invalid_verdict_value():
    with pytest.raises(ValueError):
        make_verdict(verdict="maybe")


def test_reason_code_members():
    assert {rc.value for rc in ReasonCode} == {
        "overcorrection",
        "meaning_change",
        "casing_error",
        "truncation_worse",
        "ok",
    }


def test_pair_from_dict():
    record = {
        "id": "p9",
        "original": "acme widget 3000 blk",
        "enriched": "Acme Widget 3000, Black",
        "brand": "Acme",
        "mpn": "W3000-BLK",
        "ground_truth": "approve",
        "reason": "ok",
    }
    pair = pair_from_dict(record)
    assert pair == Pair(
        id="p9",
        original="acme widget 3000 blk",
        enriched="Acme Widget 3000, Black",
        brand="Acme",
        mpn="W3000-BLK",
        ground_truth="approve",
        reason=ReasonCode.OK,
    )


def test_pair_accepts_an_unruled_row():
    # The E10 pack ships 200 rows with NO operator verdicts. Judging them is
    # useful before rulings exist (flip rates, cross-model agreement), so an
    # unruled pair must load rather than raise.
    pair = pair_from_dict(
        {
            "id": "e10-abc",
            "original": "acme widget 3000 blk",
            "enriched": "Acme Widget 3000, Black",
            "ground_truth": None,
            "reason": None,
        }
    )
    assert pair.ground_truth is None
    assert pair.reason is None
    assert pair.is_ruled is False


def test_pair_from_dict_treats_missing_ruling_fields_as_unruled():
    pair = pair_from_dict({"id": "e10-abc", "original": "a", "enriched": "b"})
    assert pair.is_ruled is False


def test_ruled_pair_reports_itself_as_ruled():
    pair = pair_from_dict(
        {
            "id": "p1",
            "original": "a",
            "enriched": "b",
            "ground_truth": "approve",
            "reason": "ok",
        }
    )
    assert pair.is_ruled is True


def test_pair_rejects_a_half_ruling():
    # A verdict with no reason (or vice versa) is a data-entry slip, not a
    # deliberate "unruled" — silently accepting it would produce a pair that
    # scores but has no reason code for the confusion matrix.
    with pytest.raises(ValueError, match="both"):
        pair_from_dict({"id": "p1", "original": "a", "enriched": "b", "ground_truth": "approve"})
    with pytest.raises(ValueError, match="both"):
        pair_from_dict({"id": "p1", "original": "a", "enriched": "b", "reason": "ok"})


def test_pair_from_dict_rejects_bad_ground_truth():
    with pytest.raises(ValueError):
        pair_from_dict(
            {
                "id": "p9",
                "original": "a",
                "enriched": "b",
                "brand": "x",
                "mpn": "y",
                "ground_truth": "yes",
                "reason": "ok",
            }
        )


# --- token usage (issue #11) -------------------------------------------------


def test_usage_from_an_openai_chat_completion_payload():
    # /chat/completions reports prompt_tokens/completion_tokens. Every backend
    # on the slate except the Anthropic one speaks this dialect.
    usage = usage_from_payload(
        {"usage": {"prompt_tokens": 412, "completion_tokens": 18, "total_tokens": 430}}
    )
    assert usage == Usage(prompt_tokens=412, completion_tokens=18, total_tokens=430)


def test_usage_captures_cached_and_reasoning_token_details():
    # cached_tokens is the tell for a host serving cached responses — the
    # failure that would collapse majority-of-3 to n=1 and drive flip rate to a
    # spurious 0.0. reasoning_tokens bill as output and roughly double a
    # reasoning backend's real cost.
    usage = usage_from_payload(
        {
            "usage": {
                "prompt_tokens": 412,
                "completion_tokens": 138,
                "total_tokens": 550,
                "prompt_tokens_details": {"cached_tokens": 384},
                "completion_tokens_details": {"reasoning_tokens": 120},
            }
        }
    )
    assert usage.cached_tokens == 384
    assert usage.reasoning_tokens == 120


def test_usage_from_an_anthropic_messages_payload():
    # Anthropic names them input/output and sends no total at all.
    usage = usage_from_payload({"usage": {"input_tokens": 300, "output_tokens": 20}})
    assert usage == Usage(prompt_tokens=300, completion_tokens=20, total_tokens=320)


def test_usage_reads_an_anthropic_cache_read():
    usage = usage_from_payload(
        {"usage": {"input_tokens": 300, "output_tokens": 20, "cache_read_input_tokens": 288}}
    )
    assert usage.cached_tokens == 288


def test_usage_is_none_when_the_host_sends_nothing_usable():
    # Not every OpenAI-compatible host populates usage. Absent must stay
    # absent rather than becoming zeros, which would read as a free call.
    assert usage_from_payload({}) is None
    assert usage_from_payload({"usage": None}) is None
    assert usage_from_payload({"usage": {}}) is None
    assert usage_from_payload({"usage": "not a dict"}) is None


def test_a_real_zero_is_recorded_rather_than_treated_as_missing():
    # 0 reasoning tokens at effort=none is a measurement, not an absence.
    # Chaining `a or b` on token counts would silently discard it.
    usage = usage_from_payload(
        {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 5,
                "total_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        }
    )
    assert usage.prompt_tokens == 0
    assert usage.reasoning_tokens == 0


def test_verdict_round_trips_its_usage():
    v = Verdict(
        pair_id="p1",
        verdict="approve",
        reason=ReasonCode.OK,
        model_id="m",
        prompt_version="v1",
        temperature=None,
        usage=Usage(prompt_tokens=412, completion_tokens=18, total_tokens=430),
    )
    assert verdict_from_json_line(verdict_to_json_line(v)) == v


def test_verdicts_written_before_usage_capture_still_load():
    # Every existing row in results/ predates this field. They must read as
    # "not measured" rather than raising, and must not look like a new config.
    line = json.dumps(
        {
            "pair_id": "p1",
            "verdict": "approve",
            "reason": "ok",
            "model_id": "m",
            "prompt_version": "v1",
            "temperature": None,
            "run_index": 0,
            "reasoning_effort": None,
        }
    )
    assert verdict_from_json_line(line).usage is None
