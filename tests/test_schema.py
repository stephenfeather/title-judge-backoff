import json

import pytest

from judge.schema import Pair, ReasonCode, Verdict, pair_from_dict, verdict_from_json_line, verdict_to_json_line


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
