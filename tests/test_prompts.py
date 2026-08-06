import pytest

from judge.prompts import PROMPT_VERSION, build_messages, parse_judge_response
from judge.schema import Pair, ReasonCode


PAIR = Pair(
    id="p1",
    original="acme widget 3000 blk",
    enriched="Acme Widget 3000, Black",
    brand="Acme",
    mpn="W3000-BLK",
    ground_truth="approve",
    reason=ReasonCode.OK,
)


def test_prompt_version_is_v1():
    assert PROMPT_VERSION == "v1"


def test_build_messages_shape_and_content():
    messages = build_messages(PAIR)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    for value in (PAIR.original, PAIR.enriched, PAIR.brand, PAIR.mpn):
        assert value in user


def test_build_messages_system_prompt_names_all_reason_codes():
    system = build_messages(PAIR)[0]["content"]
    for code in ReasonCode:
        assert code.value in system


def test_build_messages_omits_brand_and_mpn_lines_when_absent():
    pair = Pair(
        id="p2",
        original="acme widget 3000 blk",
        enriched="Acme Widget 3000, Black",
        brand=None,
        mpn=None,
        ground_truth="approve",
        reason=ReasonCode.OK,
    )
    user = build_messages(pair)[1]["content"]
    assert "BRAND:" not in user
    assert "MPN:" not in user
    assert "ORIGINAL: acme widget 3000 blk" in user
    assert "ENRICHED: Acme Widget 3000, Black" in user


def test_parse_judge_response_plain_json():
    verdict, reason = parse_judge_response('{"verdict": "approve", "reason": "ok"}')
    assert verdict == "approve"
    assert reason == ReasonCode.OK


def test_parse_judge_response_json_in_code_fence():
    text = 'Here you go:\n```json\n{"verdict": "reject", "reason": "meaning_change"}\n```'
    verdict, reason = parse_judge_response(text)
    assert verdict == "reject"
    assert reason == ReasonCode.MEANING_CHANGE


def test_parse_judge_response_rejects_garbage():
    with pytest.raises(ValueError):
        parse_judge_response("I approve of this title.")


def test_parse_judge_response_rejects_unknown_reason():
    with pytest.raises(ValueError):
        parse_judge_response('{"verdict": "approve", "reason": "vibes"}')
