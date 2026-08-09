from dataclasses import replace

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


BARE_PAIR = Pair(
    id="p2",
    original="acme widget 3000 blk",
    enriched="Acme Widget 3000, Black",
    brand=None,
    mpn=None,
    ground_truth="approve",
    reason=ReasonCode.OK,
)


def test_prompt_version_is_v2():
    # Bumped for the issue #14 fix. This correctly invalidates resume against
    # every result file produced under v1, which is the #13 guard working.
    assert PROMPT_VERSION == "v2"


def test_system_prompt_does_not_promise_brand_or_mpn_when_the_pair_has_neither():
    # Issue #14: v1 promised "the product's known BRAND and MPN" unconditionally
    # while the user message omitted them unless the pair carried them. Every
    # S1 judgment was made by a model told to expect identifying information it
    # never received.
    system = build_messages(BARE_PAIR)[0]["content"]
    assert "BRAND" not in system
    assert "MPN" not in system


def test_system_prompt_promises_brand_and_mpn_when_the_pair_carries_them():
    # The other half of the defect: stripping the promise outright would go
    # wrong the moment a corpus supplies the attributes, because the user
    # message would carry fields the prompt never mentioned.
    system = build_messages(PAIR)[0]["content"]
    assert "BRAND" in system
    assert "MPN" in system


def test_system_prompt_promises_only_the_attribute_the_pair_actually_has():
    brand_only = replace(BARE_PAIR, brand="Acme")
    system = build_messages(brand_only)[0]["content"]
    assert "BRAND" in system
    assert "MPN" not in system

    mpn_only = replace(BARE_PAIR, mpn="W3000-BLK")
    system = build_messages(mpn_only)[0]["content"]
    assert "MPN" in system
    assert "BRAND" not in system


def test_system_prompt_and_user_message_always_agree_on_attributes():
    # The invariant the whole fix exists to hold: the prompt describes exactly
    # the fields the user message carries, for every combination.
    for brand, mpn in [(None, None), ("Acme", None), (None, "W3000-BLK"), ("Acme", "W3000-BLK")]:
        system, user = (m["content"] for m in build_messages(replace(BARE_PAIR, brand=brand, mpn=mpn)))
        assert ("BRAND" in system) == ("BRAND:" in user), f"brand={brand!r} mpn={mpn!r}"
        assert ("MPN" in system) == ("MPN:" in user), f"brand={brand!r} mpn={mpn!r}"


def test_judging_criteria_are_identical_across_attribute_variants():
    # Only the input description may vary. The criteria and reason codes must
    # not, or the reason distribution would shift for reasons unrelated to the
    # inputs — and score.py builds its confusion matrix from those codes.
    with_attrs = build_messages(PAIR)[0]["content"]
    without = build_messages(BARE_PAIR)[0]["content"]
    criteria = "Approve the change only if"
    assert with_attrs[with_attrs.index(criteria):] == without[without.index(criteria):]


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
