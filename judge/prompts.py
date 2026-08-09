"""Versioned judge prompts and response parsing.

Prompt v2: the judge sees an original title, the enriched replacement, and the
known brand/MPN WHEN THE PAIR CARRIES THEM, then must approve or reject the
change with a reason code.

v1 promised brand and MPN unconditionally while `build_messages` only sent them
when the pair had them. The calibration corpus has neither field, so all 4,974
S1 judgments were made by a model told to expect identifying information it
never received — plausibly pushing it toward `truncation_worse` and
`meaning_change`, the two codes that would absorb that pressure, and those
codes are what score.py builds its confusion matrix from (issue #14).

The fix is to derive the promise from the same condition that decides the send,
so the two halves cannot disagree again. Stripping the promise outright would
have failed the other way round the moment a corpus supplied the attributes.
"""

from __future__ import annotations

import json
import re

from judge.schema import ReasonCode, _check_verdict

PROMPT_VERSION = "v2"

_INTRO = "You are a strict quality-control judge for e-commerce product titles.\n"

_INPUTS = "You will be shown an ORIGINAL product title and an ENRICHED replacement title"

# Everything below the input description is FIXED across variants. Only what the
# judge is told it will receive may change; what it is asked to decide may not,
# or the reason distribution would shift for reasons unrelated to the inputs.
_CRITERIA = """\

Approve the change only if the enriched title is a faithful, improved version
of the original: same product, same meaning, correct brand casing, no invented
attributes, and no loss of identifying detail.

Reject the change if any of these apply, and pick the single best reason code:
- overcorrection: the rewrite changed things that were already correct, or scrubbed useful detail
- meaning_change: the enriched title describes a different product, spec, or attribute than the original
- casing_error: brand or model casing is wrong in the enriched title
- truncation_worse: the enriched title dropped identifying information the original had

If the change is acceptable, approve with reason code:
- ok: the enriched title is a faithful improvement

Respond with ONLY a JSON object, no other text:
{"verdict": "approve" | "reject", "reason": "overcorrection" | "meaning_change" | "casing_error" | "truncation_worse" | "ok"}
"""

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _supplied_attributes(pair) -> list[str]:
    """The attribute labels this pair will actually put in the user message.

    Single source of truth for both halves of the prompt: the system prompt
    promises exactly what the user message carries, because both are built from
    this list. That is the invariant issue #14 was the absence of.
    """
    supplied = []
    if pair.brand:
        supplied.append("BRAND")
    if pair.mpn:
        supplied.append("MPN")
    return supplied


def build_system_prompt(pair) -> str:
    """The system prompt for one pair, describing only the inputs it will get."""
    supplied = _supplied_attributes(pair)
    if not supplied:
        return f"{_INTRO}{_INPUTS}.\n{_CRITERIA}"
    described = " and ".join(
        "MPN (manufacturer part number)" if label == "MPN" else label for label in supplied
    )
    return f"{_INTRO}{_INPUTS},\nplus the product's known {described}.\n{_CRITERIA}"


def build_messages(pair) -> list[dict[str, str]]:
    """Build the chat messages for one calibration pair."""
    values = {"BRAND": pair.brand, "MPN": pair.mpn}
    lines = [f"{label}: {values[label]}" for label in _supplied_attributes(pair)]
    lines.append(f"ORIGINAL: {pair.original}")
    lines.append(f"ENRICHED: {pair.enriched}")
    user = "\n".join(lines)
    return [
        {"role": "system", "content": build_system_prompt(pair)},
        {"role": "user", "content": user},
    ]


def parse_judge_response(text: str) -> tuple[str, ReasonCode]:
    """Extract (verdict, reason) from a judge reply.

    Tolerates surrounding prose and markdown code fences; raises ValueError if
    no well-formed verdict object can be found.
    """
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        raise ValueError(f"no JSON object in judge response: {text!r}")
    try:
        record = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in judge response: {text!r}") from exc
    verdict = _check_verdict(record.get("verdict"))
    reason = ReasonCode(record.get("reason"))
    return verdict, reason
