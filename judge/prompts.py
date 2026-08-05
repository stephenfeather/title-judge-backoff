"""Versioned judge prompts and response parsing.

Prompt v1: the judge sees an original title, the enriched replacement, and the
known brand/MPN, then must approve or reject the change with a reason code.
"""

from __future__ import annotations

import json
import re

from judge.schema import ReasonCode, _check_verdict

PROMPT_VERSION = "v1"

_SYSTEM_PROMPT_V1 = """\
You are a strict quality-control judge for e-commerce product titles.
You will be shown an ORIGINAL product title and an ENRICHED replacement title,
plus the product's known BRAND and MPN (manufacturer part number).

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


def build_messages(pair) -> list[dict[str, str]]:
    """Build the chat messages for one calibration pair."""
    user = (
        f"BRAND: {pair.brand}\n"
        f"MPN: {pair.mpn}\n"
        f"ORIGINAL: {pair.original}\n"
        f"ENRICHED: {pair.enriched}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT_V1},
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
