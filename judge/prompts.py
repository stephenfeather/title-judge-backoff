"""Versioned judge prompts and response parsing.

Prompt v3: the judge sees an original title, the enriched replacement, and the
known brand/MPN WHEN THE PAIR CARRIES THEM, then must approve or reject the
change with a reason code.

v3 withdrew `truncation_worse` from the rubric (issue #44). It fired 4 times in
~13,800 votes across two prompt versions, 3 of them from the degenerate floor
model that picks reason codes close to randomly, and an audit of the corpus
found no genuine truncation for it to describe — see RETIRED_REASON_CODES in
judge/schema.py for the evidence. A code no model emits and no operator can
legitimately choose is a permanently empty row and column in score.py's
confusion matrix; worse, if an operator DID rule some pairs that way, every one
would be a systematic miss attributed to the model rather than to a rubric the
data cannot exercise.

Withdrawing it changes what the judge is asked, so PROMPT_VERSION moved to v3
and v2 verdicts are not resumable against it. That is deliberate: the reason
distribution is the thing under measurement, and silently mixing a run that
offered five codes with one that offered four would corrupt exactly the number
this change exists to clean up.

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

from judge.schema import ACTIVE_REASON_CODES, ReasonCode, _check_verdict

PROMPT_VERSION = "v3"

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

If the change is acceptable, approve with reason code:
- ok: the enriched title is a faithful improvement

Respond with ONLY a JSON object, no other text:
{"verdict": "approve" | "reject", "reason": "overcorrection" | "meaning_change" | "casing_error" | "ok"}
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


def prompt_variant(pair) -> str:
    """Which system-prompt variant this pair renders: the manifest's unit.

    Derived from `_supplied_attributes`, so it names the variant that was
    actually sent rather than a separate description that could drift from it.
    """
    return "+".join(label.lower() for label in _supplied_attributes(pair)) or "none"


def prompt_variant_counts(pairs) -> dict[str, int]:
    """How many pairs render each system-prompt variant.

    JOINT counts, not per-attribute marginals. Marginals cannot identify the
    prompt mix: brand=1, mpn=1 is either one pair carrying both (one variant) or
    two pairs carrying one each (two variants), and v2 renders a different
    prompt for each. Since the manifest's request_payload samples only pairs[0],
    this histogram is the only thing that can reconstruct what was sent.
    """
    counts: dict[str, int] = {}
    for pair in pairs:
        variant = prompt_variant(pair)
        counts[variant] = counts.get(variant, 0) + 1
    return counts


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


class JudgeResponseError(ValueError):
    """The model answered, but not in the shape the contract requires.

    Deliberately its own type, because `RunHealth.record_failure` counts
    failures by `type(exc).__name__`. A bare ValueError lands in `error_kinds`
    under a name that says nothing about the cause and collides with any other
    ValueError raised anywhere in the call path.

    The distinction is not cosmetic (issue #41). A transport failure says
    nothing about the model — the connection dropped. THIS says the model was
    asked for `{"verdict": "approve"|"reject", "reason": ...}` and returned
    something else, which is evidence about its fitness as a judge. Merging the
    two means a model that cannot follow the output contract looks merely
    unlucky, and its failed pairs vanish from its own scores.

    Subclasses ValueError so existing callers that catch ValueError keep
    working.
    """


def parse_judge_response(text: str) -> tuple[str, ReasonCode]:
    """Extract (verdict, reason) from a judge reply.

    Tolerates surrounding prose and markdown code fences; raises
    JudgeResponseError if no well-formed verdict object can be found.

    All five failure shapes — no JSON, malformed JSON, a non-verdict in the
    verdict field, an unknown reason code, a RETIRED reason code — raise the
    SAME type. They are one finding: the model did not honour the contract it
    was given. The prompt does not offer retired codes, so a reply carrying
    one is not a historical row to preserve (those never pass through here) —
    it is a live contract breach, and must count with the others in
    error_kinds rather than enter the v3 reason distribution as a code the
    model was never offered (issue #44). Splitting any of these across
    different exception names would scatter one compliance rate across several
    rows of error_kinds.
    """
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        raise JudgeResponseError(f"no JSON object in judge response: {text!r}")
    try:
        record = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"malformed JSON in judge response: {text!r}") from exc
    try:
        verdict = _check_verdict(record.get("verdict"))
        reason = ReasonCode(record.get("reason"))
        if reason not in ACTIVE_REASON_CODES:
            raise ValueError(f"reason code {reason.value!r} is retired and not offered")
    except ValueError as exc:
        # _check_verdict and ReasonCode() both raise plain ValueError. Re-raise
        # under this type so the count lands with the other contract failures.
        raise JudgeResponseError(f"{exc} (in judge response: {text!r})") from exc
    return verdict, reason
