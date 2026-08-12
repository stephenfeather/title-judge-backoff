"""Verdict and calibration-pair records with JSONL (de)serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum

VALID_VERDICTS = ("approve", "reject")


class ReasonCode(str, Enum):
    OVERCORRECTION = "overcorrection"
    MEANING_CHANGE = "meaning_change"
    CASING_ERROR = "casing_error"
    TRUNCATION_WORSE = "truncation_worse"  # RETIRED — see RETIRED_REASON_CODES
    OK = "ok"


#: Codes withdrawn from the rubric but kept in the enum, because verdict rows
#: on disk still carry them and `verdict_from_json_line` would refuse to read a
#: historical run without the member (issue #44).
#:
#: `truncation_worse` was retired at PROMPT_VERSION v3. It fired 4 times in
#: ~13,800 votes, 3 of them from the degenerate floor model, and an audit of the
#: 200-pair calibration set found no genuine truncation to fire on: exactly 3
#: pairs drop an original token that abbreviation expansion cannot account for,
#: and all 3 are expansions the tokenizer split ("W/SHEATH" -> "with Sheath",
#: "B/C" -> "Birchwood Casey"). No dropped token anywhere in the corpus contains
#: a digit, so no caliber, capacity, barrel length or model number is ever lost.
#:
#: Retired for THIS corpus, not judged wrong in general. A vendor feed that
#: hard-truncates at ingestion would produce real cases; they are absent from
#: these 200 rows.
RETIRED_REASON_CODES = frozenset({ReasonCode.TRUNCATION_WORSE})

#: The codes the rubric currently offers. Everything that presents a choice to a
#: model or an operator derives from this, so a retirement cannot be honoured in
#: one place and forgotten in another.
ACTIVE_REASON_CODES = tuple(code for code in ReasonCode if code not in RETIRED_REASON_CODES)


def _check_verdict(value: str) -> str:
    if value not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {value!r}")
    return value


@dataclass(frozen=True)
class Pair:
    """One calibration pair: a before/after title change, optionally ruled.

    `ground_truth`/`reason` are None for an UNRULED row. The E10 pack ships 200
    rows with no operator verdicts, and plenty is measurable before rulings
    exist — flip rates, cross-model agreement, reason distribution — so judging
    an unruled row is a first-class case, not an error. Anything that compares
    against ground truth must check `is_ruled` first; see score.py, which
    refuses to compute kappa on unruled data rather than treating None as a
    label.
    """

    id: str
    original: str
    enriched: str
    # brand/mpn are absent from some sources (e.g. the E10 QA pack), in which
    # case the judge prompt simply omits those lines.
    brand: str | None = None
    mpn: str | None = None
    ground_truth: str | None = None
    reason: ReasonCode | None = None

    def __post_init__(self) -> None:
        if (self.ground_truth is None) != (self.reason is None):
            raise ValueError(
                f"pair {self.id!r}: ground_truth and reason must both be set or both be "
                f"absent, got ground_truth={self.ground_truth!r}, reason={self.reason!r}"
            )
        if self.ground_truth is not None:
            _check_verdict(self.ground_truth)

    @property
    def is_ruled(self) -> bool:
        return self.ground_truth is not None


@dataclass(frozen=True)
class Usage:
    """Tokens one call actually spent, as the host reported them.

    Captured rather than estimated. Before this existed, cost modelling had to
    derive a chars/token ratio from an unrelated metering call because no run
    had recorded a single token count.

    Every field is optional because hosts differ in what they report, and an
    absent count must stay absent: defaulting to 0 would read as a free call
    and quietly understate a bill.

    Two fields earn their place beyond raw cost:

    * `reasoning_tokens` bill as output. gpt-5.6-luna at effort medium spends
      ~120 per call, roughly doubling that backend's real cost, and none of it
      is visible in `completion_tokens` alone on every host.
    * `cached_tokens` is the tell for a host serving cached responses. The
      three votes of a majority-of-3 send byte-identical requests, so an
      undetected response cache would collapse the vote to n=1 and drive the
      flip rate to a spurious 0.0 — a failure that IMPROVES every metric while
      measuring nothing.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None  # prompt tokens served from cache
    reasoning_tokens: int | None = None  # billed as output, absent from some hosts


def _first_int(source: dict, *keys: str) -> int | None:
    """The first key present with an int value, or None.

    Explicitly key-presence based rather than `a or b`: a real 0 — no reasoning
    spent at effort=none — is a measurement, and truthiness chaining discards
    it as though the host had said nothing.
    """
    for key in keys:
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def usage_from_payload(payload: dict) -> Usage | None:
    """The `usage` block of a response, normalized across API dialects.

    Reads both families this harness talks to and tolerates either naming,
    because "OpenAI-compatible" hosts vary in which spelling they emit:

        /chat/completions   prompt_tokens / completion_tokens / total_tokens
                            prompt_tokens_details.cached_tokens
                            completion_tokens_details.reasoning_tokens
        /messages           input_tokens / output_tokens, no total
                            cache_read_input_tokens

    Returns None when the host reported nothing usable, which is a different
    statement from "this call was free". Unrecognized extra keys are dropped:
    the fields here are the ones cost modelling and cache detection need, and
    carrying an arbitrary per-host dict onto every verdict row would bloat the
    results file with something nothing reads.
    """
    block = payload.get("usage")
    if not isinstance(block, dict) or not block:
        return None

    prompt = _first_int(block, "prompt_tokens", "input_tokens")
    completion = _first_int(block, "completion_tokens", "output_tokens")
    total = _first_int(block, "total_tokens")
    if total is None and prompt is not None and completion is not None:
        # Anthropic sends no total; deriving it keeps the field comparable
        # across backends rather than leaving one dialect permanently blank.
        total = prompt + completion

    prompt_details = block.get("prompt_tokens_details") or block.get("input_tokens_details") or {}
    completion_details = (
        block.get("completion_tokens_details") or block.get("output_tokens_details") or {}
    )
    cached = _first_int(block, "cache_read_input_tokens")
    if cached is None and isinstance(prompt_details, dict):
        cached = _first_int(prompt_details, "cached_tokens")
    reasoning = (
        _first_int(completion_details, "reasoning_tokens")
        if isinstance(completion_details, dict)
        else None
    )

    usage = Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )
    # A block of keys we recognized none of is the same as no block at all.
    return usage if usage != Usage() else None


@dataclass(frozen=True)
class Verdict:
    """One judge ruling on one pair, tagged with the model/prompt that produced it.

    `temperature` is None when the field was omitted from the request rather
    than set — the GPT-5.x family rejects any explicit value above reasoning
    effort `none`, and recording 0.0 for a call that never sent it would label
    a sampled verdict as deterministic.

    `run_index` distinguishes the N votes of a majority-vote run: the same pair
    judged three times yields three verdicts that differ only in this field.
    """

    pair_id: str
    verdict: str
    reason: ReasonCode
    model_id: str
    prompt_version: str
    temperature: float | None
    run_index: int = 0
    reasoning_effort: str | None = None
    usage: Usage | None = None  # None = the host reported nothing, see Usage

    # Provenance (issue #13): where this row came from, not just what it says.
    # None on all three means ONLY "written before provenance existed" — the
    # write path refuses to produce a row it cannot identify, so a null here is
    # always a legacy row and never a live unknown.
    base_url: str | None = None  # the host that served the call
    config_digest: str | None = None  # see judge.client.config_digest
    code_version: str | None = None  # see judge.provenance.code_version

    def __post_init__(self) -> None:
        _check_verdict(self.verdict)


def pair_from_dict(record: dict) -> Pair:
    """Build a Pair; a missing or null ruling yields an unruled pair.

    The ruling-template rows produced by adapt_qa_pack carry explicit nulls,
    and the pack itself carries no verdicts at all — both must load.
    """
    reason = record.get("reason")
    return Pair(
        id=record["id"],
        original=record["original"],
        enriched=record["enriched"],
        brand=record.get("brand"),
        mpn=record.get("mpn"),
        ground_truth=record.get("ground_truth"),
        reason=ReasonCode(reason) if reason is not None else None,
    )


def verdict_to_json_line(verdict: Verdict) -> str:
    record = asdict(verdict)
    record["reason"] = verdict.reason.value
    return json.dumps(record, ensure_ascii=False)


def verdict_from_json_line(line: str) -> Verdict:
    record = json.loads(line)
    record["reason"] = ReasonCode(record["reason"])
    # Results written before majority voting carry neither field; default them
    # so old result directories stay readable instead of raising.
    record.setdefault("run_index", 0)
    record.setdefault("reasoning_effort", None)
    # Provenance predates none of results/: every existing row lacks all three.
    for field in ("base_url", "config_digest", "code_version"):
        record.setdefault(field, None)
    # Same for usage: every row written before it was captured has no key, and
    # must load as "not measured" rather than raising.
    usage = record.get("usage")
    record["usage"] = Usage(**usage) if isinstance(usage, dict) else None
    return Verdict(**record)
