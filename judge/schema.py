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
    TRUNCATION_WORSE = "truncation_worse"
    OK = "ok"


def _check_verdict(value: str) -> str:
    if value not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {value!r}")
    return value


@dataclass(frozen=True)
class Pair:
    """One calibration pair: a before/after title change with operator ruling."""

    id: str
    original: str
    enriched: str
    # brand/mpn are absent from some sources (e.g. the E10 QA pack), in which
    # case the judge prompt simply omits those lines.
    brand: str | None
    mpn: str | None
    ground_truth: str
    reason: ReasonCode

    def __post_init__(self) -> None:
        _check_verdict(self.ground_truth)


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

    def __post_init__(self) -> None:
        _check_verdict(self.verdict)


def pair_from_dict(record: dict) -> Pair:
    return Pair(
        id=record["id"],
        original=record["original"],
        enriched=record["enriched"],
        brand=record.get("brand"),
        mpn=record.get("mpn"),
        ground_truth=record["ground_truth"],
        reason=ReasonCode(record["reason"]),
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
    return Verdict(**record)
