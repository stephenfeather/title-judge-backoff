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
    """One judge ruling on one pair, tagged with the model/prompt that produced it."""

    pair_id: str
    verdict: str
    reason: ReasonCode
    model_id: str
    prompt_version: str
    temperature: float

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
    return Verdict(**record)
