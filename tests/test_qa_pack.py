import json

import pytest

from judge.qa_pack import (
    PackRow,
    merge_rulings,
    parse_qa_pack,
    row_id,
    template_line,
)

PACK = """\
# Title enrichment — operator QA pack

Prose that must be ignored, including a stray | pipe.

| Cohort | Source | Before | After | Stages |
|---|---|---|---|---|
| most-changed | davidsons | WRA 1892 DLX SR CRB | Winchester Repeating Arms 1892 Deluxe Sr Carbine | lexicon, case, brand |
| representative | zanders | RUGER LCRx .9MM 3 | Ruger LCRx .9MM 3 | case |
| unchanged | zanders | GAMO SWARM VIPER 10X GEN3i | GAMO SWARM VIPER 10X GEN3i | — |
"""


def test_parse_qa_pack_extracts_rows_and_ignores_prose():
    rows = parse_qa_pack(PACK)
    assert len(rows) == 3
    assert rows[0] == PackRow(
        cohort="most-changed",
        source="davidsons",
        before="WRA 1892 DLX SR CRB",
        after="Winchester Repeating Arms 1892 Deluxe Sr Carbine",
        stages=["lexicon", "case", "brand"],
    )


def test_parse_qa_pack_reads_em_dash_stages_as_empty():
    assert parse_qa_pack(PACK)[2].stages == []


def test_parse_qa_pack_skips_the_header_separator_row():
    assert all(row.cohort != "---" for row in parse_qa_pack(PACK))


def test_row_id_is_stable_and_content_derived():
    rows = parse_qa_pack(PACK)
    assert row_id(rows[0]) == row_id(rows[0])
    # Same content in a different pack ordering keeps the same id.
    reordered = parse_qa_pack(PACK)[::-1]
    assert row_id(reordered[-1]) == row_id(rows[0])
    assert row_id(rows[0]) != row_id(rows[1])


def test_template_line_carries_context_and_blank_verdict():
    row = parse_qa_pack(PACK)[0]
    record = json.loads(template_line(row))
    assert record["id"] == row_id(row)
    assert record["original"] == row.before
    assert record["enriched"] == row.after
    assert record["cohort"] == "most-changed"
    assert record["source"] == "davidsons"
    assert record["ground_truth"] is None
    assert record["reason"] is None


def test_template_line_omits_brand_and_mpn_when_not_joined():
    record = json.loads(template_line(parse_qa_pack(PACK)[0]))
    assert "brand" not in record
    assert "mpn" not in record


def test_template_line_includes_brand_and_mpn_when_joined():
    row = parse_qa_pack(PACK)[0]
    record = json.loads(template_line(row, attributes={"brand": "Winchester", "mpn": "WRA-1892"}))
    assert record["brand"] == "Winchester"
    assert record["mpn"] == "WRA-1892"


def test_merge_rulings_produces_pairs():
    rows = parse_qa_pack(PACK)
    template = [json.loads(template_line(r)) for r in rows]
    rulings = {
        row_id(rows[0]): {"ground_truth": "approve", "reason": "ok"},
        row_id(rows[1]): {"ground_truth": "reject", "reason": "casing_error"},
        row_id(rows[2]): {"ground_truth": "approve", "reason": "ok"},
    }
    pairs = merge_rulings(template, rulings)
    assert len(pairs) == 3
    assert pairs[0].ground_truth == "approve"
    assert pairs[1].reason.value == "casing_error"


def test_merge_rulings_reports_unruled_rows():
    rows = parse_qa_pack(PACK)
    template = [json.loads(template_line(r)) for r in rows]
    with pytest.raises(ValueError, match="2 of 3"):
        merge_rulings(template, {row_id(rows[0]): {"ground_truth": "approve", "reason": "ok"}})


def test_merge_rulings_rejects_ruling_for_unknown_id():
    rows = parse_qa_pack(PACK)
    template = [json.loads(template_line(r)) for r in rows]
    rulings = {row_id(r): {"ground_truth": "approve", "reason": "ok"} for r in rows}
    rulings["not-a-real-id"] = {"ground_truth": "approve", "reason": "ok"}
    with pytest.raises(ValueError, match="not-a-real-id"):
        merge_rulings(template, rulings)
