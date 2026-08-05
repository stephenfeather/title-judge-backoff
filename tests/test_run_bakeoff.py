from judge.schema import Pair, ReasonCode, Verdict, verdict_to_json_line
from run_bakeoff import already_judged_ids, pending_pairs


def make_pair(pair_id):
    return Pair(
        id=pair_id,
        original=f"orig {pair_id}",
        enriched=f"enriched {pair_id}",
        brand="Acme",
        mpn=f"MPN-{pair_id}",
        ground_truth="approve",
        reason=ReasonCode.OK,
    )


def make_verdict(pair_id):
    return Verdict(
        pair_id=pair_id,
        verdict="approve",
        reason=ReasonCode.OK,
        model_id="m",
        prompt_version="v1",
        temperature=0.0,
    )


def test_already_judged_ids_reads_existing_results(tmp_path):
    out = tmp_path / "backend.jsonl"
    out.write_text(
        verdict_to_json_line(make_verdict("p1")) + "\n" + verdict_to_json_line(make_verdict("p3")) + "\n"
    )
    assert already_judged_ids(out) == {"p1", "p3"}


def test_already_judged_ids_missing_file_is_empty(tmp_path):
    assert already_judged_ids(tmp_path / "nope.jsonl") == set()


def test_pending_pairs_skips_judged_and_preserves_order():
    pairs = [make_pair("p1"), make_pair("p2"), make_pair("p3")]
    assert [p.id for p in pending_pairs(pairs, {"p2"})] == ["p1", "p3"]
