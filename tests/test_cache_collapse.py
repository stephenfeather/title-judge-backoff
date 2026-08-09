from judge.cache_collapse import (
    COLLAPSE_SUSPECTED,
    OK,
    UNVERIFIABLE,
    cache_findings,
    render_cache_warning,
)
from judge.schema import ReasonCode, Verdict


def votes(pair_id, n, *, verdicts=None, reasons=None):
    """n votes on one pair; by default identical (a flat, unflipped pair)."""
    out = []
    for i in range(n):
        out.append(
            Verdict(
                pair_id=pair_id,
                verdict=(verdicts[i] if verdicts else "approve"),
                reason=(reasons[i] if reasons else ReasonCode.OK),
                model_id="m",
                prompt_version="v2",
                temperature=None,
                run_index=i,
            )
        )
    return out


def manifest(*, cache_hits=0, unmeasured=0, measured=3):
    return {
        "usage": {
            "calls_measured": measured,
            "calls_unmeasured": unmeasured,
            "calls_with_cache_hit": cache_hits,
        }
    }


def test_cache_hits_with_a_flat_flip_rate_is_a_suspected_collapse():
    # The failure this exists to catch: three identical POSTs served one cached
    # response, so majority-of-3 silently became n=1 and every metric improved.
    findings = cache_findings({"nv": votes("p1", 3)}, {"nv": manifest(cache_hits=3)})
    assert [f.status for f in findings] == [COLLAPSE_SUSPECTED]


def test_cache_hits_without_a_flat_flip_rate_are_not_a_collapse():
    # Prompt-prefix caching is normal and harmless: it reuses input tokens, not
    # the response. Variance proves the responses were not reused.
    flipped = votes("p1", 3, verdicts=["approve", "reject", "approve"])
    findings = cache_findings({"nv": flipped}, {"nv": manifest(cache_hits=3)})
    assert [f.status for f in findings] == [OK]


def test_a_flat_flip_rate_with_no_usage_data_is_unverifiable():
    # "no cache hits" and "no data about cache hits" are different answers, and
    # the second must not read as the first.
    findings = cache_findings({"nv": votes("p1", 3)}, {"nv": {}})
    assert [f.status for f in findings] == [UNVERIFIABLE]


def test_a_flat_flip_rate_with_every_call_unmeasured_is_unverifiable():
    findings = cache_findings(
        {"nv": votes("p1", 3)}, {"nv": manifest(cache_hits=0, measured=0, unmeasured=3)}
    )
    assert [f.status for f in findings] == [UNVERIFIABLE]


def test_a_flat_flip_rate_with_measured_zero_cache_hits_is_fine():
    # Genuinely stable AND measured: the floor backend's real result.
    findings = cache_findings({"nv": votes("p1", 3)}, {"nv": manifest(cache_hits=0, measured=3)})
    assert [f.status for f in findings] == [OK]


def test_a_single_vote_run_is_never_flagged():
    # flip_rate over one value is 0.0 by construction, so a --votes 1 run would
    # otherwise report a collapse on every backend.
    findings = cache_findings({"nv": votes("p1", 1)}, {"nv": manifest(cache_hits=1)})
    assert [f.status for f in findings] == [OK]


def test_reason_flips_alone_keep_a_backend_out_of_suspicion():
    # The floor backend flipped reasons 17.5% with zero verdict flips. Responses
    # that differ at all cannot have been one cached response.
    reason_flipped = votes(
        "p1", 3, reasons=[ReasonCode.OK, ReasonCode.MEANING_CHANGE, ReasonCode.OK]
    )
    findings = cache_findings({"nv": reason_flipped}, {"nv": manifest(cache_hits=3)})
    assert [f.status for f in findings] == [OK]


def test_a_backend_with_no_usage_block_does_not_render_zero_unmeasured():
    # "not checkable" beside "unmeasured: 0" is self-contradictory — 0 reads as
    # "everything was measured". No usage block at all must render as unknown.
    findings = cache_findings({"nv": votes("p1", 3)}, {"nv": {}})
    rendered = "\n".join(render_cache_warning(findings))
    assert "| 0 |" not in rendered
    assert "| — |" in rendered


def test_findings_cover_every_backend_and_are_ordered():
    by_model = {"b": votes("p1", 3), "a": votes("p1", 3)}
    manifests = {"a": manifest(cache_hits=3), "b": manifest(cache_hits=0, measured=3)}
    findings = cache_findings(by_model, manifests)
    assert [f.backend for f in findings] == ["a", "b"]
    assert [f.status for f in findings] == [COLLAPSE_SUSPECTED, OK]
