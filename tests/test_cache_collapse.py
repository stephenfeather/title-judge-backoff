from judge.cache_collapse import (
    COLLAPSE_SUSPECTED,
    OK,
    UNVERIFIABLE,
    cache_findings,
    render_cache_warning,
)
from judge.schema import ReasonCode, Usage, Verdict


def usage(cached=None, measured=True):
    """A per-row usage block. cached=None means the host reported no cache field."""
    if not measured:
        return None
    return Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110, cached_tokens=cached)


def votes(pair_id, n, *, verdicts=None, reasons=None, cached=None, measured=True):
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
                usage=usage(cached=cached, measured=measured),
            )
        )
    return out


def test_cache_hits_with_a_flat_flip_rate_is_a_suspected_collapse():
    findings = cache_findings({"nv": votes("p1", 3, cached=90)})
    assert [f.status for f in findings] == [COLLAPSE_SUSPECTED]


def test_cache_hits_without_a_flat_flip_rate_are_not_a_collapse():
    # Prompt-prefix caching reuses input tokens, not the response. Variance
    # proves the responses were not reused.
    flipped = votes("p1", 3, verdicts=["approve", "reject", "approve"], cached=90)
    assert [f.status for f in cache_findings({"nv": flipped})] == [OK]


def test_reason_flips_alone_keep_a_backend_out_of_suspicion():
    # The floor backend: 0 verdict flips, 17.5% reason flips. Responses that
    # differ at all cannot have been one cached response.
    reason_flipped = votes(
        "p1", 3, reasons=[ReasonCode.OK, ReasonCode.MEANING_CHANGE, ReasonCode.OK], cached=90
    )
    assert [f.status for f in cache_findings({"nv": reason_flipped})] == [OK]


def test_measured_zero_cache_hits_with_a_flat_rate_is_fine():
    assert [f.status for f in cache_findings({"nv": votes("p1", 3, cached=0)})] == [OK]


def test_rows_with_no_usage_at_all_are_unverifiable():
    assert [f.status for f in cache_findings({"nv": votes("p1", 3, measured=False)})] == [
        UNVERIFIABLE
    ]


def test_a_host_reporting_tokens_but_no_cache_field_is_unverifiable():
    # Codex P1: `cached_tokens` absent is NOT `cached_tokens == 0`. A host that
    # reports ordinary totals while omitting the cache field has measured
    # nothing about caching, and an absent measurement must not become a
    # measured zero.
    assert [f.status for f in cache_findings({"nv": votes("p1", 3, cached=None)})] == [UNVERIFIABLE]


def test_a_single_vote_run_is_never_flagged():
    # flip_rate over one value is 0.0 by construction.
    assert [f.status for f in cache_findings({"nv": votes("p1", 1, cached=90)})] == [OK]


def test_flatness_ignores_pairs_that_never_got_a_second_vote():
    # Claude finding 1: a single-vote pair contributes 0.0 by construction, so
    # averaging over all pairs lets 195 lost-to-errors pairs dilute the mean
    # toward zero and manufacture a collapse out of 5 repeated ones.
    repeated_and_flipping = votes("p1", 3, verdicts=["approve", "reject", "approve"], cached=90)
    singles = [v for i in range(2, 40) for v in votes(f"p{i}", 1, cached=90)]
    findings = cache_findings({"nv": repeated_and_flipping + singles})
    assert findings[0].status == OK
    # The evidence base is the repeated pairs, not the pair count.
    assert findings[0].repeated_pairs == 1
    assert findings[0].verdict_flip_rate > 0


def test_the_evidence_base_is_reported_not_just_the_verdict():
    # Cache counts span every call, including the single-vote pair; only the
    # FLATNESS evidence is restricted to repeated pairs.
    findings = cache_findings({"nv": votes("p1", 3, cached=90) + votes("p2", 1, cached=90)})
    assert findings[0].repeated_pairs == 1
    assert findings[0].total_calls == 4
    assert findings[0].calls_with_cache_hit == 4
    assert findings[0].calls_measuring_cache == 4


def test_findings_cover_every_backend_and_are_ordered():
    by_model = {"b": votes("p1", 3, cached=0), "a": votes("p1", 3, cached=90)}
    findings = cache_findings(by_model)
    assert [f.backend for f in findings] == ["a", "b"]
    assert [f.status for f in findings] == [COLLAPSE_SUSPECTED, OK]


def test_unverifiable_rows_never_render_a_zero_that_reads_as_measured():
    # "not checkable" beside "0" reads as "everything was measured".
    rendered = "\n".join(render_cache_warning(cache_findings({"nv": votes("p1", 3, cached=None)})))
    assert "not checkable" in rendered.lower()
    assert "| 0 |" not in rendered


def test_a_clean_run_says_it_was_checked_rather_than_staying_silent():
    # Claude finding 5: silence is indistinguishable from the detector not
    # existing — the same absent-vs-measured distinction, one level up.
    rendered = "\n".join(render_cache_warning(cache_findings({"nv": votes("p1", 3, cached=0)})))
    assert "checked" in rendered.lower()


def test_a_run_with_nothing_to_check_is_not_claimed_as_checked():
    # Single-vote runs are not evidence of anything, so they must not be
    # reported as "checked and clear" either.
    rendered = "\n".join(render_cache_warning(cache_findings({"nv": votes("p1", 1, cached=0)})))
    assert rendered == ""
