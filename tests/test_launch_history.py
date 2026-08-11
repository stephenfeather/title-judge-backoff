from run_bakeoff import accumulate_launches

L1 = {"calls_ok": 550, "calls_failed": 50, "latency_median": 12.0,
      "error_kinds": {"ReadTimeout": 40, "JudgeResponseError": 10}}
L2 = {"calls_ok": 47, "calls_failed": 0, "latency_median": 1.5, "error_kinds": {}}


def test_a_first_launch_records_itself():
    launches, cumulative = accumulate_launches(None, L1)
    assert launches == [L1]
    assert cumulative["launches"] == 1
    assert cumulative["calls_ok"] == 550
    assert cumulative["calls_failed"] == 50


def test_a_resume_appends_rather_than_replacing():
    # The whole defect: the second launch overwrote the first, so a backend that
    # lost 50 calls reported a flawless run once a 47-call resume finished.
    launches, cumulative = accumulate_launches({"launches": [L1]}, L2)
    assert launches == [L1, L2]
    assert cumulative["launches"] == 2
    assert cumulative["calls_ok"] == 597
    assert cumulative["calls_failed"] == 50


def test_error_kinds_are_summed_across_launches():
    first = {"calls_ok": 1, "calls_failed": 2, "error_kinds": {"ReadTimeout": 2}}
    second = {"calls_ok": 1, "calls_failed": 3,
              "error_kinds": {"ReadTimeout": 1, "JudgeResponseError": 2}}
    _, cumulative = accumulate_launches({"launches": [first]}, second)
    assert cumulative["error_kinds"] == {"ReadTimeout": 3, "JudgeResponseError": 2}


def test_error_kinds_are_ordered_so_two_runs_diff_cleanly():
    first = {"calls_ok": 0, "calls_failed": 1, "error_kinds": {"Zebra": 1}}
    second = {"calls_ok": 0, "calls_failed": 1, "error_kinds": {"Alpha": 1}}
    _, cumulative = accumulate_launches({"launches": [first]}, second)
    assert list(cumulative["error_kinds"]) == ["Alpha", "Zebra"]


def test_cumulative_carries_no_latency():
    # Medians cannot be merged — combining them needs the raw samples, and a
    # median-of-medians is simply a wrong number. Latency stays per launch,
    # where it means something: 1.5s over a 47-call resume is not comparable
    # to 12s over a 600-call run.
    _, cumulative = accumulate_launches({"launches": [L1]}, L2)
    assert not any("latency" in key for key in cumulative)
    assert cumulative["launches"] == 2


def test_each_launch_keeps_its_own_latency():
    launches, _ = accumulate_launches({"launches": [L1]}, L2)
    assert [x["latency_median"] for x in launches] == [12.0, 1.5]


def test_a_manifest_written_before_launch_history_existed_is_adopted():
    # Every manifest in results/ predates this. Its `health` block is the only
    # record of that launch, and dropping it would destroy the very history
    # this exists to keep.
    legacy = {"backend": "nv", "health": L1}
    launches, cumulative = accumulate_launches(legacy, L2)
    assert launches == [L1, L2]
    assert cumulative["calls_failed"] == 50


def test_an_unreadable_previous_manifest_starts_a_fresh_history():
    # Corrupt or absent, the run must still record what it just did rather
    # than refusing to write anything.
    launches, cumulative = accumulate_launches(None, L2)
    assert launches == [L2]
    assert cumulative["launches"] == 1


IDENTITY = {"model_id": "m", "base_url": "https://a.test/v1", "prompt_version": "v2",
            "temperature": None, "reasoning_effort": None,
            "config_digest": "dig1", "code_version": "aaa"}


def test_history_from_a_different_config_is_not_adopted():
    # PR #47 review: an all-failure launch writes a manifest but NO verdict
    # rows, so the row-based resume guard has nothing to compare and cannot
    # fire. Adopting that manifest after the model or host changed would file
    # the old provider's failures under the new one.
    previous = {**IDENTITY, "model_id": "OTHER", "launches": [L1]}
    launches, cumulative = accumulate_launches(previous, L2, identity=IDENTITY)
    assert launches == [L2], "the other model's launches must not be inherited"
    assert cumulative["calls_failed"] == 0


def test_history_from_the_same_config_is_adopted():
    previous = {**IDENTITY, "launches": [L1]}
    launches, cumulative = accumulate_launches(previous, L2, identity=IDENTITY)
    assert launches == [L1, L2]
    assert cumulative["calls_failed"] == 50


def test_discarding_a_history_is_recorded_rather_than_silent():
    # Dropping the prior launches is correct — they describe a different
    # configuration — but it is still the loss of a failure record, which is
    # the thing #40 exists to prevent. Say so.
    previous = {**IDENTITY, "code_version": "bbb", "launches": [L1, L2]}
    _, cumulative = accumulate_launches(previous, L2, identity=IDENTITY)
    assert cumulative["discarded_prior_launches"] == 2


def test_no_identity_given_adopts_the_history_unchanged():
    # Callers that do not track identity keep the previous behaviour, the same
    # concession already made for provenance in already_judged_ids.
    previous = {"launches": [L1]}
    launches, _ = accumulate_launches(previous, L2)
    assert launches == [L1, L2]


def test_a_legacy_manifest_with_no_identity_is_adopted():
    # Manifests predating provenance carry no config_digest or code_version.
    # Absent is unknown, not a mismatch — the same rule the resume guard uses.
    previous = {"model_id": "m", "base_url": "https://a.test/v1", "launches": [L1]}
    launches, _ = accumulate_launches(previous, L2, identity=IDENTITY)
    assert launches == [L1, L2]


def test_a_legacy_manifest_with_no_health_contributes_nothing():
    launches, cumulative = accumulate_launches({"backend": "nv"}, L2)
    assert launches == [L2]
    assert cumulative["calls_ok"] == 47
