import json

import pytest

from judge.provenance import DirtyTree, GitUnavailable, audit_results_file, code_version


def write_rows(path, *rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def row(**provenance):
    base = {"pair_id": "p1", "verdict": "approve", "reason": "ok", "model_id": "m"}
    base.update(provenance)
    return base


def test_audit_reports_a_single_clean_version(tmp_path):
    path = tmp_path / "backend.jsonl"
    write_rows(path, row(code_version="aaa"), row(code_version="aaa"))
    audit = audit_results_file(path)
    assert audit.code_versions == ("aaa",)
    assert not audit.is_mixed


def test_audit_detects_two_code_versions(tmp_path):
    # The #10 scenario, found after the fact instead of by inspection.
    path = tmp_path / "backend.jsonl"
    write_rows(path, row(code_version="aaa"), row(code_version="bbb"))
    audit = audit_results_file(path)
    assert audit.code_versions == ("aaa", "bbb")
    assert audit.is_mixed


def test_audit_detects_a_legacy_and_new_mixture(tmp_path):
    # Per the review: unknown is its own bucket. A file where half the rows
    # predate provenance is mixed, not "one known version".
    path = tmp_path / "backend.jsonl"
    write_rows(path, row(), row(code_version="aaa"))
    audit = audit_results_file(path)
    assert audit.rows_without_provenance == 1
    assert audit.is_mixed


def test_audit_of_an_all_legacy_file_is_not_mixed(tmp_path):
    # Uniformly unknown is unauditable, not evidence of mixing.
    path = tmp_path / "backend.jsonl"
    write_rows(path, row(), row())
    audit = audit_results_file(path)
    assert audit.rows_without_provenance == 2
    assert not audit.is_mixed


def test_audit_counts_malformed_rows(tmp_path):
    # PR #28 review: a truncated or corrupt row was counted in total_rows and
    # then dropped, so a damaged file with otherwise uniform provenance reported
    # is_mixed=False — telling a consumer an unauditable file is coherent.
    path = tmp_path / "backend.jsonl"
    path.write_text(json.dumps(row(code_version="aaa")) + "\n{not valid json\n")
    audit = audit_results_file(path)
    assert audit.malformed_rows == 1
    assert audit.total_rows == 2


def test_audit_of_a_damaged_file_is_never_reported_coherent(tmp_path):
    path = tmp_path / "backend.jsonl"
    path.write_text(json.dumps(row(code_version="aaa")) + "\n{truncated\n")
    assert audit_results_file(path).is_mixed


def test_audit_reports_distinct_hosts_and_digests(tmp_path):
    path = tmp_path / "backend.jsonl"
    write_rows(
        path,
        row(code_version="aaa", base_url="https://host-a.test/v1", config_digest="d1"),
        row(code_version="aaa", base_url="https://host-b.test/v1", config_digest="d2"),
    )
    audit = audit_results_file(path)
    assert audit.base_urls == ("https://host-a.test/v1", "https://host-b.test/v1")
    assert audit.config_digests == ("d1", "d2")
    assert audit.is_mixed

SHA = "9b0d01a1c2d3"


def test_clean_tree_is_just_the_sha():
    assert code_version(head=SHA, tree_diff="", allow_dirty=False) == SHA


def test_dirty_tree_refuses_by_default():
    # A paid calibration sweep must not run from an uncommitted tree.
    with pytest.raises(DirtyTree, match="--allow-dirty"):
        code_version(head=SHA, tree_diff="diff --git a/judge/client.py", allow_dirty=False)


def test_dirty_tree_is_labelled_when_explicitly_allowed():
    version = code_version(head=SHA, tree_diff="diff --git a/judge/client.py", allow_dirty=True)
    assert version.startswith(f"{SHA}-dirty-")


def test_different_dirty_trees_at_one_commit_get_different_versions():
    # The rev 1 hole: a bare `-dirty` suffix collapsed every uncommitted tree at
    # a commit into one value, so edit-resume-edit appended incompatible rows.
    one = code_version(head=SHA, tree_diff="edit one", allow_dirty=True)
    two = code_version(head=SHA, tree_diff="edit two", allow_dirty=True)
    assert one != two


def test_the_same_dirty_tree_is_reproducible():
    one = code_version(head=SHA, tree_diff="edit one", allow_dirty=True)
    two = code_version(head=SHA, tree_diff="edit one", allow_dirty=True)
    assert one == two


def test_unresolvable_git_state_refuses():
    # Writing a row whose provenance is unknowable is the thing being fixed, so
    # there is no fallback to None on the write path.
    with pytest.raises(GitUnavailable):
        code_version(head=None, tree_diff="", allow_dirty=False)


def test_unresolvable_git_state_refuses_even_with_allow_dirty():
    # --allow-dirty relaxes "uncommitted", never "unidentifiable".
    with pytest.raises(GitUnavailable):
        code_version(head=None, tree_diff="whatever", allow_dirty=True)
