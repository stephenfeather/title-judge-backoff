from judge.client import Backend
from judge.check import BackendCheck, check_key_presence, render_check_report


def make_backend(name, api_key_env, role="contender", eval_only=False):
    return Backend(
        name=name,
        base_url="https://example.test/v1",
        model_id=f"model-{name}",
        rpm=40,
        eval_only=eval_only,
        api_key_env=api_key_env,
        role=role,
    )


def test_check_key_presence_marks_ready_when_key_set(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "x")
    check = check_key_presence(make_backend("has-key", "PRESENT_KEY"))
    assert check == BackendCheck(
        name="has-key", status="ready", detail="PRESENT_KEY is set", reachable=None
    )


def test_check_key_presence_marks_skipped_when_key_missing(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    check = check_key_presence(make_backend("no-key", "ABSENT_KEY"))
    assert check.status == "skipped"
    assert "ABSENT_KEY" in check.detail


def test_check_key_presence_treats_blank_key_as_missing(monkeypatch):
    monkeypatch.setenv("BLANK_KEY", "   ")
    assert check_key_presence(make_backend("blank", "BLANK_KEY")).status == "skipped"


def test_render_check_report_lists_both_groups():
    report = render_check_report(
        [
            BackendCheck(name="ready-one", status="ready", detail="K is set", reachable=None),
            BackendCheck(name="skip-one", status="skipped", detail="K2 is not set", reachable=None),
        ]
    )
    assert "ready-one" in report
    assert "skip-one" in report
    assert "1 ready" in report
    assert "1 skipped" in report
