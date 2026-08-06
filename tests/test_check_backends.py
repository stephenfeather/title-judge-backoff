import httpx

from judge.client import Backend, JudgeClient
from judge.check import BackendCheck, check_key_presence, ping_backend, render_check_report


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


def test_failed_ping_reports_the_response_body(monkeypatch):
    # A bare "400 Bad Request" read as a bad API key for long enough to matter:
    # the body said the model rejected temperature=0. The provider's explanation
    # is the whole diagnostic value of a ping, so it must survive into the report.
    monkeypatch.setenv("PING_KEY", "x")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unsupported value: 'temperature' does not support 0 with this model.",
                    "code": "unsupported_value",
                }
            },
        )

    check = ping_backend(
        make_backend("bad-request", "PING_KEY"), transport=httpx.MockTransport(handler)
    )
    assert check.reachable is False
    assert "temperature" in check.detail
    assert "unsupported_value" in check.detail


def test_ping_closes_its_client_on_success_and_failure(monkeypatch):
    # Pinging a 10-backend slate opened 10 clients and closed none. Each holds a
    # connection pool, so a --ping across a large slate leaked file descriptors.
    monkeypatch.setenv("PING_KEY", "x")
    closed = []

    class TrackingClient(JudgeClient):
        def close(self):
            closed.append(self.backend.name)
            super().close()

    monkeypatch.setattr("judge.check.JudgeClient", TrackingClient)

    ping_backend(
        make_backend("ok", "PING_KEY"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"verdict":"approve","reason":"ok"}'}}]},
            )
        ),
    )
    ping_backend(
        make_backend("boom", "PING_KEY"),
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="nope")),
    )
    assert closed == ["ok", "boom"]


def test_successful_ping_is_marked_reachable(monkeypatch):
    monkeypatch.setenv("PING_KEY", "x")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"verdict": "approve", "reason": "ok"}'}}]},
        )

    check = ping_backend(make_backend("ok", "PING_KEY"), transport=httpx.MockTransport(handler))
    assert check.reachable is True


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
