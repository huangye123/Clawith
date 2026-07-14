import uuid

import pytest
from loguru import logger

from app.api import external_http


@pytest.fixture
def log_messages():
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        format="{message}",
        level="INFO",
        enqueue=False,
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def make_state(*, mode: str | None = "sync") -> external_http.ExternalHttpRequestState:
    return external_http.ExternalHttpRequestState(
        request_id="req-123",
        agent_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        mode=mode,
        started_at=10.0,
    )


def test_lifecycle_log_is_searchable_and_filters_sensitive_fields(monkeypatch, log_messages):
    monkeypatch.setattr(external_http.time, "monotonic", lambda: 10.125)
    state = make_state()

    external_http._log_external_http_event(
        "INFO",
        "validated",
        state,
        status_code=200,
        payload_bytes=321,
        content="private-message",
        metadata={"secret": "metadata-secret"},
        api_key="ext-secret-key",
        signature="sha256=secret-signature",
        external_user_id="private-user",
        conversation_id="private-conversation",
    )

    message = log_messages[-1]
    assert "[ExternalHTTP]" in message
    assert 'event="validated"' in message
    assert 'request_id="req-123"' in message
    assert 'agent_id="00000000-0000-0000-0000-000000000001"' in message
    assert 'mode="sync"' in message
    assert "status_code=200" in message
    assert "duration_ms=125" in message
    assert "payload_bytes=321" in message
    for secret in (
        "private-message",
        "metadata-secret",
        "ext-secret-key",
        "secret-signature",
        "private-user",
        "private-conversation",
    ):
        assert secret not in message


def test_lifecycle_logging_is_best_effort(monkeypatch):
    state = make_state()

    def fail_to_log(*_args, **_kwargs):
        raise RuntimeError("sink unavailable")

    monkeypatch.setattr(external_http.logger, "log", fail_to_log)
    external_http._log_external_http_event("INFO", "received", state)
