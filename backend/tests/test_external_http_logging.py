import asyncio
import json
import uuid

import pytest
from fastapi import HTTPException
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


async def test_processing_stage_wraps_unexpected_error_with_public_reason():
    state = make_state()

    with pytest.raises(external_http.ExternalHttpProcessingError) as exc_info:
        async with external_http._processing_stage(
            state,
            "agent_inference",
            "Agent inference failed",
        ):
            raise RuntimeError("provider-secret-detail")

    assert state.stage == "agent_inference"
    assert exc_info.value.stage == "agent_inference"
    assert exc_info.value.public_reason == "Agent inference failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_processing_stage_preserves_expected_http_exception():
    state = make_state()

    with pytest.raises(HTTPException) as exc_info:
        async with external_http._processing_stage(
            state,
            "prepare_session",
            "Failed to prepare agent session",
        ):
            raise HTTPException(status_code=404, detail="Agent not found")

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent not found"


async def test_processing_stage_preserves_cancellation():
    state = make_state()

    with pytest.raises(asyncio.CancelledError):
        async with external_http._processing_stage(
            state,
            "agent_inference",
            "Agent inference failed",
        ):
            raise asyncio.CancelledError


async def test_processor_without_state_creates_fallback_for_stage_classification(monkeypatch):
    agent_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    message = external_http.ExternalHttpMessageIn(content="hello", mode="async")
    state_type = external_http.ExternalHttpRequestState
    created_states: list[external_http.ExternalHttpRequestState] = []

    def capture_state(**kwargs):
        state = state_type(**kwargs)
        created_states.append(state)
        return state

    class FailingSession:
        async def __aenter__(self):
            raise RuntimeError("database unavailable")

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(external_http, "ExternalHttpRequestState", capture_state)
    monkeypatch.setattr(external_http, "async_session", FailingSession)

    with pytest.raises(external_http.ExternalHttpProcessingError) as exc_info:
        await external_http._process_external_http_message(
            agent_id=agent_id,
            message=message,
            request_id="req-fallback",
        )

    assert len(created_states) == 1
    assert created_states[0].request_id == "req-fallback"
    assert created_states[0].agent_id == agent_id
    assert created_states[0].mode == "async"
    assert created_states[0].stage == "prepare_session"
    assert exc_info.value.stage == "prepare_session"
    assert exc_info.value.public_reason == "Failed to prepare agent session"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_public_error_response_contains_only_safe_reason():
    state = make_state()
    response = external_http._public_error_response(
        state,
        "Agent inference failed",
        status_code=500,
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "ok": False,
        "request_id": "req-123",
        "error": "Agent inference failed",
    }
    assert b"provider-secret-detail" not in response.body


def test_unexpected_failure_logs_safe_event_and_internal_traceback(log_messages):
    state = make_state()
    try:
        raise RuntimeError("provider failed")
    except RuntimeError as cause:
        exc = external_http.ExternalHttpProcessingError(
            "agent_inference",
            "Agent inference failed",
        )
        exc.__cause__ = cause

    external_http._log_unexpected_failure(state, exc)

    output = "\n".join(log_messages)
    assert 'event="failed"' in output
    assert 'reason="Agent inference failed"' in output
    assert 'error_type="RuntimeError"' in output
    assert "Traceback (most recent call last)" in output
