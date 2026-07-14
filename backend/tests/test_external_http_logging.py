import asyncio
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.api import external_http
from app.services import activity_logger, channel_llm, chat_session_service


ROUTE_AGENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
ROUTE_API_KEY = "ext-route-private-api-key"


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalars(self):
        return self

    def all(self):
        return self.value


class FakeSession:
    def __init__(self, values):
        self.values = iter(values)
        self.added = []
        self.commit_count = 0

    async def execute(self, _query):
        return FakeResult(next(self.values))

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commit_count += 1


class FakeSessionFactory:
    def __init__(self, session=None, *, enter_error=None):
        self.session = session
        self.enter_error = enter_error

    def __call__(self):
        return self

    async def __aenter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self.session

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False


def make_route_config(
    *,
    require_hmac: bool = False,
    signing_secret: str | None = None,
    sync_timeout_seconds: int = 120,
):
    return SimpleNamespace(
        agent_id=ROUTE_AGENT_ID,
        encrypt_key=signing_secret,
        extra_config={
            "api_key_hash": external_http._hash_secret(ROUTE_API_KEY),
            "require_hmac": require_hmac,
            "max_payload_bytes": 65536,
            "sync_timeout_seconds": sync_timeout_seconds,
        },
    )


def install_route_processing_boundaries(monkeypatch, config, llm_call, *, history=None):
    validation_agent = SimpleNamespace(webhook_rate_limit=5)
    processing_agent = SimpleNamespace(context_window_size=None)
    platform_user = SimpleNamespace(id=uuid.UUID("00000000-0000-0000-0000-000000000002"))
    channel_session = SimpleNamespace(
        id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
        last_message_at=None,
    )
    session = FakeSession((config, validation_agent, history or [], channel_session))

    async def count_hits(_config):
        return 1

    async def load_agent_and_model(_db, _agent_id):
        return processing_agent, object(), None

    async def resolve_channel_user(**_kwargs):
        return platform_user

    async def find_channel_session(**_kwargs):
        return channel_session

    async def record_activity(*_args, **_kwargs):
        return None

    monkeypatch.setattr(external_http, "async_session", FakeSessionFactory(session))
    monkeypatch.setattr(external_http, "_record_and_count_hits", count_hits)
    monkeypatch.setattr(channel_llm, "load_agent_and_models", load_agent_and_model)
    monkeypatch.setattr(channel_llm, "call_channel_llm", llm_call)
    monkeypatch.setattr(external_http.channel_user_service, "resolve_channel_user", resolve_channel_user)
    monkeypatch.setattr(external_http, "find_or_create_channel_session", find_channel_session)
    monkeypatch.setattr(activity_logger, "log_activity", record_activity)
    return session


@pytest.fixture
async def route_client():
    app = FastAPI()
    app.include_router(external_http.router, prefix="/api")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


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


def test_lifecycle_log_includes_session_id_from_request_state(monkeypatch, log_messages):
    monkeypatch.setattr(external_http.time, "monotonic", lambda: 10.125)
    state = make_state()
    state.session_id = "business-conversation-42"

    external_http._log_external_http_event("INFO", "processing", state)

    message = log_messages[-1]
    assert 'event="processing"' in message
    assert 'session_id="business-conversation-42"' in message


def test_lifecycle_logging_is_best_effort(monkeypatch):
    state = make_state()

    def fail_to_log(*_args, **_kwargs):
        raise RuntimeError("sink unavailable")

    monkeypatch.setattr(external_http.logger, "log", fail_to_log)
    external_http._log_external_http_event("INFO", "received", state)


async def test_rate_limiter_failure_log_redacts_exception_message(monkeypatch, log_messages):
    secret = "redis://admin:private-password@redis.internal:6379/0"

    async def fail_to_get_redis():
        raise RuntimeError(secret)

    monkeypatch.setattr(external_http, "get_redis", fail_to_get_redis)
    external_http._LOCAL_RATE_HITS.clear()

    count = await external_http._record_and_count_hits(make_route_config())

    assert count == 1
    message = log_messages[-1]
    assert "Rate limiter unavailable" in message
    assert "RuntimeError" in message
    assert secret not in message
    assert "private-password" not in message


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
        "error_code": "internal_processing_failed",
        "error": "Agent inference failed",
    }
    assert b"provider-secret-detail" not in response.body


def test_unexpected_failure_logs_safe_event_and_redacted_internal_traceback(log_messages):
    state = make_state()
    prohibited_values = (
        "private-message-content",
        "private-metadata-value",
        "ext-private-api-key",
        "private-signing-secret",
        "sha256=private-signature",
        "private-authorization-header",
        "private-external-user",
        "private-external-conversation",
        "private-provider-response-body",
        "private-database-credential",
    )
    try:
        raise RuntimeError(" | ".join(prohibited_values))
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
    assert "RuntimeError" in output
    assert "test_unexpected_failure_logs_safe_event_and_redacted_internal_traceback" in output
    for prohibited in prohibited_values:
        assert prohibited not in output


async def test_runner_emits_heartbeat_and_stops_after_completion(log_messages):
    state = make_state()
    state.stage = "agent_inference"

    async def complete_later():
        await asyncio.sleep(0.03)
        return {"ok": True, "session_id": "session-1"}

    result = await external_http._run_with_heartbeat(
        complete_later(),
        state=state,
        timeout_seconds=1.0,
        heartbeat_interval=0.01,
    )
    count_after_completion = sum('event="processing"' in message for message in log_messages)
    await asyncio.sleep(0.02)

    assert result["session_id"] == "session-1"
    assert count_after_completion >= 1
    assert sum('event="processing"' in message for message in log_messages) == count_after_completion


async def test_runner_stops_heartbeat_after_operation_failure(log_messages):
    state = make_state()
    state.stage = "agent_inference"

    async def fail_later():
        await asyncio.sleep(0.03)
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await external_http._run_with_heartbeat(
            fail_later(),
            state=state,
            timeout_seconds=1.0,
            heartbeat_interval=0.01,
        )

    count_after_failure = sum('event="processing"' in message for message in log_messages)
    await asyncio.sleep(0.02)

    assert count_after_failure >= 1
    assert sum('event="processing"' in message for message in log_messages) == count_after_failure


async def test_runner_cancels_processing_on_timeout_without_retry():
    state = make_state()
    attempts = 0
    cancelled = asyncio.Event()

    async def never_finishes():
        nonlocal attempts
        attempts += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(TimeoutError):
        await external_http._run_with_heartbeat(
            never_finishes(),
            state=state,
            timeout_seconds=0.01,
            heartbeat_interval=1.0,
        )

    assert cancelled.is_set()
    assert attempts == 1


async def test_runner_propagates_caller_cancellation():
    state = make_state()
    operation_cancelled = asyncio.Event()

    async def never_finishes():
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    runner = asyncio.create_task(
        external_http._run_with_heartbeat(
            never_finishes(),
            state=state,
            timeout_seconds=1.0,
            heartbeat_interval=1.0,
        )
    )
    await asyncio.sleep(0)
    runner.cancel()

    with pytest.raises(asyncio.CancelledError):
        await runner
    assert operation_cancelled.is_set()


async def test_runner_preserves_caller_cancellation_during_heartbeat_cleanup(monkeypatch):
    state = make_state()
    cleanup_started = asyncio.Event()

    async def heartbeat_with_cleanup_window(_state, _interval_seconds):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(external_http, "_emit_processing_heartbeat", heartbeat_with_cleanup_window)
    runner = asyncio.create_task(
        external_http._run_with_heartbeat(
            asyncio.sleep(0, result="completed-result"),
            state=state,
            timeout_seconds=1.0,
            heartbeat_interval=1.0,
        )
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
    runner.cancel()

    with pytest.raises(asyncio.CancelledError):
        await runner


async def test_sync_orchestrator_returns_success_and_logs_completion(monkeypatch, log_messages):
    state = make_state()
    state.session_id = "business-conversation-42"
    message = external_http.ExternalHttpMessageIn(content="secret content")

    async def process(**_kwargs):
        return {"ok": True, "request_id": "req-123", "session_id": "session-1", "reply": "done"}

    monkeypatch.setattr(external_http, "_process_external_http_message", process)
    result = await external_http._run_sync_external_http(
        state=state,
        message=message,
        timeout_seconds=1.0,
        heartbeat_interval=1.0,
    )

    assert result["reply"] == "done"
    output = "\n".join(log_messages)
    assert 'event="completed"' in output
    assert 'session_id="business-conversation-42"' in output
    assert 'session_id="session-1"' not in output
    assert "secret content" not in output


async def test_sync_orchestrator_returns_sanitized_500(monkeypatch, log_messages):
    state = make_state()
    message = external_http.ExternalHttpMessageIn(content="secret content")

    async def fail(**_kwargs):
        try:
            raise RuntimeError("provider-internal-detail")
        except RuntimeError as cause:
            raise external_http.ExternalHttpProcessingError(
                "agent_inference",
                "Agent inference failed",
            ) from cause

    monkeypatch.setattr(external_http, "_process_external_http_message", fail)
    response = await external_http._run_sync_external_http(
        state=state,
        message=message,
        timeout_seconds=1.0,
        heartbeat_interval=1.0,
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "ok": False,
        "request_id": "req-123",
        "error_code": "processing_failed",
        "error": "Agent inference failed",
    }
    assert b"provider-internal-detail" not in response.body
    assert 'event="failed"' in "\n".join(log_messages)


async def test_sync_orchestrator_returns_504_and_cancels(monkeypatch, log_messages):
    state = make_state()
    message = external_http.ExternalHttpMessageIn(content="secret content")
    cancelled = asyncio.Event()

    async def hang(**_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(external_http, "_process_external_http_message", hang)
    response = await external_http._run_sync_external_http(
        state=state,
        message=message,
        timeout_seconds=0.01,
        heartbeat_interval=1.0,
    )

    assert response.status_code == 504
    assert json.loads(response.body)["error"] == "Agent processing timed out"
    assert cancelled.is_set()
    assert 'event="timeout"' in "\n".join(log_messages)


async def test_async_orchestrator_times_out_and_consumes_failure(monkeypatch, log_messages):
    state = make_state(mode="async")
    message = external_http.ExternalHttpMessageIn(content="secret content", mode="async")
    cancelled = asyncio.Event()

    async def hang(**_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(external_http, "_process_external_http_message", hang)
    await external_http._run_async_external_http(
        state=state,
        message=message,
        timeout_seconds=0.01,
        heartbeat_interval=1.0,
    )

    assert cancelled.is_set()
    assert 'event="timeout"' in "\n".join(log_messages)


async def test_background_task_is_owned_then_removed(monkeypatch):
    state = make_state(mode="async")
    message = external_http.ExternalHttpMessageIn(content="secret content", mode="async")
    release = asyncio.Event()

    async def process(**_kwargs):
        await release.wait()
        return {"ok": True, "session_id": "session-1"}

    monkeypatch.setattr(external_http, "_process_external_http_message", process)
    task = external_http._start_external_http_background_task(state=state, message=message)
    assert task in external_http._EXTERNAL_HTTP_BACKGROUND_TASKS

    release.set()
    await task
    await asyncio.sleep(0)
    assert task not in external_http._EXTERNAL_HTTP_BACKGROUND_TASKS


async def test_background_task_logs_escaped_failure_once_and_removes(monkeypatch, log_messages):
    state = make_state(mode="async")
    message = external_http.ExternalHttpMessageIn(content="secret content", mode="async")

    async def malformed_result(**_kwargs):
        return None

    monkeypatch.setattr(external_http, "_process_external_http_message", malformed_result)
    task = external_http._start_external_http_background_task(state=state, message=message)

    with pytest.raises(AttributeError):
        await task
    await asyncio.sleep(0)

    output = "\n".join(log_messages)
    assert task not in external_http._EXTERNAL_HTTP_BACKGROUND_TASKS
    assert output.count('event="failed"') == 1
    assert 'reason="Internal processing failed"' in output
    assert 'error_type="AttributeError"' in output
    assert "Traceback (most recent call last)" in output
    assert "secret content" not in output


async def test_cancelled_background_task_is_removed_without_duplicate_log(monkeypatch, log_messages):
    state = make_state(mode="async")
    message = external_http.ExternalHttpMessageIn(content="secret content", mode="async")
    started = asyncio.Event()

    async def hang(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(external_http, "_process_external_http_message", hang)
    task = external_http._start_external_http_background_task(state=state, message=message)
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    output = "\n".join(log_messages)
    assert task not in external_http._EXTERNAL_HTTP_BACKGROUND_TASKS
    assert output.count('event="failed"') == 1
    assert 'reason="request_cancelled"' in output
    assert "secret content" not in output


async def test_route_rejects_invalid_api_key_without_logging_credentials(
    monkeypatch,
    route_client,
    log_messages,
):
    config = make_route_config()
    session = FakeSession((config,))
    monkeypatch.setattr(external_http, "async_session", FakeSessionFactory(session))

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": "Bearer provided-private-api-key"},
        json={"content": "private-message-content"},
    )

    assert response.status_code == 401
    assert response.json()["error_code"] == "invalid_api_key"
    assert response.json()["error"] == "Invalid external HTTP channel API key"
    assert response.headers["x-request-id"] == response.json()["request_id"]
    output = "\n".join(log_messages)
    assert 'event="rejected"' in output
    assert "provided-private-api-key" not in output
    assert ROUTE_API_KEY not in output
    assert "private-message-content" not in output


async def test_route_rejects_invalid_hmac_without_logging_authentication_secrets(
    monkeypatch,
    route_client,
    log_messages,
):
    signing_secret = "private-route-signing-secret"
    config = make_route_config(require_hmac=True, signing_secret=signing_secret)
    session = FakeSession((config,))
    monkeypatch.setattr(external_http, "async_session", FakeSessionFactory(session))

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={
            "authorization": f"Bearer {ROUTE_API_KEY}",
            "x-timestamp": str(int(external_http.time.time())),
            "x-signature-sha256": "sha256=private-route-signature",
        },
        json={"content": "private-message-content"},
    )

    assert response.status_code == 401
    assert response.json()["error_code"] == "invalid_hmac_signature"
    assert response.json()["error"] == "Invalid HMAC signature"
    output = "\n".join(log_messages)
    assert 'event="rejected"' in output
    for secret in (
        ROUTE_API_KEY,
        signing_secret,
        "private-route-signature",
        "private-message-content",
    ):
        assert secret not in output


async def test_route_serializes_payload_validation_failure(monkeypatch, route_client, log_messages):
    config = make_route_config()
    validation_agent = SimpleNamespace(webhook_rate_limit=5)
    session = FakeSession((config, validation_agent))

    async def count_hits(_config):
        return 1

    monkeypatch.setattr(external_http, "async_session", FakeSessionFactory(session))
    monkeypatch.setattr(external_http, "_record_and_count_hits", count_hits)

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        content=b"{}",
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_request_body"
    assert response.json()["error"] == "Invalid request body"
    assert 'event="rejected"' in "\n".join(log_messages)


@pytest.mark.parametrize(("configured_timeout", "expected_timeout"), ((1, 5), (999, 300)))
async def test_route_clamps_sync_timeout_before_processing(
    monkeypatch,
    route_client,
    configured_timeout,
    expected_timeout,
):
    requested_timeouts = []
    original_wait_for = asyncio.wait_for

    async def capture_wait_for(awaitable, timeout):
        requested_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    async def reply(*_args, **_kwargs):
        return "route-reply"

    config = make_route_config(sync_timeout_seconds=configured_timeout)
    install_route_processing_boundaries(monkeypatch, config, reply)
    monkeypatch.setattr(external_http.asyncio, "wait_for", capture_wait_for)

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "route-message"},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "route-reply"
    assert requested_timeouts == [expected_timeout]


async def test_route_accepts_async_processing_and_serializes_response(monkeypatch, route_client):
    model_started = asyncio.Event()
    release_model = asyncio.Event()

    async def reply(*_args, **_kwargs):
        model_started.set()
        await release_model.wait()
        return "async-route-reply"

    config = make_route_config()
    install_route_processing_boundaries(monkeypatch, config, reply)

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "route-message", "mode": "async"},
    )
    await asyncio.wait_for(model_started.wait(), timeout=1.0)

    assert response.status_code == 202
    assert set(response.json()) == {"ok", "status", "request_id"}
    assert response.json()["ok"] is True
    assert response.json()["status"] == "accepted"
    assert response.headers["x-request-id"] == response.json()["request_id"]
    tasks = tuple(external_http._EXTERNAL_HTTP_BACKGROUND_TASKS)
    assert len(tasks) == 1

    release_model.set()
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    assert not external_http._EXTERNAL_HTTP_BACKGROUND_TASKS


async def test_route_serializes_sanitized_500_and_redacts_provider_failure(
    monkeypatch,
    route_client,
    log_messages,
):
    provider_secret = "private-route-provider-response"

    async def fail(*_args, **_kwargs):
        raise RuntimeError(provider_secret)

    config = make_route_config()
    install_route_processing_boundaries(monkeypatch, config, fail)

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "private-route-message"},
    )

    body = response.json()
    assert response.status_code == 500
    assert set(body) == {"ok", "request_id", "error_code", "error"}
    assert body["ok"] is False
    assert body["error"] == "Agent inference failed"
    assert provider_secret not in response.text
    output = "\n".join(log_messages)
    assert "RuntimeError" in output
    assert provider_secret not in output
    assert "private-route-message" not in output
    assert ROUTE_API_KEY not in output


async def test_route_converts_internal_tool_call_history_before_llm_request(
    monkeypatch,
    route_client,
):
    captured_history = None

    async def reply(*_args, **kwargs):
        nonlocal captured_history
        captured_history = kwargs["history"]
        return "route-reply"

    tool_record = SimpleNamespace(
        id=42,
        role="tool_call",
        content=json.dumps({
            "name": "read_file",
            "args": {"path": "notes.txt"},
            "result": "file contents",
        }),
        thinking=None,
    )
    config = make_route_config()
    install_route_processing_boundaries(
        monkeypatch,
        config,
        reply,
        history=[tool_record],
    )

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "route-message"},
    )

    assert response.status_code == 200
    assert [message["role"] for message in captured_history] == ["assistant", "tool"]
    assert captured_history[0]["tool_calls"][0]["id"] == "call_42"
    assert captured_history[1]["tool_call_id"] == "call_42"


async def test_route_persists_tool_callbacks_and_returns_safe_summary(
    monkeypatch,
    route_client,
):
    saved_tool_calls = []

    async def save_tool_call(**kwargs):
        saved_tool_calls.append(kwargs)

    async def reply(*_args, **kwargs):
        callback = kwargs["on_tool_call"]
        await callback({
            "name": "read_file",
            "call_id": "call-safe-summary",
            "args": {"path": "private.txt"},
            "status": "running",
        })
        await callback({
            "name": "read_file",
            "call_id": "call-safe-summary",
            "args": {"path": "private.txt"},
            "status": "done",
            "result": "Error: " + "private-result" * 100,
            "reasoning_content": "private-reasoning",
        })
        return "route-reply"

    monkeypatch.setattr(chat_session_service, "save_tool_call_log", save_tool_call)
    config = make_route_config()
    install_route_processing_boundaries(monkeypatch, config, reply)

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "route-message"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["tool_calls"] == [{
        "name": "read_file",
        "call_id": "call-safe-summary",
        "status": "done",
        "outcome": "error",
    }]
    assert body["tool_errors"] == ["read_file"]
    assert "private.txt" not in response.text
    assert "private-result" not in response.text
    assert len(saved_tool_calls) == 1
    assert saved_tool_calls[0]["tool_call_id"] == "call-safe-summary"
    assert len(saved_tool_calls[0]["result"]) == 500


async def test_route_returns_502_and_does_not_save_llm_error_as_assistant(
    monkeypatch,
    route_client,
    log_messages,
):
    provider_secret = "private-upstream-provider-response"

    async def reply(*_args, **_kwargs):
        return f"[LLM Error] HTTP 400: {provider_secret}"

    config = make_route_config()
    session = install_route_processing_boundaries(monkeypatch, config, reply)

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "private-route-message"},
    )

    assert response.status_code == 502
    assert response.json()["ok"] is False
    assert response.json()["error_code"] == "upstream_llm_error"
    assert response.json()["error"] == "Upstream model request failed"
    assert [message.role for message in session.added] == ["user"]
    assert provider_secret not in response.text
    assert provider_secret not in "\n".join(log_messages)
    assert 'event="completed"' not in "\n".join(log_messages)


async def test_route_serializes_sanitized_504_and_cancels_model(monkeypatch, route_client, log_messages):
    requested_timeouts = []
    original_wait_for = asyncio.wait_for
    model_cancelled = asyncio.Event()

    async def short_wait_for(awaitable, timeout):
        requested_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout=0.01)

    async def hang(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            model_cancelled.set()

    config = make_route_config()
    install_route_processing_boundaries(monkeypatch, config, hang)
    monkeypatch.setattr(external_http.asyncio, "wait_for", short_wait_for)

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "private-route-message"},
    )

    body = response.json()
    assert response.status_code == 504
    assert set(body) == {"ok", "request_id", "error_code", "error"}
    assert body["ok"] is False
    assert body["error"] == "Agent processing timed out"
    assert requested_timeouts == [120]
    assert model_cancelled.is_set()
    output = "\n".join(log_messages)
    assert 'event="timeout"' in output
    assert "private-route-message" not in output
    assert ROUTE_API_KEY not in output


async def test_route_redacts_configuration_secrets_from_sanitized_500(
    monkeypatch,
    route_client,
    log_messages,
):
    configuration_secrets = (
        "private-config-api-key-hash",
        "private-config-signing-secret",
        "private-config-database-password",
    )
    failure = RuntimeError(" | ".join(configuration_secrets))
    monkeypatch.setattr(
        external_http,
        "async_session",
        FakeSessionFactory(enter_error=failure),
    )

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "private-route-message"},
    )

    body = response.json()
    assert response.status_code == 500
    assert set(body) == {"ok", "request_id", "error_code", "error"}
    assert body["ok"] is False
    assert body["error"] == "Internal processing failed"
    output = "\n".join(log_messages)
    assert "RuntimeError" in output
    for secret in (*configuration_secrets, ROUTE_API_KEY, "private-route-message"):
        assert secret not in response.text
        assert secret not in output


def test_message_validation_normalizes_identifiers_and_rejects_blank_values():
    message = external_http.ExternalHttpMessageIn(
        content="  keep content spacing  ",
        external_user_id="  business-user  ",
        external_user_name="  Business User  ",
        conversation_id="  order-1  ",
    )

    assert message.content == "  keep content spacing  "
    assert message.external_user_id == "business-user"
    assert message.external_user_name == "Business User"
    assert message.conversation_id == "order-1"

    for field, value in (
        ("content", "   "),
        ("external_user_id", "   "),
        ("external_user_name", "   "),
        ("conversation_id", "   "),
    ):
        payload = {"content": "hello", field: value}
        with pytest.raises(ValueError):
            external_http.ExternalHttpMessageIn.model_validate(payload)


def test_message_validation_bounds_database_identifiers_and_metadata():
    with pytest.raises(ValueError):
        external_http.ExternalHttpMessageIn(content="hello", external_user_id="u" * 101)
    with pytest.raises(ValueError):
        external_http.ExternalHttpMessageIn(content="hello", external_user_name="n" * 101)

    nested = "leaf"
    for _ in range(external_http.MAX_METADATA_DEPTH + 1):
        nested = {"child": nested}
    with pytest.raises(ValueError):
        external_http.ExternalHttpMessageIn(content="hello", metadata=nested)


def test_external_conversation_key_is_bounded_and_user_scoped():
    first = external_http._external_conversation_key("user-a", "c" * 255)
    second = external_http._external_conversation_key("user-b", "c" * 255)

    assert first != second
    assert first == external_http._external_conversation_key("user-a", "c" * 255)
    assert len(first) <= 200
    assert "user-a" not in first
    assert "c" * 20 not in first


async def test_hmac_replay_claim_is_atomic_and_rejects_duplicate(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.results = iter((True, None))
            self.keys = []

        async def set(self, key, value, *, ex, nx):
            self.keys.append((key, value, ex, nx))
            return next(self.results)

    redis = FakeRedis()

    async def get_fake_redis():
        return redis

    monkeypatch.setattr(external_http, "get_redis", get_fake_redis)
    config = make_route_config(require_hmac=True, signing_secret="signing-secret")
    request = SimpleNamespace(headers={"x-signature-sha256": f"sha256={'a' * 64}"})

    await external_http._claim_hmac_signature(config, request)
    with pytest.raises(HTTPException) as exc_info:
        await external_http._claim_hmac_signature(config, request)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Replayed HMAC request"
    assert redis.keys[0][2:] == (external_http.HMAC_TIMESTAMP_WINDOW_SECONDS, True)


async def test_route_rejects_body_before_buffering_beyond_configured_limit(
    monkeypatch,
    route_client,
):
    config = make_route_config()
    config.extra_config["max_payload_bytes"] = 1024
    monkeypatch.setattr(
        external_http,
        "async_session",
        FakeSessionFactory(FakeSession((config,))),
    )

    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        content=b"x" * 1025,
    )

    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"


async def test_route_returns_retry_headers_for_rate_and_capacity_limits(
    monkeypatch,
    route_client,
):
    config = make_route_config()
    validation_agent = SimpleNamespace(webhook_rate_limit=5)

    async def over_limit(_config):
        return 6

    monkeypatch.setattr(
        external_http,
        "async_session",
        FakeSessionFactory(FakeSession((config, validation_agent))),
    )
    monkeypatch.setattr(external_http, "_record_and_count_hits", over_limit)
    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "hello"},
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.json()["error_code"] == "rate_limit_exceeded"

    async def no_capacity(_agent_id):
        return None

    async def within_limit(_config):
        return 1

    monkeypatch.setattr(
        external_http,
        "async_session",
        FakeSessionFactory(FakeSession((config, validation_agent))),
    )
    monkeypatch.setattr(external_http, "_record_and_count_hits", within_limit)
    monkeypatch.setattr(external_http, "_try_acquire_processing_lease", no_capacity)
    response = await route_client.post(
        f"/api/channel/external-http/{ROUTE_AGENT_ID}/message",
        headers={"authorization": f"Bearer {ROUTE_API_KEY}"},
        json={"content": "hello"},
    )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["error_code"] == "processing_capacity_exhausted"
