# External HTTP Docker Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe Docker-only lifecycle logs, sanitized public failures, and bounded/cancellable long-running model inference to the External HTTP message endpoint.

**Architecture:** Keep all production behavior in `backend/app/api/external_http.py`. Introduce a request-state value object, safe key/value logging, stage-aware exception wrapping, a heartbeat/timeout runner, and explicit synchronous/asynchronous orchestration; cover each boundary with focused async unit tests before integrating it into the endpoint.

**Tech Stack:** Python 3.11+, FastAPI, asyncio, Loguru, pytest, pytest-asyncio, Ruff.

## Global Constraints

- New request lifecycle logs must use the existing Loguru stdout sink and must not be persisted in a new table or exposed through an API/UI.
- Never log message content, metadata, API keys, HMAC signatures, headers, raw request bodies, external user IDs, or external conversation IDs.
- Expected HTTP errors keep their existing status code and response detail.
- Unexpected public errors contain only `request_id` and a short stage-based reason; raw exception strings and tracebacks stay out of HTTP responses.
- Synchronous timeout remains configurable from 5 through 300 seconds with the existing 120-second default.
- Asynchronous processing has a fixed 300-second hard timeout.
- Long-running processing emits a heartbeat every 30 seconds.
- Timeout cancels and awaits processing; it never continues in the background and never retries model inference.
- Preserve existing chat messages, sessions, and activity records.
- Do not add database migrations, frontend changes, dependencies, or unrelated refactors.

---

## File Map

- Modify `backend/app/api/external_http.py`: request state, safe logs, public error mapping, stage boundaries, heartbeat/timeout execution, task ownership, and endpoint integration.
- Create `backend/tests/test_external_http_logging.py`: focused deterministic tests for the new behavior.
- Modify `deploy/docs/external-http-channel.md`: document Docker events, sanitized errors, heartbeat, and sync/async timeout behavior.

### Task 1: Safe lifecycle state and Docker log formatting

**Files:**
- Modify: `backend/app/api/external_http.py:5-53`
- Create: `backend/tests/test_external_http_logging.py`

**Interfaces:**
- Produces: `ExternalHttpRequestState`, `_log_external_http_event(level, event, state, **fields)`, `ASYNC_PROCESSING_TIMEOUT_SECONDS`, and `PROCESSING_HEARTBEAT_SECONDS`.
- Consumes: existing module-level Loguru `logger` and `time.monotonic()`.

- [ ] **Step 1: Write failing tests for allowlisted, searchable lifecycle logs**

Create `backend/tests/test_external_http_logging.py` with:

```python
import asyncio
import json
import uuid
from types import SimpleNamespace

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
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q
```

Expected: collection fails because `ExternalHttpRequestState` and `_log_external_http_event` do not exist.

- [ ] **Step 3: Implement the minimal safe lifecycle primitives**

Add imports near the top of `backend/app/api/external_http.py`:

```python
from dataclasses import dataclass, field
```

Add immediately after the existing timeout constants:

```python
PROCESSING_HEARTBEAT_SECONDS = 30.0
ASYNC_PROCESSING_TIMEOUT_SECONDS = 300.0

_LOG_FIELD_ORDER = (
    "event",
    "request_id",
    "agent_id",
    "mode",
    "stage",
    "status_code",
    "duration_ms",
    "payload_bytes",
    "session_id",
    "error_type",
    "reason",
)


@dataclass
class ExternalHttpRequestState:
    request_id: str
    agent_id: uuid.UUID
    started_at: float = field(default_factory=time.monotonic)
    mode: str | None = None
    stage: str = "received"
    payload_bytes: int | None = None

    def elapsed_ms(self) -> int:
        return max(0, round((time.monotonic() - self.started_at) * 1000))


def _log_value(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


def _log_external_http_event(
    level: str,
    event: str,
    state: ExternalHttpRequestState,
    **fields: Any,
) -> None:
    values: dict[str, Any] = {
        "event": event,
        "request_id": state.request_id,
        "agent_id": state.agent_id,
        "mode": state.mode,
        "stage": state.stage,
        "duration_ms": state.elapsed_ms(),
        "payload_bytes": state.payload_bytes,
    }
    values.update({key: value for key, value in fields.items() if key in _LOG_FIELD_ORDER})
    message = "[ExternalHTTP] " + " ".join(
        f"{key}={_log_value(values[key])}"
        for key in _LOG_FIELD_ORDER
        if values.get(key) is not None
    )
    try:
        logger.log(level.upper(), message)
    except Exception:
        pass
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Run Ruff for the new surface**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m ruff check app/api/external_http.py tests/test_external_http_logging.py
```

Expected: exit 0 with no diagnostics.

- [ ] **Step 6: Commit the lifecycle primitive**

```powershell
git add backend/app/api/external_http.py backend/tests/test_external_http_logging.py
git commit -m "feat: add safe external HTTP lifecycle logs"
```

### Task 2: Stage-aware failures and sanitized public responses

**Files:**
- Modify: `backend/app/api/external_http.py:5-30,172-294`
- Modify: `backend/tests/test_external_http_logging.py`

**Interfaces:**
- Consumes: `ExternalHttpRequestState` and `_log_external_http_event` from Task 1.
- Produces: `ExternalHttpProcessingError`, `_processing_stage(state, stage, public_reason)`, `_public_error_response(state, reason, status_code)`, and `_log_unexpected_failure(state, exc)`.

- [ ] **Step 1: Write failing tests for stage wrapping, cancellation, and sanitized responses**

Append to `backend/tests/test_external_http_logging.py`:

```python
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
```

- [ ] **Step 2: Run the five new tests and verify RED**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q
```

Expected: failures because the stage context manager, processing error, public response, and failure logger do not exist.

- [ ] **Step 3: Implement stage and public-error helpers**

Add imports:

```python
import traceback
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi.responses import JSONResponse
```

Add after `ExternalHttpRequestState`:

```python
class ExternalHttpProcessingError(Exception):
    def __init__(self, stage: str, public_reason: str) -> None:
        super().__init__(public_reason)
        self.stage = stage
        self.public_reason = public_reason


@asynccontextmanager
async def _processing_stage(
    state: ExternalHttpRequestState,
    stage: str,
    public_reason: str,
) -> AsyncIterator[None]:
    state.stage = stage
    try:
        yield
    except (HTTPException, asyncio.CancelledError, ExternalHttpProcessingError):
        raise
    except Exception as exc:
        raise ExternalHttpProcessingError(stage, public_reason) from exc


def _public_error_response(
    state: ExternalHttpRequestState,
    reason: str,
    *,
    status_code: int,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "request_id": state.request_id,
            "error": reason,
        },
    )


def _public_reason(exc: Exception) -> str:
    if isinstance(exc, ExternalHttpProcessingError):
        return exc.public_reason
    return "Internal processing failed"


def _log_unexpected_failure(state: ExternalHttpRequestState, exc: Exception) -> None:
    root_exc = exc.__cause__ or exc
    reason = _public_reason(exc)
    _log_external_http_event(
        "ERROR",
        "failed",
        state,
        error_type=type(root_exc).__name__,
        reason=reason,
    )
    trace = "".join(traceback.format_exception(type(root_exc), root_exc, root_exc.__traceback__))
    try:
        logger.error(f"[ExternalHTTP] traceback request_id={state.request_id!r}\n{trace}")
    except Exception:
        pass
```

- [ ] **Step 4: Replace the processor with the exact three stage boundaries**

Replace `_process_external_http_message` with this implementation; the storage payloads remain byte-for-byte equivalent to the existing behavior:

```python
async def _process_external_http_message(
    *,
    agent_id: uuid.UUID,
    message: ExternalHttpMessageIn,
    request_id: str,
    state: ExternalHttpRequestState,
) -> dict:
    from app.api.feishu import _call_llm_with_config, _load_agent_and_model
    from app.models.agent import DEFAULT_CONTEXT_WINDOW_SIZE
    from app.models.chat_session import ChatSession
    from app.services.activity_logger import log_activity

    async with _processing_stage(state, "prepare_session", "Failed to prepare agent session"):
        async with async_session() as db:
            agent, model, fallback_model = await _load_agent_and_model(db, agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            ctx_size = agent.context_window_size or DEFAULT_CONTEXT_WINDOW_SIZE
            external_user_id = message.external_user_id.strip()
            external_name = (message.external_user_name or "").strip() or f"External User {external_user_id[:8]}"
            platform_user = await channel_user_service.resolve_channel_user(
                db=db,
                agent=agent,
                channel_type=CHANNEL_TYPE,
                external_user_id=external_user_id,
                extra_info={
                    "name": external_name,
                    "external_id": external_user_id,
                },
            )

            external_conv = (message.conversation_id or external_user_id).strip()
            external_conv_id = f"{CHANNEL_TYPE}:{external_conv}"
            session = await find_or_create_channel_session(
                db=db,
                agent_id=agent_id,
                user_id=platform_user.id,
                external_conv_id=external_conv_id,
                source_channel=CHANNEL_TYPE,
                first_message_title=message.content,
            )
            session_id = str(session.id)

            history_r = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.agent_id == agent_id, ChatMessage.conversation_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(ctx_size)
            )
            history = [{"role": item.role, "content": item.content} for item in reversed(history_r.scalars().all())]

            content_for_llm = _llm_text(message)
            db.add(
                ChatMessage(
                    agent_id=agent_id,
                    user_id=platform_user.id,
                    role="user",
                    content=content_for_llm,
                    conversation_id=session_id,
                )
            )
            session.last_message_at = datetime.now(timezone.utc)
            await db.commit()
            platform_user_id = platform_user.id

    async with _processing_stage(state, "agent_inference", "Agent inference failed"):
        reply_text = await _call_llm_with_config(
            agent,
            model,
            fallback_model,
            agent_id,
            content_for_llm,
            history=history,
            user_id=platform_user_id,
            session_id=session_id,
        )

    async with _processing_stage(state, "save_response", "Failed to save agent response"):
        async with async_session() as db:
            db.add(
                ChatMessage(
                    agent_id=agent_id,
                    user_id=platform_user_id,
                    role="assistant",
                    content=reply_text,
                    conversation_id=session_id,
                )
            )
            session_r = await db.execute(select(ChatSession).where(ChatSession.id == uuid.UUID(session_id)))
            session = session_r.scalar_one_or_none()
            if session:
                session.last_message_at = datetime.now(timezone.utc)
            await db.commit()

        await log_activity(
            agent_id,
            "chat_reply",
            f"Replied to external HTTP message: {reply_text[:80]}",
            detail={
                "channel": CHANNEL_TYPE,
                "request_id": request_id,
                "external_user_id": external_user_id,
                "conversation_id": message.conversation_id,
                "user_text": message.content[:500],
                "reply": reply_text[:500],
            },
        )

    return {
        "ok": True,
        "request_id": request_id,
        "session_id": session_id,
        "reply": reply_text,
    }
```

- [ ] **Step 5: Run focused tests and Ruff**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q
..\.venv\Scripts\python.exe -m ruff check app/api/external_http.py tests/test_external_http_logging.py
```

Expected: `7 passed`; Ruff exits 0.

- [ ] **Step 6: Commit stage-aware exception behavior**

```powershell
git add backend/app/api/external_http.py backend/tests/test_external_http_logging.py
git commit -m "feat: classify external HTTP processing failures"
```

### Task 3: Cancellable timeout runner and long-processing heartbeat

**Files:**
- Modify: `backend/app/api/external_http.py:5-20,36-170`
- Modify: `backend/tests/test_external_http_logging.py`

**Interfaces:**
- Consumes: `ExternalHttpRequestState` and `_log_external_http_event`.
- Produces: `_run_with_heartbeat(operation, state, timeout_seconds, heartbeat_interval)`.

- [ ] **Step 1: Write failing tests for heartbeat, cancellation, and no retry**

Append:

```python
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
```

- [ ] **Step 2: Run the three tests and verify RED**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q
```

Expected: failures because `_run_with_heartbeat` does not exist.

- [ ] **Step 3: Implement heartbeat and timeout control**

Add imports:

```python
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager, suppress
from typing import Any, TypeVar
```

Add a type variable and helpers near the lifecycle helpers:

```python
T = TypeVar("T")


async def _emit_processing_heartbeat(
    state: ExternalHttpRequestState,
    interval_seconds: float,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        _log_external_http_event("INFO", "processing", state)


async def _run_with_heartbeat(
    operation: Awaitable[T],
    *,
    state: ExternalHttpRequestState,
    timeout_seconds: float,
    heartbeat_interval: float = PROCESSING_HEARTBEAT_SECONDS,
) -> T:
    heartbeat = asyncio.create_task(_emit_processing_heartbeat(state, heartbeat_interval))
    try:
        return await asyncio.wait_for(operation, timeout=timeout_seconds)
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
```

Do not shield `operation`: `asyncio.wait_for` must cancel and await it on timeout, and caller cancellation must propagate through it.

- [ ] **Step 4: Run focused tests three consecutive times**

Run:

```powershell
Set-Location backend
1..3 | ForEach-Object { ..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q }
```

Expected: all three runs pass, demonstrating the short async timing tests are stable.

- [ ] **Step 5: Run Ruff and commit**

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m ruff check app/api/external_http.py tests/test_external_http_logging.py
Set-Location ..
git add backend/app/api/external_http.py backend/tests/test_external_http_logging.py
git commit -m "feat: bound external HTTP model processing"
```

Expected: Ruff exits 0 and the commit succeeds.

### Task 4: Synchronous/asynchronous orchestration and endpoint integration

**Files:**
- Modify: `backend/app/api/external_http.py:410-476`
- Modify: `backend/tests/test_external_http_logging.py`

**Interfaces:**
- Consumes: all helpers from Tasks 1–3 and existing `_process_external_http_message`.
- Produces: `_run_sync_external_http`, `_run_async_external_http`, `_start_external_http_background_task`, `_EXTERNAL_HTTP_BACKGROUND_TASKS`, and the integrated `external_http_message` endpoint.

- [ ] **Step 1: Write failing synchronous orchestration tests**

Append:

```python
async def test_sync_orchestrator_returns_success_and_logs_completion(monkeypatch, log_messages):
    state = make_state()
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
    assert 'session_id="session-1"' in output
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
```

- [ ] **Step 2: Write failing asynchronous orchestration tests**

Append:

```python
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


async def test_endpoint_logs_received_and_validated_without_request_secrets(monkeypatch, log_messages):
    agent_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    body = json.dumps(
        {
            "content": "private-message",
            "external_user_id": "private-user",
            "conversation_id": "private-conversation",
            "metadata": {"secret": "metadata-secret"},
            "mode": "sync",
        }
    ).encode()
    config = SimpleNamespace(
        agent_id=agent_id,
        encrypt_key=None,
        extra_config={
            "api_key_hash": external_http._hash_secret("ext-secret-key"),
            "require_hmac": False,
            "max_payload_bytes": 65536,
            "sync_timeout_seconds": 120,
        },
    )
    agent = SimpleNamespace(webhook_rate_limit=5)

    class FakeResult:
        def __init__(self, value):
            self.value = value

        def scalar_one_or_none(self):
            return self.value

    class FakeSession:
        def __init__(self):
            self.values = iter((config, agent))

        async def execute(self, _query):
            return FakeResult(next(self.values))

    class FakeSessionContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    class FakeRequest:
        headers = {"authorization": "Bearer ext-secret-key"}

        async def body(self):
            return body

    captured_state = None

    async def count_hits(_config):
        return 1

    async def run_sync(*, state, message, timeout_seconds, **_kwargs):
        nonlocal captured_state
        captured_state = state
        assert message.mode == "sync"
        assert timeout_seconds == 120
        return {"ok": True, "request_id": state.request_id, "session_id": "session-1", "reply": "done"}

    monkeypatch.setattr(external_http, "async_session", lambda: FakeSessionContext())
    monkeypatch.setattr(external_http, "_record_and_count_hits", count_hits)
    monkeypatch.setattr(external_http, "_run_sync_external_http", run_sync)

    result = await external_http.external_http_message(agent_id, FakeRequest())

    assert result["ok"] is True
    assert captured_state is not None
    output = "\n".join(log_messages)
    assert 'event="received"' in output
    assert 'event="validated"' in output
    assert f'request_id="{captured_state.request_id}"' in output
    for secret in (
        "private-message",
        "private-user",
        "private-conversation",
        "metadata-secret",
        "ext-secret-key",
    ):
        assert secret not in output
```

- [ ] **Step 3: Run the six orchestration/integration tests and verify RED**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q
```

Expected: failures because the sync/async orchestration and task ownership helpers do not exist.

- [ ] **Step 4: Implement synchronous and asynchronous orchestration**

Add near the runner helpers:

```python
_EXTERNAL_HTTP_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def _log_rejected(state: ExternalHttpRequestState, exc: HTTPException) -> None:
    _log_external_http_event(
        "WARNING",
        "rejected",
        state,
        status_code=exc.status_code,
        reason="expected_http_error",
    )


async def _run_sync_external_http(
    *,
    state: ExternalHttpRequestState,
    message: ExternalHttpMessageIn,
    timeout_seconds: float,
    heartbeat_interval: float = PROCESSING_HEARTBEAT_SECONDS,
) -> dict | JSONResponse:
    try:
        result = await _run_with_heartbeat(
            _process_external_http_message(
                agent_id=state.agent_id,
                message=message,
                request_id=state.request_id,
                state=state,
            ),
            state=state,
            timeout_seconds=timeout_seconds,
            heartbeat_interval=heartbeat_interval,
        )
    except TimeoutError:
        _log_external_http_event("ERROR", "timeout", state, status_code=504, reason="Agent processing timed out")
        return _public_error_response(state, "Agent processing timed out", status_code=504)
    except HTTPException as exc:
        _log_rejected(state, exc)
        raise
    except asyncio.CancelledError:
        _log_external_http_event("WARNING", "failed", state, reason="request_cancelled")
        raise
    except Exception as exc:
        _log_unexpected_failure(state, exc)
        return _public_error_response(state, _public_reason(exc), status_code=500)

    state.stage = "completed"
    _log_external_http_event("INFO", "completed", state, status_code=200, session_id=result.get("session_id"))
    return result


async def _run_async_external_http(
    *,
    state: ExternalHttpRequestState,
    message: ExternalHttpMessageIn,
    timeout_seconds: float = ASYNC_PROCESSING_TIMEOUT_SECONDS,
    heartbeat_interval: float = PROCESSING_HEARTBEAT_SECONDS,
) -> None:
    try:
        result = await _run_with_heartbeat(
            _process_external_http_message(
                agent_id=state.agent_id,
                message=message,
                request_id=state.request_id,
                state=state,
            ),
            state=state,
            timeout_seconds=timeout_seconds,
            heartbeat_interval=heartbeat_interval,
        )
    except TimeoutError:
        _log_external_http_event("ERROR", "timeout", state, reason="Agent processing timed out")
        return
    except HTTPException as exc:
        _log_rejected(state, exc)
        return
    except asyncio.CancelledError:
        _log_external_http_event("WARNING", "failed", state, reason="request_cancelled")
        raise
    except Exception as exc:
        _log_unexpected_failure(state, exc)
        return

    state.stage = "completed"
    _log_external_http_event("INFO", "completed", state, status_code=200, session_id=result.get("session_id"))


def _consume_external_http_background_task(task: asyncio.Task[None]) -> None:
    _EXTERNAL_HTTP_BACKGROUND_TASKS.discard(task)
    if not task.cancelled():
        task.exception()


def _start_external_http_background_task(
    *,
    state: ExternalHttpRequestState,
    message: ExternalHttpMessageIn,
) -> asyncio.Task[None]:
    task = asyncio.create_task(_run_async_external_http(state=state, message=message))
    _EXTERNAL_HTTP_BACKGROUND_TASKS.add(task)
    task.add_done_callback(_consume_external_http_background_task)
    return task
```

- [ ] **Step 5: Integrate the request state into the endpoint**

Remove `Response` from the FastAPI import because the endpoint now uses `JSONResponse`. Replace `external_http_message` with the complete implementation:

```python
@router.post("/channel/external-http/{agent_id}/message")
async def external_http_message(
    agent_id: uuid.UUID,
    request: Request,
):
    state = ExternalHttpRequestState(request_id=str(uuid.uuid4()), agent_id=agent_id)
    _log_external_http_event("INFO", "received", state)

    try:
        async with async_session() as db:
            result = await db.execute(
                select(ChannelConfig).where(
                    ChannelConfig.agent_id == agent_id,
                    ChannelConfig.channel_type == CHANNEL_TYPE,
                    ChannelConfig.is_configured == True,  # noqa: E712
                )
            )
            config = result.scalar_one_or_none()
            if not config:
                raise HTTPException(status_code=404, detail="External HTTP channel not configured")

            _verify_api_key(config, request)

            max_payload = int((config.extra_config or {}).get("max_payload_bytes") or DEFAULT_MAX_PAYLOAD_BYTES)
            body = await request.body()
            if len(body) > max_payload:
                raise HTTPException(status_code=413, detail="Payload too large")

            _verify_hmac_signature(config, request, body)

            hit_count = await _record_and_count_hits(config)
            from app.models.agent import Agent

            agent_r = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = agent_r.scalar_one_or_none()
            rate_limit = (agent.webhook_rate_limit if agent else None) or 5
            if hit_count > rate_limit:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            timeout_seconds = max(
                5,
                min(
                    300,
                    int((config.extra_config or {}).get("sync_timeout_seconds") or DEFAULT_SYNC_TIMEOUT_SECONDS),
                ),
            )

        try:
            payload = ExternalHttpMessageIn.model_validate_json(body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid request body: {exc}") from None
    except HTTPException as exc:
        _log_rejected(state, exc)
        raise
    except asyncio.CancelledError:
        _log_external_http_event("WARNING", "failed", state, reason="request_cancelled")
        raise
    except Exception as exc:
        _log_unexpected_failure(state, exc)
        return _public_error_response(state, "Internal processing failed", status_code=500)

    state.mode = payload.mode
    state.payload_bytes = len(body)
    state.stage = "validated"
    _log_external_http_event("INFO", "validated", state)

    if payload.mode == "async":
        state.stage = "accepted"
        _start_external_http_background_task(state=state, message=payload)
        _log_external_http_event("INFO", "accepted", state, status_code=200)
        return {"ok": True, "status": "accepted", "request_id": state.request_id}

    return await _run_sync_external_http(
        state=state,
        message=payload,
        timeout_seconds=timeout_seconds,
    )
```

Delete the old nested `_log_background_result` callback and old direct `asyncio.wait_for` block.

- [ ] **Step 6: Run all focused tests repeatedly and Ruff**

Run:

```powershell
Set-Location backend
1..3 | ForEach-Object { ..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q }
..\.venv\Scripts\python.exe -m ruff check app/api/external_http.py tests/test_external_http_logging.py
```

Expected: all focused tests pass in all three runs; Ruff exits 0.

- [ ] **Step 7: Commit endpoint integration**

```powershell
Set-Location ..
git add backend/app/api/external_http.py backend/tests/test_external_http_logging.py
git commit -m "feat: control external HTTP request outcomes"
```

### Task 5: Operator documentation and final verification

**Files:**
- Modify: `deploy/docs/external-http-channel.md`
- Verify: `backend/app/api/external_http.py`
- Verify: `backend/tests/test_external_http_logging.py`

**Interfaces:**
- Consumes: the final event names, safe fields, public error reasons, and timeout behavior from Tasks 1–4.
- Produces: operator-facing Docker troubleshooting guidance and final verification evidence.

- [ ] **Step 1: Document the exact operator-visible behavior**

Add a section after the existing error/troubleshooting section in `deploy/docs/external-http-channel.md`:

````markdown
### Docker 请求生命周期日志

External HTTP 调用的新增生命周期日志只输出到后端容器 stdout，不写入新的请求日志表，也不提供日志查询 API。可通过下面的命令按请求 ID 排查：

```bash
docker logs clawith-backend 2>&1 | grep 'request_id="225e7600-8247-46ae-8fbe-df29f5951c8d"'
```

事件包括 `received`、`validated`、`accepted`、`processing`、`completed`、`rejected`、`timeout` 和 `failed`。长时间推理每 30 秒输出一次 `processing` 心跳。

日志不会记录消息正文、metadata、API Key、HMAC 签名、请求头、原始请求体、外部用户 ID 或外部会话 ID。

同步调用继续使用渠道配置的 `sync_timeout_seconds`，范围为 5–300 秒，默认 120 秒。异步调用有固定 300 秒硬超时。任一模式超时后都会取消当前推理，不会自动重试或继续后台执行。

未预期的同步错误返回简短脱敏原因和 `request_id`；完整诊断堆栈只出现在 Docker 日志中。例如：

```json
{
  "ok": false,
  "request_id": "225e7600-8247-46ae-8fbe-df29f5951c8d",
  "error": "Agent inference failed"
}
```
````

- [ ] **Step 2: Run focused verification from a clean process**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest tests/test_external_http_logging.py -q
..\.venv\Scripts\python.exe -m ruff check app/api/external_http.py tests/test_external_http_logging.py
```

Expected: focused tests all pass; Ruff exits 0.

- [ ] **Step 3: Run the full backend suite and preserve the exact result**

Run:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest -q --tb=short
```

Expected branch baseline before this feature: `130 passed, 16 failed`. Compare the final output by test name. No new failure may involve `test_external_http_logging.py` or `app/api/external_http.py`; report every remaining failure rather than claiming the full suite passes.

- [ ] **Step 4: Inspect the final diff for scope and secrets**

Run:

```powershell
Set-Location ..
git diff --check
git diff --stat origin/feature/external-http-channel-call...HEAD
git diff origin/feature/external-http-channel-call...HEAD -- backend/app/api/external_http.py backend/tests/test_external_http_logging.py deploy/docs/external-http-channel.md
```

Expected: only the approved design/plan, External HTTP backend/test changes, and External HTTP deployment documentation appear. Confirm no literal key, signature, message, or metadata fixture is emitted by production logging code.

- [ ] **Step 5: Commit documentation**

```powershell
git add deploy/docs/external-http-channel.md
git commit -m "docs: document external HTTP Docker logs"
```

- [ ] **Step 6: Run final repository status and log checks**

Run:

```powershell
git status --short --branch
git log --oneline --decorate -8
```

Expected: no tracked working-tree changes remain; pre-existing untracked IDE/PID/personal files remain untouched.
