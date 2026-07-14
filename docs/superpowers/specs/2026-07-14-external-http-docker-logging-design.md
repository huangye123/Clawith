# External HTTP Docker Logging and Exception Control Design

## Summary

Improve the existing External HTTP message endpoint with request-lifecycle logging, controlled public error responses, and bounded long-running model inference. All new lifecycle logs are emitted through the existing Loguru stdout sink so they are available through Docker logs. The design adds no request-log database table, log API, frontend screen, migration, configuration service, or dependency.

## Goals

- Make every External HTTP request traceable by a generated `request_id` from receipt through completion.
- Emit concise lifecycle events that are useful in `docker logs` without logging credentials, request content, metadata, or external identifiers.
- Preserve existing expected HTTP errors while converting unexpected failures into controlled JSON responses with a short, sanitized reason.
- Keep full internal diagnostic tracebacks in Docker output without exposing them to callers.
- Bound long-running generic-model inference, provide periodic progress logs, and cancel work when its timeout expires.
- Cover the new behavior with focused automated tests written before production changes.

## Non-Goals

- Persisting request lifecycle logs in PostgreSQL, Redis, or any other store.
- Adding a request-log query API, dashboard, or frontend controls.
- Logging message content, metadata, API keys, HMAC signatures, external user IDs, external conversation IDs, headers, or raw request bodies.
- Retrying model inference automatically.
- Changing existing chat-message, chat-session, or activity-record behavior.
- Raising the existing synchronous timeout limit above 300 seconds.

## Existing Behavior

`POST /api/channel/external-http/{agent_id}/message` currently:

1. Loads the channel configuration.
2. Validates the API key, payload size, optional HMAC signature, and rate limit.
3. Parses the request body.
4. Processes the message synchronously with `asyncio.wait_for`, or starts an unbounded in-process background task for asynchronous mode.
5. Returns a controlled 504 only for synchronous timeout.

The application already configures Loguru to write to stdout. The general HTTP middleware logs method, path, status, and duration, while the External HTTP endpoint only logs rate-limiter degradation and asynchronous task failure. There are no External HTTP-specific backend tests on this branch.

## Architecture

Keep the implementation in `backend/app/api/external_http.py` and extract small, independently testable helpers for four responsibilities:

1. Safe lifecycle log formatting.
2. Stage-aware public error classification.
3. Model-processing execution with heartbeat and timeout control.
4. Background-task ownership and cleanup.

The endpoint remains responsible for HTTP validation and choosing synchronous or asynchronous mode. `_process_external_http_message` remains responsible for resolving the agent and channel user, managing chat persistence, invoking the model, and recording the reply.

No new Loguru sink is added. Lifecycle messages flow through the existing stdout sink and therefore appear in backend container output.

## Request Lifecycle

The endpoint generates `request_id` and captures a monotonic start time before database access or authentication. This guarantees that configuration, authentication, validation, and processing failures share one correlation identifier.

Lifecycle events are:

- `received`: the endpoint accepted control of the HTTP request.
- `validated`: configuration, authentication, signature, size, rate-limit, and body validation completed.
- `accepted`: asynchronous mode returned acceptance to the caller.
- `processing`: a long-running inference heartbeat.
- `completed`: processing and response persistence completed successfully.
- `rejected`: an expected HTTP/business validation error occurred.
- `timeout`: the configured processing deadline expired and processing was cancelled.
- `failed`: an unexpected internal failure occurred.

Messages use stable `key=value` fields because the existing Loguru output format renders the message text but does not render arbitrary bound extras. Each event includes only fields that are available and relevant from this allowlist:

- `event`
- `request_id`
- `agent_id`
- `mode`
- `stage`
- `status_code`
- `duration_ms`
- `payload_bytes`
- `session_id`
- `error_type`
- `reason`

The logger never receives message content, metadata, API keys, HMAC signatures, external user IDs, external conversation IDs, headers, or the raw body. Values are serialized deterministically so tests and operators can search by `request_id` and event name.

Lifecycle logging is best-effort: a formatting or logging problem is contained and must not change the request result.

## Processing Stages and Public Errors

Processing tracks a coarse stage so callers receive a useful but sanitized reason:

- Agent/session preparation failure: `Failed to prepare agent session`.
- Generic model invocation failure: `Agent inference failed`.
- Reply persistence failure: `Failed to save agent response`.
- Any failure outside a classified processing stage: `Internal processing failed`.

The public response does not contain the raw exception string. Raw database errors, provider responses, connection addresses, SQL, stack frames, and secrets therefore do not cross the API boundary.

Expected `HTTPException` responses retain their current status code and detail, including 401, 404, 413, 422, and 429. They produce a `rejected` warning without a traceback.

An unexpected synchronous failure returns HTTP 500:

```json
{
  "ok": false,
  "request_id": "request-uuid",
  "error": "Agent inference failed"
}
```

The Docker log contains the request ID, safe classification, exception type, and a standard Python formatted traceback. The traceback is logged as formatted text rather than through Loguru diagnostic exception rendering, avoiding automatic local-variable dumps that could expose request data.

`asyncio.CancelledError` preserves cancellation semantics. It is logged at the lifecycle boundary when appropriate and re-raised instead of being converted to HTTP 500.

## Long-Running Generic Model Inference

Synchronous mode keeps the existing configurable `sync_timeout_seconds` range of 5 through 300 seconds and the existing default of 120 seconds. The maximum is not increased.

Asynchronous mode receives a fixed 300-second hard processing timeout. This prevents an unresponsive provider call from occupying an in-process task indefinitely without adding another frontend or channel configuration field.

Both modes emit a `processing` heartbeat every 30 seconds while model processing remains active. Each heartbeat includes request ID, current stage, mode, and elapsed duration, but no user or model content.

Timeout behavior is deliberate:

- The active processing task is cancelled.
- Cancellation is awaited so database/model cleanup can run.
- No retry is attempted, preventing duplicate inference charges, duplicated messages, or competing writes.
- Synchronous mode returns HTTP 504 with `request_id` and `Agent processing timed out`.
- Asynchronous mode records `timeout` in Docker logs; the already returned accepted response cannot be changed.
- Timed-out synchronous work is not allowed to continue in the background.

## Asynchronous Task Ownership

Asynchronous requests still return immediately with `ok`, `status=accepted`, and `request_id`. Created tasks are stored in a module-level set so the event loop task has a strong reference for its full lifetime.

A completion callback always:

1. Consumes the result or exception.
2. Emits the final `completed`, `timeout`, or `failed` lifecycle event.
3. Removes the task from the ownership set in a `finally` path.

This prevents unobserved-task warnings and avoids retaining completed tasks.

## Response Compatibility

Successful synchronous and asynchronous response shapes remain unchanged. Expected validation responses remain unchanged. Only unexpected synchronous failures gain a stable JSON body, and timeout JSON receives the same concise public-reason convention.

No request/response schema or frontend change is required.

## Testing Strategy

Add `backend/tests/test_external_http_logging.py`. Tests use short injected heartbeat and timeout values so they complete quickly while exercising real async cancellation behavior.

The focused test suite covers:

1. Safe lifecycle formatting for successful, rejected, timeout, and failed requests.
2. Absence of request content, metadata, API key, signature, external user ID, and external conversation ID from captured logs and public errors.
3. Preservation of expected `HTTPException` status and detail.
4. Sanitized stage-specific HTTP 500 responses for unexpected synchronous errors.
5. Python traceback output in Docker logs without Loguru local-variable diagnostics.
6. Periodic heartbeat emission and heartbeat shutdown after success, failure, cancellation, and timeout.
7. Synchronous timeout cancellation and 504 response behavior.
8. Asynchronous 300-second deadline behavior using an injected short test deadline.
9. Strong task ownership while running and removal after completion.
10. Consumption and logging of asynchronous exceptions.

Verification commands will include the focused pytest file, Ruff on changed Python files, and the full backend test suite. The branch already has 16 unrelated baseline failures in the current local environment; those remain outside this feature's scope and will be reported separately rather than hidden.

## Files

- Modify `backend/app/api/external_http.py` for lifecycle logging, public-error mapping, heartbeat, timeout, and task ownership.
- Create `backend/tests/test_external_http_logging.py` for focused behavior tests.
- Update `deploy/docs/external-http-channel.md` only if implementation changes operator-visible error or timeout documentation beyond what this design already specifies.

## Acceptance Criteria

- Every message request has one request ID across validation, processing, and final outcome logs.
- New lifecycle logs appear only through the existing stdout/Docker logging path.
- Sensitive request and authentication data never appear in lifecycle log messages or public internal-error responses.
- Expected HTTP errors keep their current semantics.
- Unexpected synchronous errors return a sanitized reason and request ID while Docker output contains diagnostic traceback text.
- Synchronous inference respects the existing 5–300 second configuration and cancels on timeout.
- Asynchronous inference cancels after 300 seconds.
- Long-running inference emits 30-second heartbeat logs.
- No automatic model retry occurs.
- Background tasks are strongly referenced, consumed, and removed.
- Focused tests and Ruff pass.
