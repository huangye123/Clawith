"""External HTTP channel for business-system integrations."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import time
import traceback
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import get_redis
from app.core.permissions import check_agent_access, is_agent_creator
from app.core.security import get_current_user
from app.database import async_session, get_db
from app.models.audit import ChatMessage
from app.models.channel_config import ChannelConfig
from app.models.user import User
from app.schemas.schemas import ChannelConfigOut
from app.services.channel_session import find_or_create_channel_session
from app.services.channel_user_service import channel_user_service

router = APIRouter(tags=["external-http"])

CHANNEL_TYPE = "external_http"
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024
DEFAULT_SYNC_TIMEOUT_SECONDS = 120
PROCESSING_HEARTBEAT_SECONDS = 30.0
ASYNC_PROCESSING_TIMEOUT_SECONDS = 300.0
HMAC_TIMESTAMP_WINDOW_SECONDS = 300
MAX_EXTERNAL_USER_ID_LENGTH = 100
MAX_EXTERNAL_USER_NAME_LENGTH = 100
MAX_CONVERSATION_ID_LENGTH = 255
MAX_METADATA_JSON_BYTES = 64 * 1024
MAX_METADATA_DEPTH = 10
MAX_METADATA_NODES = 1024
GLOBAL_PROCESSING_CONCURRENCY = 128
PER_AGENT_PROCESSING_CONCURRENCY = 4

T = TypeVar("T")

_GLOBAL_PROCESSING_SEMAPHORE = asyncio.Semaphore(GLOBAL_PROCESSING_CONCURRENCY)
_AGENT_PROCESSING_SEMAPHORES: dict[uuid.UUID, asyncio.Semaphore] = {}
_LOCAL_RATE_HITS: dict[str, deque[float]] = {}
_LOCAL_RATE_LOCK = asyncio.Lock()

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
    session_id: str | None = None

    def elapsed_ms(self) -> int:
        return max(0, round((time.monotonic() - self.started_at) * 1000))


class ExternalHttpProcessingError(Exception):
    def __init__(
        self,
        stage: str,
        public_reason: str,
        *,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "processing_failed",
    ) -> None:
        super().__init__(public_reason)
        self.stage = stage
        self.public_reason = public_reason
        self.status_code = status_code
        self.error_code = error_code


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
    error_code: str = "internal_processing_failed",
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    response_headers = dict(headers or {})
    response_headers.setdefault("X-Request-ID", state.request_id)
    return JSONResponse(
        status_code=status_code,
        headers=response_headers,
        content={
            "ok": False,
            "request_id": state.request_id,
            "error_code": error_code,
            "error": reason,
        },
    )


_HTTP_ERROR_CODES = {
    "External HTTP channel not configured": "channel_not_configured",
    "Missing external HTTP channel API key": "missing_api_key",
    "Invalid external HTTP channel API key": "invalid_api_key",
    "External HTTP channel signing secret is not configured": "signing_secret_not_configured",
    "Missing HMAC signature headers": "missing_hmac_headers",
    "Invalid HMAC timestamp": "invalid_hmac_timestamp",
    "Expired HMAC timestamp": "expired_hmac_timestamp",
    "Invalid HMAC signature": "invalid_hmac_signature",
    "Replayed HMAC request": "replayed_hmac_request",
    "HMAC replay protection unavailable": "replay_protection_unavailable",
    "Payload too large": "payload_too_large",
    "Invalid Content-Length": "invalid_content_length",
    "Invalid request body": "invalid_request_body",
    "Rate limit exceeded": "rate_limit_exceeded",
    "Processing capacity exhausted": "processing_capacity_exhausted",
    "Agent not found": "agent_not_found",
}


def _http_exception_response(state: ExternalHttpRequestState, exc: HTTPException) -> JSONResponse:
    reason = exc.detail if isinstance(exc.detail, str) else "Request rejected"
    headers = dict(exc.headers or {})
    if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        headers.setdefault("Retry-After", "60")
    elif exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        headers.setdefault("Retry-After", "1")
    return _public_error_response(
        state,
        reason,
        status_code=exc.status_code,
        error_code=_HTTP_ERROR_CODES.get(reason, "request_rejected"),
        headers=headers or None,
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
    trace = "".join(
        (
            "Traceback (most recent call last):\n",
            *traceback.format_tb(root_exc.__traceback__),
            f"{type(root_exc).__name__}: <message redacted>\n",
        )
    )
    try:
        logger.error(f"[ExternalHTTP] traceback request_id={state.request_id!r}\n{trace}")
    except Exception:
        pass


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
        "session_id": state.session_id,
    }
    values.update({key: value for key, value in fields.items() if key in _LOG_FIELD_ORDER})
    message = "[ExternalHTTP] " + " ".join(
        f"{key}={_log_value(values[key])}" for key in _LOG_FIELD_ORDER if values.get(key) is not None
    )
    try:
        logger.log(level.upper(), message)
    except Exception:
        pass


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
        caller = asyncio.current_task()
        cancellation_count = caller.cancelling() if caller is not None else 0
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            if caller is not None and caller.cancelling() > cancellation_count:
                raise


class ExternalHttpChannelConfigIn(BaseModel):
    require_hmac: bool = False
    sync_timeout_seconds: int = Field(DEFAULT_SYNC_TIMEOUT_SECONDS, ge=5, le=300)
    max_payload_bytes: int = Field(DEFAULT_MAX_PAYLOAD_BYTES, ge=1024, le=1024 * 1024)
    regenerate_api_key: bool = False
    regenerate_signing_secret: bool = False


class ExternalHttpMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=60000)
    external_user_id: str = Field(default="external", min_length=1, max_length=MAX_EXTERNAL_USER_ID_LENGTH)
    external_user_name: str | None = Field(default=None, max_length=MAX_EXTERNAL_USER_NAME_LENGTH)
    conversation_id: str | None = Field(default=None, max_length=MAX_CONVERSATION_ID_LENGTH)
    metadata: dict[str, Any] | None = None
    mode: str = Field(default="sync", pattern="^(sync|async)$")

    @field_validator("content")
    @classmethod
    def _content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("external_user_id")
    @classmethod
    def _normalize_external_user_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("external_user_id must not be blank")
        return normalized

    @field_validator("external_user_name", "conversation_id")
    @classmethod
    def _normalize_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("metadata")
    @classmethod
    def _bound_metadata(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        if len(encoded) > MAX_METADATA_JSON_BYTES:
            raise ValueError("metadata is too large")

        nodes = 0
        stack: list[tuple[Any, int]] = [(value, 1)]
        while stack:
            current, depth = stack.pop()
            nodes += 1
            if nodes > MAX_METADATA_NODES:
                raise ValueError("metadata contains too many values")
            if depth > MAX_METADATA_DEPTH:
                raise ValueError("metadata is nested too deeply")
            if isinstance(current, dict):
                stack.extend((item, depth + 1) for item in current.values())
            elif isinstance(current, list):
                stack.extend((item, depth + 1) for item in current)
        return value


_EXTERNAL_HTTP_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def _log_rejected(state: ExternalHttpRequestState, exc: HTTPException) -> None:
    _log_external_http_event(
        "WARNING",
        "rejected",
        state,
        status_code=exc.status_code,
        reason="expected_http_error",
    )


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _new_api_key() -> str:
    return f"ext-{secrets.token_urlsafe(32)}"


def _new_signing_secret() -> str:
    return secrets.token_urlsafe(32)


def _extract_api_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return (request.headers.get("x-api-key") or "").strip()


def _safe_extra(config: ChannelConfig) -> dict:
    extra = dict(config.extra_config or {})
    extra.pop("api_key_hash", None)
    return extra


def _serialize_config(
    config: ChannelConfig,
    *,
    api_key: str | None = None,
    signing_secret: str | None = None,
    webhook_url: str | None = None,
) -> dict:
    payload = ChannelConfigOut.model_validate(config).model_dump()
    payload["extra_config"] = _safe_extra(config)
    payload["app_secret"] = None
    payload["encrypt_key"] = None
    if api_key:
        payload["api_key"] = api_key
    if signing_secret:
        payload["signing_secret"] = signing_secret
    if webhook_url:
        payload["webhook_url"] = webhook_url
    return payload


async def _public_message_url(request: Request, db: AsyncSession, agent_id: uuid.UUID) -> str:
    from app.services.platform_service import platform_service

    public_base = await platform_service.get_public_base_url(db, request)
    return f"{public_base.rstrip('/')}/api/channel/external-http/{agent_id}/message"


def _verify_api_key(config: ChannelConfig, request: Request) -> None:
    expected_hash = (config.extra_config or {}).get("api_key_hash") or ""
    api_key = _extract_api_key(request)
    if not api_key or not expected_hash:
        raise HTTPException(status_code=401, detail="Missing external HTTP channel API key")
    if not hmac.compare_digest(_hash_secret(api_key), expected_hash):
        raise HTTPException(status_code=401, detail="Invalid external HTTP channel API key")


def _verify_hmac_signature(config: ChannelConfig, request: Request, body: bytes) -> None:
    extra = config.extra_config or {}
    if not extra.get("require_hmac"):
        return

    signing_secret = config.encrypt_key or ""
    if not signing_secret:
        raise HTTPException(status_code=401, detail="External HTTP channel signing secret is not configured")

    timestamp = request.headers.get("x-timestamp", "")
    signature = request.headers.get("x-signature-sha256", "")
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature headers")

    if len(timestamp) > 12 or not timestamp.isascii() or not timestamp.isdigit():
        raise HTTPException(status_code=401, detail="Invalid HMAC timestamp") from None
    ts_value = int(timestamp)

    if abs(int(time.time()) - ts_value) > HMAC_TIMESTAMP_WINDOW_SECONDS:
        raise HTTPException(status_code=401, detail="Expired HMAC timestamp")

    signed_payload = timestamp.encode("utf-8") + b"." + body
    expected = hmac.new(signing_secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    provided = signature.removeprefix("sha256=").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", provided) is None:
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")


async def _claim_hmac_signature(config: ChannelConfig, request: Request) -> None:
    if not (config.extra_config or {}).get("require_hmac"):
        return
    signature = request.headers.get("x-signature-sha256", "").removeprefix("sha256=").strip().lower()
    token_key = (config.extra_config or {}).get("api_key_hash") or str(config.agent_id)
    replay_key = f"external_http:replay:{token_key[:16]}:{signature}"
    try:
        redis = await get_redis()
        claimed = await redis.set(replay_key, "1", ex=HMAC_TIMESTAMP_WINDOW_SECONDS, nx=True)
    except Exception as exc:
        logger.warning(
            "[ExternalHTTP] HMAC replay protection unavailable: "
            f'reason="dependency_error" exception_type="{type(exc).__name__}"'
        )
        raise HTTPException(status_code=503, detail="HMAC replay protection unavailable") from None
    if not claimed:
        raise HTTPException(status_code=409, detail="Replayed HMAC request")


def _rate_limit_key(config: ChannelConfig) -> str:
    return (config.extra_config or {}).get("api_key_hash") or str(config.agent_id)


async def _record_local_rate_hit(token_key: str, now: float) -> int:
    async with _LOCAL_RATE_LOCK:
        hits = _LOCAL_RATE_HITS.setdefault(token_key, deque())
        cutoff = now - 60
        while hits and hits[0] <= cutoff:
            hits.popleft()
        hits.append(now)
        return len(hits)


async def _record_and_count_hits(config: ChannelConfig) -> int:
    now = time.time()
    token_key = _rate_limit_key(config)
    try:
        redis = await get_redis()
        key = f"external_http:rate:{token_key}"
        member = f"{now}:{secrets.token_hex(4)}"
        async with redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, now - 60)
            pipe.zadd(key, {member: now})
            pipe.zcard(key)
            pipe.expire(key, 120)
            _, _, count, _ = await pipe.execute()
        return int(count)
    except Exception as exc:
        logger.warning(
            f'[ExternalHTTP] Rate limiter unavailable: reason="local_fallback" exception_type="{type(exc).__name__}"'
        )
        return await _record_local_rate_hit(token_key, now)


def _bounded_config_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))


async def _read_limited_body(request: Request, max_payload_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from None
        if declared_size < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
        if declared_size > max_payload_bytes:
            raise HTTPException(status_code=413, detail="Payload too large")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_payload_bytes:
            raise HTTPException(status_code=413, detail="Payload too large")
        body.extend(chunk)
    return bytes(body)


def _external_conversation_key(external_user_id: str, conversation_id: str | None) -> str:
    user_digest = hashlib.sha256(external_user_id.encode("utf-8")).hexdigest()
    conversation = conversation_id or external_user_id
    conversation_digest = hashlib.sha256(conversation.encode("utf-8")).hexdigest()
    return f"{CHANNEL_TYPE}:{user_digest}:{conversation_digest}"


@dataclass
class _ProcessingLease:
    global_semaphore: asyncio.Semaphore
    agent_semaphore: asyncio.Semaphore
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.agent_semaphore.release()
        self.global_semaphore.release()


async def _try_acquire_processing_lease(agent_id: uuid.UUID) -> _ProcessingLease | None:
    agent_semaphore = _AGENT_PROCESSING_SEMAPHORES.setdefault(
        agent_id,
        asyncio.Semaphore(PER_AGENT_PROCESSING_CONCURRENCY),
    )
    if _GLOBAL_PROCESSING_SEMAPHORE.locked() or agent_semaphore.locked():
        return None
    await _GLOBAL_PROCESSING_SEMAPHORE.acquire()
    if agent_semaphore.locked():
        _GLOBAL_PROCESSING_SEMAPHORE.release()
        return None
    await agent_semaphore.acquire()
    return _ProcessingLease(_GLOBAL_PROCESSING_SEMAPHORE, agent_semaphore)


def _llm_text(message: ExternalHttpMessageIn) -> str:
    if not message.metadata:
        return message.content
    metadata_text = json.dumps(message.metadata, ensure_ascii=False, indent=2, default=str)
    return f"{message.content}\n\n[External HTTP metadata]\n{metadata_text}"


def _raise_for_llm_error_reply(reply: str) -> None:
    """Turn call_llm's legacy error strings into a channel-level failure."""
    if reply.startswith(("[LLM Error]", "[LLM call error]", "[Error]")):
        raise ExternalHttpProcessingError(
            "agent_inference",
            "Upstream model request failed",
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="upstream_llm_error",
        )


def _tool_result_is_error(result: object) -> bool:
    normalized = str(result or "").strip().lower()
    return normalized.startswith((
        "error:",
        "[error]",
        "[llm error]",
        "[llm call error]",
        "failed:",
        "timeout:",
        "⚠️",
    ))


async def _process_external_http_message(
    *,
    agent_id: uuid.UUID,
    message: ExternalHttpMessageIn,
    request_id: str,
    state: ExternalHttpRequestState | None = None,
) -> dict:
    if state is None:
        state = ExternalHttpRequestState(request_id=request_id, agent_id=agent_id, mode=message.mode)
    # Lifecycle logs intentionally expose the caller's conversation identifier,
    # not Clawith's internal ChatSession UUID.
    state.session_id = message.conversation_id

    from app.models.agent import DEFAULT_CONTEXT_WINDOW_SIZE
    from app.models.chat_session import ChatSession
    from app.services.activity_logger import log_activity
    from app.services.channel_llm import call_channel_llm, load_agent_and_models
    from app.services.chat_session_service import save_tool_call_log
    from app.services.llm.utils import convert_chat_messages_to_llm_format

    async with _processing_stage(state, "prepare_session", "Failed to prepare agent session"):
        async with async_session() as db:
            agent, model, fallback_model = await load_agent_and_models(db, agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            ctx_size = agent.context_window_size or DEFAULT_CONTEXT_WINDOW_SIZE
            external_user_id = message.external_user_id
            external_name = message.external_user_name or f"External User {external_user_id[:8]}"
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
            external_conv_id = _external_conversation_key(external_user_id, message.conversation_id)
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
            history = convert_chat_messages_to_llm_format(reversed(history_r.scalars().all()))

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

    tool_call_summaries: dict[str, dict[str, Any]] = {}
    persisted_tool_calls: set[str] = set()

    async def _on_tool_call(event: dict) -> None:
        tool_name = str(event.get("name") or "unknown_tool")
        call_id = str(event.get("call_id") or "")
        summary_key = call_id or tool_name
        tool_status = str(event.get("status") or "unknown").lower()
        summary: dict[str, Any] = {
            "name": tool_name,
            "call_id": call_id or None,
            "status": tool_status,
        }
        if tool_status == "done":
            result = event.get("result")
            summary["outcome"] = "error" if _tool_result_is_error(result) else "success"
            if summary_key not in persisted_tool_calls:
                persisted_tool_calls.add(summary_key)
                await save_tool_call_log(
                    agent_id=agent_id,
                    user_id=platform_user_id,
                    conversation_id=session_id,
                    tool_name=tool_name,
                    arguments=event.get("args") or event.get("arguments") or {},
                    result=str(result or "")[:500],
                    status=tool_status,
                    tool_call_id=call_id or None,
                    reasoning_content=event.get("reasoning_content"),
                )
        tool_call_summaries[summary_key] = summary

    async with _processing_stage(state, "agent_inference", "Agent inference failed"):
        reply_text = await call_channel_llm(
            agent,
            model,
            fallback_model,
            agent_id,
            content_for_llm,
            history=history,
            user_id=platform_user_id,
            session_id=session_id,
            on_tool_call=_on_tool_call,
        )
        _raise_for_llm_error_reply(reply_text)

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
        "tool_calls": list(tool_call_summaries.values()),
        "tool_errors": [
            summary["name"]
            for summary in tool_call_summaries.values()
            if summary.get("outcome") == "error"
        ],
    }


async def _run_sync_external_http(
    *,
    state: ExternalHttpRequestState,
    message: ExternalHttpMessageIn,
    timeout_seconds: float,
    heartbeat_interval: float = PROCESSING_HEARTBEAT_SECONDS,
    processing_lease: _ProcessingLease | None = None,
) -> dict | JSONResponse:
    try:
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
        finally:
            if processing_lease is not None:
                processing_lease.release()
    except TimeoutError:
        _log_external_http_event("ERROR", "timeout", state, status_code=504, reason="Agent processing timed out")
        return _public_error_response(
            state,
            "Agent processing timed out",
            status_code=504,
            error_code="processing_timeout",
        )
    except HTTPException as exc:
        _log_rejected(state, exc)
        return _http_exception_response(state, exc)
    except asyncio.CancelledError:
        _log_external_http_event("WARNING", "failed", state, reason="request_cancelled")
        raise
    except ExternalHttpProcessingError as exc:
        _log_unexpected_failure(state, exc)
        return _public_error_response(
            state,
            exc.public_reason,
            status_code=exc.status_code,
            error_code=exc.error_code,
        )
    except Exception as exc:
        _log_unexpected_failure(state, exc)
        return _public_error_response(
            state,
            _public_reason(exc),
            status_code=500,
            error_code="processing_failed",
        )

    state.stage = "completed"
    _log_external_http_event("INFO", "completed", state, status_code=200)
    return result


async def _run_async_external_http(
    *,
    state: ExternalHttpRequestState,
    message: ExternalHttpMessageIn,
    timeout_seconds: float = ASYNC_PROCESSING_TIMEOUT_SECONDS,
    heartbeat_interval: float = PROCESSING_HEARTBEAT_SECONDS,
    processing_lease: _ProcessingLease | None = None,
) -> None:
    try:
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
        finally:
            if processing_lease is not None:
                processing_lease.release()
    except TimeoutError:
        _log_external_http_event("ERROR", "timeout", state, reason="Agent processing timed out")
        return
    except HTTPException as exc:
        _log_rejected(state, exc)
        return
    except asyncio.CancelledError:
        _log_external_http_event("WARNING", "failed", state, reason="request_cancelled")
        raise
    except ExternalHttpProcessingError as exc:
        _log_unexpected_failure(state, exc)
        return
    except Exception as exc:
        _log_unexpected_failure(state, exc)
        return

    # Validate the processor contract without logging its internal ChatSession UUID.
    result.get("session_id")
    state.stage = "completed"
    _log_external_http_event("INFO", "completed", state, status_code=200)


def _consume_external_http_background_task(
    task: asyncio.Task[None],
    *,
    state: ExternalHttpRequestState,
) -> None:
    try:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _log_unexpected_failure(state, exc)
    finally:
        _EXTERNAL_HTTP_BACKGROUND_TASKS.discard(task)


def _start_external_http_background_task(
    *,
    state: ExternalHttpRequestState,
    message: ExternalHttpMessageIn,
    processing_lease: _ProcessingLease | None = None,
) -> asyncio.Task[None]:
    task = asyncio.create_task(
        _run_async_external_http(
            state=state,
            message=message,
            processing_lease=processing_lease,
        )
    )
    _EXTERNAL_HTTP_BACKGROUND_TASKS.add(task)
    task.add_done_callback(lambda done_task: _consume_external_http_background_task(done_task, state=state))
    return task


@router.post("/agents/{agent_id}/external-http-channel", status_code=status.HTTP_201_CREATED)
async def configure_external_http_channel(
    agent_id: uuid.UUID,
    request: Request,
    data: ExternalHttpChannelConfigIn = ExternalHttpChannelConfigIn(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent, _ = await check_agent_access(db, current_user, agent_id)
    if not is_agent_creator(current_user, agent):
        raise HTTPException(status_code=403, detail="Only creator can configure channel")

    result = await db.execute(
        select(ChannelConfig).where(
            ChannelConfig.agent_id == agent_id,
            ChannelConfig.channel_type == CHANNEL_TYPE,
        )
    )
    config = result.scalar_one_or_none()

    generated_api_key = None
    generated_signing_secret = None
    extra = {
        "require_hmac": data.require_hmac,
        "sync_timeout_seconds": data.sync_timeout_seconds,
        "max_payload_bytes": data.max_payload_bytes,
        "auth_scheme": "bearer",
        "signature": "HMAC-SHA256 over '<x-timestamp>.<raw-body>' in X-Signature-SHA256",
    }

    if config:
        old_extra = config.extra_config or {}
        if data.regenerate_api_key or not old_extra.get("api_key_hash"):
            generated_api_key = _new_api_key()
            extra["api_key_hash"] = _hash_secret(generated_api_key)
        else:
            extra["api_key_hash"] = old_extra.get("api_key_hash")

        if data.regenerate_signing_secret or (data.require_hmac and not config.encrypt_key):
            generated_signing_secret = _new_signing_secret()
            config.encrypt_key = generated_signing_secret

        config.app_id = CHANNEL_TYPE
        config.app_secret = None
        config.extra_config = extra
        config.is_configured = True
        config.is_connected = True
        await db.flush()
    else:
        generated_api_key = _new_api_key()
        extra["api_key_hash"] = _hash_secret(generated_api_key)
        generated_signing_secret = _new_signing_secret() if data.require_hmac else None
        config = ChannelConfig(
            agent_id=agent_id,
            channel_type=CHANNEL_TYPE,
            app_id=CHANNEL_TYPE,
            app_secret=None,
            encrypt_key=generated_signing_secret,
            extra_config=extra,
            is_configured=True,
            is_connected=True,
        )
        db.add(config)
        await db.flush()

    webhook_url = await _public_message_url(request, db, agent_id)
    await db.commit()
    return _serialize_config(
        config,
        api_key=generated_api_key,
        signing_secret=generated_signing_secret,
        webhook_url=webhook_url,
    )


@router.get("/agents/{agent_id}/external-http-channel")
async def get_external_http_channel(
    agent_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await check_agent_access(db, current_user, agent_id)
    result = await db.execute(
        select(ChannelConfig).where(
            ChannelConfig.agent_id == agent_id,
            ChannelConfig.channel_type == CHANNEL_TYPE,
        )
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="External HTTP channel not configured")
    return _serialize_config(config)


@router.get("/agents/{agent_id}/external-http-channel/webhook-url")
async def get_external_http_message_url(
    agent_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return {"webhook_url": await _public_message_url(request, db, agent_id)}


@router.delete("/agents/{agent_id}/external-http-channel", status_code=status.HTTP_204_NO_CONTENT)
async def delete_external_http_channel(
    agent_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    agent, _ = await check_agent_access(db, current_user, agent_id)
    if not is_agent_creator(current_user, agent):
        raise HTTPException(status_code=403, detail="Only creator can remove channel")
    result = await db.execute(
        select(ChannelConfig).where(
            ChannelConfig.agent_id == agent_id,
            ChannelConfig.channel_type == CHANNEL_TYPE,
        )
    )
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="External HTTP channel not configured")
    await db.delete(config)
    await db.commit()


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

            max_payload = _bounded_config_int(
                (config.extra_config or {}).get("max_payload_bytes"),
                default=DEFAULT_MAX_PAYLOAD_BYTES,
                minimum=1024,
                maximum=1024 * 1024,
            )
            body = await _read_limited_body(request, max_payload)
            state.payload_bytes = len(body)

            _verify_hmac_signature(config, request, body)
            await _claim_hmac_signature(config, request)

            hit_count = await _record_and_count_hits(config)
            from app.models.agent import Agent

            agent_r = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = agent_r.scalar_one_or_none()
            rate_limit = _bounded_config_int(
                agent.webhook_rate_limit if agent else None,
                default=5,
                minimum=1,
                maximum=100000,
            )
            if hit_count > rate_limit:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            timeout_seconds = _bounded_config_int(
                (config.extra_config or {}).get("sync_timeout_seconds"),
                default=DEFAULT_SYNC_TIMEOUT_SECONDS,
                minimum=5,
                maximum=300,
            )

        try:
            payload = ExternalHttpMessageIn.model_validate_json(body)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid request body") from None
    except HTTPException as exc:
        _log_rejected(state, exc)
        return _http_exception_response(state, exc)
    except asyncio.CancelledError:
        _log_external_http_event("WARNING", "failed", state, reason="request_cancelled")
        raise
    except Exception as exc:
        _log_unexpected_failure(state, exc)
        return _public_error_response(
            state,
            "Internal processing failed",
            status_code=500,
            error_code="internal_processing_failed",
        )

    state.mode = payload.mode
    state.session_id = payload.conversation_id
    state.stage = "validated"
    _log_external_http_event("INFO", "validated", state)

    processing_lease = await _try_acquire_processing_lease(agent_id)
    if processing_lease is None:
        exc = HTTPException(status_code=503, detail="Processing capacity exhausted")
        _log_rejected(state, exc)
        return _http_exception_response(state, exc)

    if payload.mode == "async":
        state.stage = "accepted"
        try:
            _start_external_http_background_task(
                state=state,
                message=payload,
                processing_lease=processing_lease,
            )
        except Exception:
            processing_lease.release()
            raise
        _log_external_http_event("INFO", "accepted", state, status_code=202)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            headers={"X-Request-ID": state.request_id},
            content={"ok": True, "status": "accepted", "request_id": state.request_id},
        )

    result = await _run_sync_external_http(
        state=state,
        message=payload,
        timeout_seconds=timeout_seconds,
        processing_lease=processing_lease,
    )
    if isinstance(result, JSONResponse):
        return result
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        headers={"X-Request-ID": state.request_id},
        content=result,
    )
