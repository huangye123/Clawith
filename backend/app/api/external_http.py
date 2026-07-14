"""External HTTP channel for business-system integrations."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field
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

T = TypeVar("T")

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
    external_user_id: str = Field(default="external", min_length=1, max_length=255)
    external_user_name: str | None = Field(default=None, max_length=255)
    conversation_id: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] | None = None
    mode: str = Field(default="sync", pattern="^(sync|async)$")


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

    try:
        ts_value = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid HMAC timestamp") from None

    if abs(int(time.time()) - ts_value) > 300:
        raise HTTPException(status_code=401, detail="Expired HMAC timestamp")

    signed_payload = timestamp.encode("utf-8") + b"." + body
    expected = hmac.new(signing_secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    provided = signature.removeprefix("sha256=").strip()
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")


async def _record_and_count_hits(config: ChannelConfig) -> int:
    try:
        redis = await get_redis()
        now = time.time()
        token_key = (config.extra_config or {}).get("api_key_hash") or str(config.agent_id)
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
            "[ExternalHTTP] Rate limiter unavailable: "
            f'reason="dependency_error" exception_type="{type(exc).__name__}"'
        )
        return 1


def _llm_text(message: ExternalHttpMessageIn) -> str:
    if not message.metadata:
        return message.content
    metadata_text = json.dumps(message.metadata, ensure_ascii=False, indent=2, default=str)
    return f"{message.content}\n\n[External HTTP metadata]\n{metadata_text}"


async def _process_external_http_message(
    *,
    agent_id: uuid.UUID,
    message: ExternalHttpMessageIn,
    request_id: str,
    state: ExternalHttpRequestState | None = None,
) -> dict:
    if state is None:
        state = ExternalHttpRequestState(request_id=request_id, agent_id=agent_id, mode=message.mode)

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
) -> asyncio.Task[None]:
    task = asyncio.create_task(_run_async_external_http(state=state, message=message))
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
