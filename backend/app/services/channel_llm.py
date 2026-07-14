"""Shared LLM runtime for external messaging channels.

Channel API modules should depend on this service instead of importing private
helpers from another channel's API module.
"""

from __future__ import annotations

import asyncio
import traceback
import uuid

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import is_agent_expired
from app.services.llm.utils import truncate_messages_with_pair_integrity


CHANNEL_LLM_TIMEOUT_SECONDS_DEFAULT = 180.0


def get_channel_llm_timeout(model) -> float:
    """Return a model's configured timeout or the channel default."""
    timeout = getattr(model, "request_timeout", None)
    if timeout and float(timeout) > 0:
        return float(timeout)
    return CHANNEL_LLM_TIMEOUT_SECONDS_DEFAULT


async def load_agent_and_models(
    db: AsyncSession,
    agent_id: uuid.UUID,
):
    """Load an agent and its enabled primary/fallback model configs."""
    from app.models.agent import Agent
    from app.models.llm import LLMModel

    agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = agent_result.scalar_one_or_none()
    if not agent:
        return None, None, None

    model = None
    if agent.primary_model_id:
        model_result = await db.execute(select(LLMModel).where(LLMModel.id == agent.primary_model_id))
        model = model_result.scalar_one_or_none()
        if model and not model.enabled:
            logger.info(f"[Channel] Primary model {model.model} is disabled, skipping")
            model = None

    fallback_model = None
    if agent.fallback_model_id:
        fallback_result = await db.execute(select(LLMModel).where(LLMModel.id == agent.fallback_model_id))
        fallback_model = fallback_result.scalar_one_or_none()
        if fallback_model and not fallback_model.enabled:
            logger.info(f"[Channel] Fallback model {fallback_model.model} is disabled, skipping")
            fallback_model = None

    if not model and fallback_model:
        model = fallback_model
        fallback_model = None
        logger.warning(f"[Channel] Primary model unavailable, using fallback: {model.model}")

    return agent, model, fallback_model


async def call_channel_llm(
    agent,
    model,
    fallback_model,
    agent_id: uuid.UUID,
    user_text: str,
    history: list[dict] | None = None,
    user_id=None,
    session_id: str = "",
    on_chunk=None,
    on_thinking=None,
    on_tool_call=None,
) -> str:
    """Call an LLM using detached channel configuration objects."""
    from app.models.agent import DEFAULT_CONTEXT_WINDOW_SIZE
    from app.services.llm import call_llm

    if is_agent_expired(agent):
        return "This Agent has expired and is off duty. Please contact your admin to extend its service."

    if not model:
        return f"⚠️ {agent.name} 未配置 LLM 模型，请在管理后台设置。"

    messages: list[dict] = []
    ctx_size = agent.context_window_size or DEFAULT_CONTEXT_WINDOW_SIZE
    if history:
        messages.extend(truncate_messages_with_pair_integrity(history, ctx_size))
    messages.append({"role": "user", "content": user_text})

    effective_user_id = user_id or agent_id
    timeout = get_channel_llm_timeout(model)

    try:
        return await asyncio.wait_for(
            call_llm(
                model,
                messages,
                agent.name,
                agent.role_description or "",
                agent_id=agent_id,
                user_id=effective_user_id,
                session_id=session_id,
                supports_vision=getattr(model, "supports_vision", False),
                on_chunk=on_chunk,
                on_thinking=on_thinking,
                on_tool_call=on_tool_call,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[LLM] Call timed out after {timeout}s "
            f"(agent_id={agent_id}, model={getattr(model, 'model', 'unknown')})"
        )
        if fallback_model:
            fallback_timeout = get_channel_llm_timeout(fallback_model)
            logger.info(
                f"[LLM] Retrying timed-out request with fallback model: "
                f"{fallback_model.model} (timeout={fallback_timeout}s)"
            )
            try:
                return await asyncio.wait_for(
                    call_llm(
                        fallback_model,
                        messages,
                        agent.name,
                        agent.role_description or "",
                        agent_id=agent_id,
                        user_id=effective_user_id,
                        session_id=session_id,
                        supports_vision=getattr(fallback_model, "supports_vision", False),
                        on_chunk=on_chunk,
                        on_thinking=on_thinking,
                        on_tool_call=on_tool_call,
                    ),
                    timeout=fallback_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[LLM] Fallback call also timed out after {fallback_timeout}s "
                    f"(agent_id={agent_id}, model={getattr(fallback_model, 'model', 'unknown')})"
                )
                return f"⚠️ Model response timed out (>{int(fallback_timeout)}s). Please retry or shorten your request."
            except Exception as fallback_error:
                traceback.print_exc()
                return f"⚠️ Model error: Primary Timeout | Fallback: {str(fallback_error)[:80]}"
        return f"⚠️ Model response timed out (>{int(timeout)}s). Please retry or shorten your request."
    except Exception as error:
        traceback.print_exc()
        error_msg = str(error) or repr(error)
        logger.error(f"[LLM] Primary model error: {error_msg}")
        if fallback_model:
            logger.info(f"[LLM] Retrying with fallback model: {fallback_model.model}")
            try:
                fallback_timeout = get_channel_llm_timeout(fallback_model)
                return await asyncio.wait_for(
                    call_llm(
                        fallback_model,
                        messages,
                        agent.name,
                        agent.role_description or "",
                        agent_id=agent_id,
                        user_id=effective_user_id,
                        session_id=session_id,
                        supports_vision=getattr(fallback_model, "supports_vision", False),
                        on_chunk=on_chunk,
                        on_thinking=on_thinking,
                        on_tool_call=on_tool_call,
                    ),
                    timeout=fallback_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"[LLM] Fallback call timed out after {fallback_timeout}s "
                    f"(agent_id={agent_id}, model={getattr(fallback_model, 'model', 'unknown')})"
                )
                return f"⚠️ Model error: Primary: {str(error)[:80]} | Fallback Timeout"
            except Exception as fallback_error:
                traceback.print_exc()
                return f"⚠️ Model error: Primary: {str(error)[:80]} | Fallback: {str(fallback_error)[:80]}"
        return f"⚠️ 调用模型出错: {error_msg[:150]}"


__all__ = [
    "CHANNEL_LLM_TIMEOUT_SECONDS_DEFAULT",
    "call_channel_llm",
    "get_channel_llm_timeout",
    "load_agent_and_models",
]
