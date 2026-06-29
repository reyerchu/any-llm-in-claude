"""Request builder for FreeLLMAPI (OpenAI-compatible chat completions).

freellmapi exposes a standard ``/v1/chat/completions`` surface and forwards to
whichever upstream free-tier provider its router selects, so we build the plain
OpenAI request body from the inbound Anthropic request with no provider-specific
field renaming (``max_tokens`` is kept as-is).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build an OpenAI-format request body from an Anthropic request for FreeLLMAPI."""
    logger.debug(
        "FREELLM_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    request_extra = getattr(request_data, "extra_body", None)
    if isinstance(request_extra, dict) and request_extra:
        body["extra_body"] = dict(request_extra)

    logger.debug(
        "FREELLM_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
