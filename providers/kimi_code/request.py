"""Native Anthropic Messages request builder for Kimi Code."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build JSON for Kimi Code Anthropic-compat ``POST …/coding/v1/messages``."""
    body = build_base_native_anthropic_request_body(
        request_data,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=thinking_enabled,
    )
    body["stream"] = True
    logger.debug(
        "KIMI_CODE_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
