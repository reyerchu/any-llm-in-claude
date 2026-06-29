"""Native Anthropic Messages request builder for the account (OAuth) provider.

The official Anthropic Messages API rejects ``claude setup-token`` OAuth access
tokens unless the request is presented as Claude Code: the first system block must
be the Claude Code identity string. We inject it (idempotently) so a subscription
token authenticates exactly like the official CLI.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    DEFAULT_THINKING_BUDGET_TOKENS,
    build_base_native_anthropic_request_body,
)

# Anthropic enforces this exact identity for subscription (OAuth) access tokens.
CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."


def _ensure_claude_code_system(system: Any) -> Any:
    """Return ``system`` with the Claude Code identity guaranteed as the first block."""
    identity = {"type": "text", "text": CLAUDE_CODE_SYSTEM_PROMPT}

    if system is None or system == "":
        return [identity]

    if isinstance(system, str):
        if system.strip() == CLAUDE_CODE_SYSTEM_PROMPT:
            return [identity]
        return [identity, {"type": "text", "text": system}]

    if isinstance(system, list):
        first = system[0] if system else None
        if (
            isinstance(first, dict)
            and first.get("type") == "text"
            and isinstance(first.get("text"), str)
            and first["text"].strip() == CLAUDE_CODE_SYSTEM_PROMPT
        ):
            return system
        return [identity, *system]

    # Unknown shape: prepend identity and keep the original payload alongside it.
    return [identity, system]


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build JSON for the official Anthropic ``POST …/messages`` endpoint."""
    body = build_base_native_anthropic_request_body(
        request_data,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=thinking_enabled,
        default_thinking_budget=DEFAULT_THINKING_BUDGET_TOKENS,
    )
    body["system"] = _ensure_claude_code_system(body.get("system"))
    body["stream"] = True

    logger.debug(
        "ANTHROPIC_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
