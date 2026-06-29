"""FreeLLMAPI provider implementation.

A thin OpenAI-compatible wrapper around a freellmapi instance. Reuses the
default ``OpenAIChatTransport`` request building — no per-provider quirks
(neither NIM-style tool-aliases nor Kimi-style oauth-beta), so the inherited
streaming / rate-limit paths are used as-is.
"""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import FREELLM_DEFAULT_BASE
from providers.transports.openai_chat import OpenAIChatTransport

from .request import build_request_body


class FreeLlmProvider(OpenAIChatTransport):
    """FreeLLMAPI proxy at ``http://localhost:3001/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="FREELLM",
            base_url=config.base_url or FREELLM_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
