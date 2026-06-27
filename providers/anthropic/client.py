"""Anthropic account provider — official Messages API via a ``setup-token``.

Authenticates against ``https://api.anthropic.com/v1`` with a Claude *account*
OAuth access token (``claude setup-token``; Pro/Max subscription) instead of a paid
console API key. The shared transport injects ``Authorization: Bearer`` plus the
``anthropic-beta`` OAuth flag (see ``ProviderConfig.auth_scheme == "oauth"``).
"""

from __future__ import annotations

from typing import Any

from config.provider_catalog import ANTHROPIC_DEFAULT_BASE
from providers.base import ProviderConfig
from providers.transports.anthropic_messages import AnthropicMessagesTransport

from .request import build_request_body

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(AnthropicMessagesTransport):
    """Official Anthropic Messages provider driven by an account setup-token."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ANTHROPIC",
            default_base_url=ANTHROPIC_DEFAULT_BASE,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _request_headers(self) -> dict[str, str]:
        # ``Authorization`` + ``anthropic-beta`` are added by the transport's OAuth
        # header injection; we only supply the static request headers here.
        return {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {"anthropic-version": _ANTHROPIC_VERSION}
