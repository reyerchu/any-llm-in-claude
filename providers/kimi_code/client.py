"""Kimi Code provider — Anthropic-compatible coding endpoint via a subscription token.

Authenticates against ``https://api.kimi.com/coding/v1`` with the self-refreshing
OAuth *account* token the ``kimi`` CLI stores on disk (no API key). The shared
transport's OAuth path adds ``Authorization: Bearer`` + ``anthropic-beta``; this
provider just keeps ``self._api_key`` pointed at a freshly-refreshed access token.
"""

from __future__ import annotations

from typing import Any

import httpx

from config.provider_catalog import KIMI_CODE_DEFAULT_BASE
from providers.base import ProviderConfig
from providers.transports.anthropic_messages import AnthropicMessagesTransport

from .credentials import KimiCodeCredentialStore
from .request import build_request_body

_ANTHROPIC_VERSION = "2023-06-01"


class KimiCodeProvider(AnthropicMessagesTransport):
    """Kimi Code subscription provider with an auto-refreshing OAuth token."""

    def __init__(self, config: ProviderConfig, *, credentials_path: str):
        super().__init__(
            config,
            provider_name="KIMI_CODE",
            default_base_url=KIMI_CODE_DEFAULT_BASE,
        )
        self._store = KimiCodeCredentialStore(credentials_path, client=self._client)

    async def _refresh_credential(self) -> None:
        """Point ``self._api_key`` at a valid (refreshed if needed) access token."""
        self._api_key = await self._store.access_token()

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _request_headers(self) -> dict[str, str]:
        # Authorization + anthropic-beta are added by the transport OAuth path.
        return {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _model_list_headers(self) -> dict[str, str]:
        return {"anthropic-version": _ANTHROPIC_VERSION}

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        await self._refresh_credential()
        return await super()._send_stream_request(body)

    async def _send_model_list_request(self) -> httpx.Response:
        await self._refresh_credential()
        return await self._client.get(
            "/models",
            headers=self._apply_oauth_headers(self._model_list_headers()),
        )
