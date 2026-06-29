"""Anthropic account provider — official Messages API on a Claude subscription.

Authenticates against ``https://api.anthropic.com/v1`` with a Claude *account* OAuth
access token instead of a paid console API key. Two sources, in priority order:

1. A static ``ANTHROPIC_OAUTH_TOKEN`` (a ``claude setup-token``), if set; or
2. the browser-login token the ``claude`` CLI stores at ``~/.claude/.credentials.json``
   after ``claude /login`` — read and **auto-refreshed** per request.

The shared transport injects ``Authorization: Bearer`` plus the ``anthropic-beta``
OAuth flag (see ``ProviderConfig.auth_scheme == "oauth"``).
"""

from __future__ import annotations

from typing import Any

import httpx

from config.provider_catalog import ANTHROPIC_DEFAULT_BASE
from providers.base import ProviderConfig
from providers.transports.anthropic_messages import AnthropicMessagesTransport

from .credentials import ClaudeOAuthCredentialStore
from .request import build_request_body

_ANTHROPIC_VERSION = "2023-06-01"

# Beta required to authorize the ``context_management`` request field. Claude Code
# sends it alongside the field; the proxy must re-add it (the field is forwarded in
# the body but the client's anthropic-beta header is not).
_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"


class AnthropicProvider(AnthropicMessagesTransport):
    """Official Anthropic Messages provider on a Claude subscription OAuth token."""

    def __init__(self, config: ProviderConfig, *, credentials_path: str | None = None):
        super().__init__(
            config,
            provider_name="ANTHROPIC",
            default_base_url=ANTHROPIC_DEFAULT_BASE,
        )
        # A non-empty configured token (setup-token) is used verbatim; otherwise we
        # fall back to the auto-refreshing browser-login credentials file.
        self._static_token = (config.api_key or "").strip()
        self._store: ClaudeOAuthCredentialStore | None = None
        if not self._static_token and credentials_path:
            self._store = ClaudeOAuthCredentialStore(
                credentials_path, client=self._client
            )

    async def _refresh_credential(self) -> None:
        """Point ``self._api_key`` at a valid access token (refreshing if file-based)."""
        if self._store is not None:
            self._api_key = await self._store.access_token()

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _body_required_betas(self, body: Any) -> tuple[str, ...]:
        if isinstance(body, dict) and body.get("context_management") is not None:
            return (_CONTEXT_MANAGEMENT_BETA,)
        return ()

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

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        await self._refresh_credential()
        return await super()._send_stream_request(body)

    async def _send_model_list_request(self) -> httpx.Response:
        await self._refresh_credential()
        return await self._client.get(
            "/models",
            headers=self._apply_oauth_headers(self._model_list_headers()),
        )
