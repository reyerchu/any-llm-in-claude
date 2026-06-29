"""Beta-gated body fields must carry their authorizing ``anthropic-beta`` flag.

Regression: selecting Claude Code's "1M context" mode sends a ``context_management``
field in the body. The proxy forwards the field (it is a passthrough request field)
but reconstructs request headers statically, dropping the client's
``context-management-2025-06-27`` beta. Anthropic then rejects the unauthorized
field with HTTP 400 "context_management: Extra inputs are not permitted".
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.anthropic.client import (
    _CONTEXT_MANAGEMENT_BETA,
    AnthropicProvider,
)
from providers.base import ProviderConfig
from providers.transports.anthropic_messages import AnthropicMessagesTransport
from providers.transports.anthropic_messages.transport import _merge_anthropic_betas


# --- the merge helper ---------------------------------------------------------
def test_merge_anthropic_betas_appends_and_dedupes() -> None:
    headers = {"anthropic-beta": "oauth-2025-04-20"}
    _merge_anthropic_betas(
        headers, ("context-management-2025-06-27", "oauth-2025-04-20")
    )
    assert headers["anthropic-beta"] == (
        "oauth-2025-04-20,context-management-2025-06-27"
    )


def test_merge_anthropic_betas_sets_header_when_absent() -> None:
    headers: dict[str, str] = {}
    _merge_anthropic_betas(headers, ("context-management-2025-06-27",))
    assert headers["anthropic-beta"] == "context-management-2025-06-27"


def test_merge_anthropic_betas_noop_when_empty() -> None:
    headers = {"anthropic-beta": "oauth-2025-04-20"}
    _merge_anthropic_betas(headers, ())
    assert headers == {"anthropic-beta": "oauth-2025-04-20"}


# --- the provider hook --------------------------------------------------------
def test_anthropic_provider_declares_context_management_beta() -> None:
    required = AnthropicProvider._body_required_betas
    dummy = object.__new__(AnthropicProvider)  # bypass network __init__
    assert required(dummy, {"context_management": {"edits": []}}) == (
        _CONTEXT_MANAGEMENT_BETA,
    )
    # present-but-empty still needs the beta (the field itself is gated)
    assert required(dummy, {"context_management": {}}) == (_CONTEXT_MANAGEMENT_BETA,)
    # absent -> no extra beta
    assert required(dummy, {"messages": []}) == ()
    assert required(dummy, None) == ()


# --- transport wiring: betas reach the outgoing request -----------------------
class _CMProvider(AnthropicMessagesTransport):
    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="TEST_CM",
            default_base_url="https://example.test/v1",
        )

    def _request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _body_required_betas(self, body):
        if isinstance(body, dict) and body.get("context_management") is not None:
            return ("context-management-2025-06-27",)
        return ()


@pytest.fixture(autouse=True)
def _mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch(
        "providers.transports.anthropic_messages.transport.GlobalRateLimiter"
    ) as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield


@pytest.mark.asyncio
async def test_send_stream_request_injects_body_required_betas() -> None:
    config = ProviderConfig(api_key="test-key")
    with patch("httpx.AsyncClient"):
        provider = _CMProvider(config)

    captured: dict[str, dict] = {}

    def _build_request(method, url, *, json, headers):
        captured["headers"] = headers
        return MagicMock()

    provider._client.build_request = MagicMock(side_effect=_build_request)
    provider._client.send = AsyncMock(return_value=MagicMock())

    await provider._send_stream_request({"context_management": {"edits": []}})
    assert (
        "context-management-2025-06-27" in captured["headers"]["anthropic-beta"]
    )

    captured.clear()
    await provider._send_stream_request({"messages": []})
    assert "anthropic-beta" not in captured["headers"]
