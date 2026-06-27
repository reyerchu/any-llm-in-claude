"""Account-token / OAuth auth: use an LLM account token instead of an API key.

Covers the headline ``any-llm-in-claude`` feature:

* a new ``anthropic`` provider authenticated by a ``claude setup-token`` OAuth
  access token (Bearer + ``anthropic-beta`` OAuth flag, never ``x-api-key``), and
* a generic per-provider account-token override (e.g. ``KIMI_OAUTH_TOKEN``) that
  takes precedence over the API key and switches the request to Bearer OAuth.
"""

from __future__ import annotations

import pytest

from config.provider_catalog import ANTHROPIC_OAUTH_BETA, PROVIDER_CATALOG
from config.settings import Settings
from providers.anthropic.request import (
    CLAUDE_CODE_SYSTEM_PROMPT,
    build_request_body,
)
from providers.registry import build_provider_config, create_provider


def _settings(**env: str) -> Settings:
    """Build Settings from env-alias kwargs (e.g. ANTHROPIC_OAUTH_TOKEN=...)."""
    return Settings(_env_file=None, **env)


def test_anthropic_provider_in_catalog_uses_oauth() -> None:
    desc = PROVIDER_CATALOG["anthropic"]
    assert desc.auth_scheme == "oauth"
    assert desc.oauth_beta == ANTHROPIC_OAUTH_BETA
    # No credential_env / admin field: token comes from `claude /login` at runtime.
    assert desc.credential_env is None
    assert desc.credential_optional is True
    assert desc.credential_attr == "anthropic_oauth_token"  # optional .env override
    assert desc.default_base_url == "https://api.anthropic.com/v1"


def test_anthropic_oauth_headers_use_bearer_not_api_key() -> None:
    provider = create_provider(
        "anthropic", _settings(ANTHROPIC_OAUTH_TOKEN="sk-ant-oat01-TEST")
    )
    headers = provider._apply_oauth_headers(provider._request_headers())
    assert headers["Authorization"] == "Bearer sk-ant-oat01-TEST"
    assert headers["anthropic-beta"] == ANTHROPIC_OAUTH_BETA
    assert "x-api-key" not in headers


def test_anthropic_request_injects_claude_code_identity() -> None:
    class _Req:
        def model_dump(self, exclude_none: bool = True) -> dict:
            return {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
            }

    body = build_request_body(_Req(), thinking_enabled=False)
    assert body["system"][0]["text"] == CLAUDE_CODE_SYSTEM_PROMPT


def test_account_token_overrides_api_key_and_switches_to_oauth() -> None:
    settings = _settings(KIMI_API_KEY="key-AAA", KIMI_OAUTH_TOKEN="acct-BBB")
    config = build_provider_config(PROVIDER_CATALOG["kimi"], settings)
    assert config.api_key == "acct-BBB"  # account token wins
    assert config.auth_scheme == "oauth"

    provider = create_provider("kimi", settings)
    headers = provider._apply_oauth_headers(provider._request_headers())
    assert headers["Authorization"] == "Bearer acct-BBB"
    assert "x-api-key" not in headers


def test_api_key_only_keeps_api_key_scheme() -> None:
    settings = _settings(KIMI_API_KEY="key-AAA")
    config = build_provider_config(PROVIDER_CATALOG["kimi"], settings)
    assert config.api_key == "key-AAA"
    assert config.auth_scheme == "api_key"
    assert config.oauth_beta is None


def test_anthropic_static_token_overrides_login_file() -> None:
    # A configured setup-token is used verbatim (no browser-login file read).
    provider = create_provider(
        "anthropic", _settings(ANTHROPIC_OAUTH_TOKEN="sk-ant-oat01-STATIC")
    )
    assert provider._static_token == "sk-ant-oat01-STATIC"
    assert provider._store is None


def test_anthropic_without_token_uses_login_file_store() -> None:
    # No setup-token configured -> provider sources the auto-refreshing login file.
    provider = create_provider("anthropic", _settings())
    assert provider._static_token == ""
    assert provider._store is not None  # reads ~/.claude/.credentials.json


@pytest.mark.asyncio
async def test_missing_login_file_raises_helpful_error(tmp_path) -> None:
    from providers.exceptions import AuthenticationError

    provider = create_provider(
        "anthropic",
        _settings(CLAUDE_CREDENTIALS_PATH=str(tmp_path / "nope.json")),
    )
    with pytest.raises(AuthenticationError, match="Claude login credentials not found"):
        await provider._refresh_credential()
    await provider.cleanup()
