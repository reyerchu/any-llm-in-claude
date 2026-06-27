"""Claude Code browser-login OAuth store: self-refreshing account token (no API key)."""

from __future__ import annotations

import json
import time

import pytest

from providers.anthropic.credentials import ClaudeOAuthCredentialStore
from providers.exceptions import AuthenticationError


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[dict] = []

    async def post(self, url: str, *, json: dict, headers: dict) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self._payload)


def _write(path, **oauth_over) -> None:
    doc = {
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat-old",
            "refreshToken": "sk-ant-ort-old",
            "expiresAt": int((time.time() + 3600) * 1000),  # ms, 1h ahead
            "subscriptionType": "max",
            **oauth_over,
        }
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.mark.asyncio
async def test_returns_cached_token_when_not_expired(tmp_path) -> None:
    cred = tmp_path / ".credentials.json"
    _write(cred)
    client = _FakeClient({"access_token": "sk-ant-oat-new"})
    store = ClaudeOAuthCredentialStore(str(cred), client=client)

    assert await store.access_token() == "sk-ant-oat-old"
    assert client.calls == []


@pytest.mark.asyncio
async def test_refreshes_with_user_agent_and_persists_ms_expiry(tmp_path) -> None:
    cred = tmp_path / ".credentials.json"
    _write(cred, expiresAt=int((time.time() - 10) * 1000))  # expired
    client = _FakeClient(
        {
            "access_token": "sk-ant-oat-new",
            "refresh_token": "sk-ant-ort-rotated",
            "expires_in": 28800,
        }
    )
    store = ClaudeOAuthCredentialStore(str(cred), client=client)

    assert await store.access_token() == "sk-ant-oat-new"
    call = client.calls[0]
    assert call["json"]["grant_type"] == "refresh_token"
    assert call["json"]["refresh_token"] == "sk-ant-ort-old"
    assert "claude-cli" in call["headers"]["User-Agent"]  # Cloudflare needs a UA

    on_disk = json.loads(cred.read_text(encoding="utf-8"))["claudeAiOauth"]
    assert on_disk["accessToken"] == "sk-ant-oat-new"
    assert on_disk["refreshToken"] == "sk-ant-ort-rotated"
    # expiresAt stays in milliseconds and is ~8h ahead.
    assert on_disk["expiresAt"] > time.time() * 1000 + 27000 * 1000


@pytest.mark.asyncio
async def test_missing_login_file_raises(tmp_path) -> None:
    store = ClaudeOAuthCredentialStore(
        str(tmp_path / "nope.json"), client=_FakeClient({})
    )
    with pytest.raises(AuthenticationError, match="Claude login credentials not found"):
        await store.access_token()
