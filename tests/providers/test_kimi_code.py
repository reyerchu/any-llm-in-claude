"""Kimi Code subscription provider: self-refreshing OAuth account token (no API key)."""

from __future__ import annotations

import json
import time

import pytest

from config.provider_catalog import (
    ANTHROPIC_OAUTH_BETA,
    KIMI_CODE_DEFAULT_BASE,
    PROVIDER_CATALOG,
)
from providers.exceptions import AuthenticationError
from providers.kimi_code.credentials import KimiCodeCredentialStore


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Minimal async httpx stand-in capturing the refresh POST."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[dict] = []

    async def post(self, url: str, *, data: dict, headers: dict) -> _FakeResponse:
        self.calls.append({"url": url, "data": data, "headers": headers})
        return _FakeResponse(self._payload)


def test_kimi_code_descriptor_is_oauth_no_api_key() -> None:
    desc = PROVIDER_CATALOG["kimi_code"]
    assert desc.auth_scheme == "oauth"
    assert desc.oauth_beta == ANTHROPIC_OAUTH_BETA
    assert desc.credential_env is None  # no API key — file-based account token
    assert desc.default_base_url == KIMI_CODE_DEFAULT_BASE


def _write(path, **over) -> None:
    data = {
        "access_token": "AT-old",
        "refresh_token": "RT-old",
        "token_type": "Bearer",
        "expires_at": int(time.time()) + 3600,
        **over,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.asyncio
async def test_returns_cached_token_when_not_expired(tmp_path) -> None:
    cred = tmp_path / "kimi-code.json"
    _write(cred)  # expires in 1h → no refresh
    client = _FakeClient({"access_token": "AT-new"})
    store = KimiCodeCredentialStore(str(cred), client=client)

    assert await store.access_token() == "AT-old"
    assert client.calls == []  # never hit the network


@pytest.mark.asyncio
async def test_refreshes_and_persists_rotated_tokens(tmp_path) -> None:
    cred = tmp_path / "kimi-code.json"
    _write(cred, expires_at=int(time.time()) - 10)  # already expired
    client = _FakeClient(
        {
            "access_token": "AT-new",
            "refresh_token": "RT-rotated",
            "token_type": "Bearer",
            "expires_in": 900,
        }
    )
    store = KimiCodeCredentialStore(str(cred), client=client)

    assert await store.access_token() == "AT-new"
    # Sent an x-www-form-urlencoded refresh_token grant.
    assert client.calls[0]["data"]["grant_type"] == "refresh_token"
    assert client.calls[0]["data"]["refresh_token"] == "RT-old"
    # Rotated tokens + recomputed expiry written back to disk.
    on_disk = json.loads(cred.read_text(encoding="utf-8"))
    assert on_disk["access_token"] == "AT-new"
    assert on_disk["refresh_token"] == "RT-rotated"
    assert on_disk["expires_at"] > time.time()


@pytest.mark.asyncio
async def test_missing_credentials_file_raises(tmp_path) -> None:
    store = KimiCodeCredentialStore(str(tmp_path / "nope.json"), client=_FakeClient({}))
    with pytest.raises(AuthenticationError, match="Kimi Code credentials not found"):
        await store.access_token()
