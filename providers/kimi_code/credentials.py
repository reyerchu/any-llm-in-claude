"""Self-refreshing OAuth credential store for the Kimi Code subscription token.

The ``kimi`` CLI logs in once and stores a short-lived (~15 min) OAuth access token
plus a rotating refresh token in a JSON file. This store reads that file, refreshes
the access token when it is near expiry (writing the rotated tokens back so the CLI
and this proxy stay in sync), and hands the live access token to the provider.

No API key is involved — the credential is the user's Kimi Code *account* token.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from config.provider_catalog import (
    KIMI_CODE_AUTH_TOKEN_URL,
    KIMI_CODE_CLIENT_ID,
)
from providers.exceptions import AuthenticationError

# Refresh this many seconds before the stated expiry to avoid mid-request expiry.
_REFRESH_SKEW_SECONDS = 120


class KimiCodeCredentialStore:
    """Read, cache, and refresh the ``kimi`` CLI OAuth credentials JSON."""

    def __init__(self, credentials_path: str, *, client: httpx.AsyncClient):
        self._path = Path(os.path.expanduser(credentials_path))
        self._client = client
        self._lock = asyncio.Lock()
        self._cache: dict[str, Any] | None = None

    async def access_token(self) -> str:
        """Return a valid access token, refreshing (and persisting) when near expiry."""
        async with self._lock:
            data = self._cache or self._load()
            if self._needs_refresh(data):
                data = await self._refresh(data)
            self._cache = data
            token = data.get("access_token")
            if not isinstance(token, str) or not token.strip():
                raise AuthenticationError(self._missing_message())
            return token

    def _load(self) -> dict[str, Any]:
        if not self._path.is_file():
            raise AuthenticationError(self._missing_message())
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AuthenticationError(
                f"Kimi Code credentials at {self._path} are unreadable: {exc}. "
                "Run `kimi` and log in again."
            ) from exc
        if not isinstance(data, dict):
            raise AuthenticationError(self._missing_message())
        return data

    @staticmethod
    def _needs_refresh(data: dict[str, Any]) -> bool:
        expires_at = data.get("expires_at")
        if not isinstance(expires_at, (int, float)):
            return True
        return time.time() >= (expires_at - _REFRESH_SKEW_SECONDS)

    async def _refresh(self, data: dict[str, Any]) -> dict[str, Any]:
        refresh_token = data.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token.strip():
            raise AuthenticationError(self._missing_message())
        try:
            response = await self._client.post(
                KIMI_CODE_AUTH_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": KIMI_CODE_CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise AuthenticationError(
                "Kimi Code token refresh failed against "
                f"{KIMI_CODE_AUTH_TOKEN_URL}: {type(exc).__name__}. "
                "Run `kimi` and log in again."
            ) from exc

        merged = {**data, **payload}
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            merged["expires_at"] = int(time.time()) + int(expires_in)
        self._persist(merged)
        logger.info("Kimi Code OAuth token refreshed (expires_in={})", expires_in)
        return merged

    def _persist(self, data: dict[str, Any]) -> None:
        """Atomically write refreshed tokens back so the CLI sees the latest pair."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".kimi-code-", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp, self._path)
            os.chmod(self._path, 0o600)
        except OSError as exc:
            # A failed write-back is non-fatal: the in-memory token still works.
            logger.warning("Could not persist refreshed Kimi Code token: {}", exc)

    def _missing_message(self) -> str:
        return (
            f"Kimi Code credentials not found at {self._path}. "
            "Install the `kimi` CLI and run `kimi` to log in, or set "
            "KIMI_CODE_CREDENTIALS_PATH to your credentials JSON."
        )
