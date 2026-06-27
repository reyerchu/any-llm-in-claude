"""Self-refreshing OAuth store for the Claude Code browser-login credentials.

``claude /login`` (or ``claude setup-token``) stores a Claude *account* OAuth token
under ``claudeAiOauth`` in ``~/.claude/.credentials.json``: a short-lived access
token, a rotating refresh token, and ``expiresAt`` in **milliseconds**. This store
reads that file, refreshes the access token when near expiry (writing the rotated
pair back in the same schema so the CLI and proxy stay in sync), and hands the live
access token to the provider. No API key is involved.
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
    ANTHROPIC_OAUTH_CLIENT_ID,
    ANTHROPIC_OAUTH_TOKEN_URL,
)
from providers.exceptions import AuthenticationError

# Refresh this many seconds before the stated expiry to avoid mid-request expiry.
_REFRESH_SKEW_SECONDS = 120
# platform.claude.com sits behind Cloudflare and rejects the default httpx UA.
_REFRESH_USER_AGENT = "claude-cli/2.1.195 (external, cli)"
_OAUTH_KEY = "claudeAiOauth"


class ClaudeOAuthCredentialStore:
    """Read, cache, and refresh the ``claude`` CLI browser-login credentials JSON."""

    def __init__(self, credentials_path: str, *, client: httpx.AsyncClient):
        self._path = Path(os.path.expanduser(credentials_path))
        self._client = client
        self._lock = asyncio.Lock()
        self._cache: dict[str, Any] | None = None  # the full file document

    async def access_token(self) -> str:
        """Return a valid access token, refreshing (and persisting) when near expiry."""
        async with self._lock:
            doc = self._cache or self._load()
            oauth = doc.get(_OAUTH_KEY, {})
            if self._needs_refresh(oauth):
                doc = await self._refresh(doc)
                oauth = doc.get(_OAUTH_KEY, {})
            self._cache = doc
            token = oauth.get("accessToken")
            if not isinstance(token, str) or not token.strip():
                raise AuthenticationError(self._missing_message())
            return token

    def _load(self) -> dict[str, Any]:
        if not self._path.is_file():
            raise AuthenticationError(self._missing_message())
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AuthenticationError(
                f"Claude credentials at {self._path} are unreadable: {exc}. "
                "Run `claude /login` again."
            ) from exc
        if not isinstance(doc, dict) or not isinstance(doc.get(_OAUTH_KEY), dict):
            raise AuthenticationError(self._missing_message())
        return doc

    @staticmethod
    def _needs_refresh(oauth: dict[str, Any]) -> bool:
        expires_at_ms = oauth.get("expiresAt")
        if not isinstance(expires_at_ms, (int, float)):
            return True
        return time.time() >= (expires_at_ms / 1000.0 - _REFRESH_SKEW_SECONDS)

    async def _refresh(self, doc: dict[str, Any]) -> dict[str, Any]:
        oauth = doc.get(_OAUTH_KEY, {})
        refresh_token = oauth.get("refreshToken")
        if not isinstance(refresh_token, str) or not refresh_token.strip():
            raise AuthenticationError(self._missing_message())
        try:
            response = await self._client.post(
                ANTHROPIC_OAUTH_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": _REFRESH_USER_AGENT,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise AuthenticationError(
                "Claude OAuth token refresh failed against "
                f"{ANTHROPIC_OAUTH_TOKEN_URL}: {type(exc).__name__}. "
                "Run `claude /login` again."
            ) from exc

        new_oauth = dict(oauth)
        new_oauth["accessToken"] = payload.get("access_token", oauth.get("accessToken"))
        if payload.get("refresh_token"):
            new_oauth["refreshToken"] = payload["refresh_token"]
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            new_oauth["expiresAt"] = int(time.time() * 1000) + int(expires_in) * 1000
        merged = {**doc, _OAUTH_KEY: new_oauth}
        self._persist(merged)
        logger.info("Claude OAuth token refreshed (expires_in={})", expires_in)
        return merged

    def _persist(self, doc: dict[str, Any]) -> None:
        """Atomically write refreshed tokens back so the CLI sees the latest pair."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".credentials-", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(doc, handle)
            os.replace(tmp, self._path)
            os.chmod(self._path, 0o600)
        except OSError as exc:
            logger.warning("Could not persist refreshed Claude token: {}", exc)

    def _missing_message(self) -> str:
        return (
            f"Claude login credentials not found at {self._path}. "
            "Run `claude /login` (Pro/Max subscription), set ANTHROPIC_OAUTH_TOKEN "
            "to a `claude setup-token`, or set CLAUDE_CREDENTIALS_PATH."
        )
