"""Neutral provider catalog: IDs, credentials, defaults, proxy and capability metadata.

Adapter factories live in :mod:`providers.registry`; this module stays free of
provider implementation imports (see contract tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransportType = Literal["openai_chat", "anthropic_messages"]
# How a provider authenticates upstream. ``api_key`` is the default (provider API
# key in the native ``x-api-key`` / ``Authorization: Bearer`` slot). ``oauth`` uses
# an *account* access token (e.g. ``claude setup-token`` for Anthropic) sent as a
# ``Authorization: Bearer`` token, optionally with an ``anthropic-beta`` OAuth flag.
AuthScheme = Literal["api_key", "oauth"]

# OAuth beta flag Anthropic requires when authenticating with a subscription
# (``claude setup-token`` / ``claude /login``) access token instead of an API key.
ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"
# Optional beta enabling the 1M-token context window on the Anthropic provider.
# Off by default; toggled per instance via the ``ANTHROPIC_CONTEXT_1M`` setting.
ANTHROPIC_CONTEXT_1M_BETA = "context-1m-2025-08-07"
# Anthropic OAuth refresh endpoint + public Claude Code client id (used to refresh
# the access token the ``claude`` CLI stores after a browser ``/login``).
ANTHROPIC_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
ANTHROPIC_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
# Default location of the Claude Code browser-login credentials JSON.
CLAUDE_CODE_DEFAULT_CREDENTIALS_PATH = "~/.claude/.credentials.json"

# Default upstream base URLs (also re-exported via :mod:`providers.defaults`)
# Official Anthropic Messages API — used with a Claude account ``setup-token`` OAuth
# access token (Pro/Max subscription) instead of a paid ``x-api-key``.
ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com/v1"
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
# Moonshot Kimi Anthropic-compatible Messages API (POST …/messages).
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/anthropic/v1"
# Kimi Code (`kimi` CLI subscription) Anthropic-compatible coding endpoint. Auth is a
# self-refreshing OAuth account token stored by the `kimi` CLI — NOT a console API key.
KIMI_CODE_DEFAULT_BASE = "https://api.kimi.com/coding/v1"
KIMI_CODE_AUTH_TOKEN_URL = "https://auth.kimi.com/api/oauth/token"
KIMI_CODE_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
# Default location where the `kimi` CLI stores its OAuth credentials JSON.
KIMI_CODE_DEFAULT_CREDENTIALS_PATH = "~/.kimi-code/credentials/kimi-code.json"
WAFER_DEFAULT_BASE = "https://pass.wafer.ai/v1"
# DeepSeek Anthropic-compatible Messages API (not OpenAI ``/v1`` chat completions).
DEEPSEEK_ANTHROPIC_DEFAULT_BASE = "https://api.deepseek.com/anthropic"
# Historical export name: DeepSeek upstream is the native Anthropic path above.
DEEPSEEK_DEFAULT_BASE = DEEPSEEK_ANTHROPIC_DEFAULT_BASE
FIREWORKS_DEFAULT_BASE = "https://api.fireworks.ai/inference/v1"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
MISTRAL_DEFAULT_BASE = "https://api.mistral.ai/v1"
# Codestral IDE/personal endpoint (distinct from La Plateforme ``api.mistral.ai`` keys).
CODESTRAL_DEFAULT_BASE = "https://codestral.mistral.ai/v1"
LMSTUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
LLAMACPP_DEFAULT_BASE = "http://localhost:8080/v1"
OLLAMA_DEFAULT_BASE = "http://localhost:11434"
OPENCODE_DEFAULT_BASE = "https://opencode.ai/zen/v1"
OPENCODE_GO_DEFAULT_BASE = "https://opencode.ai/zen/go/v1"
# Z.ai Anthropic-compatible Messages API (not OpenAI Coding Plan chat completions).
ZAI_DEFAULT_BASE = "https://api.z.ai/api/anthropic/v1"
# Google AI Studio Gemini API OpenAI-compat layer (not Vertex AI).
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_DEFAULT_BASE = "https://api.groq.com/openai/v1"
CEREBRAS_DEFAULT_BASE = "https://api.cerebras.ai/v1"
# FreeLLMAPI (https://github.com/tashfeenahmed/freellmapi) — OpenAI-compatible
# unified proxy that rotates the free tiers of 16 LLM providers behind one
# /v1 endpoint. Local instance defaults to http://localhost:3001/v1.
FREELLM_DEFAULT_BASE = "http://localhost:3001/v1"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Metadata for building :class:`~providers.base.ProviderConfig` and factory wiring."""

    provider_id: str
    transport_type: TransportType
    capabilities: tuple[str, ...]
    credential_env: str | None = None
    credential_url: str | None = None
    credential_attr: str | None = None
    static_credential: str | None = None
    default_base_url: str | None = None
    base_url_attr: str | None = None
    proxy_attr: str | None = None
    # Account-token / OAuth (use an LLM *account* token instead of an API key).
    # ``oauth_env`` / ``oauth_attr`` name the env var and Settings attribute holding
    # the account access token. When that token is set it takes precedence over the
    # API key and the request is authenticated as ``auth_scheme="oauth"`` (Bearer).
    oauth_env: str | None = None
    oauth_attr: str | None = None
    # When the descriptor's *primary* credential is itself an OAuth account token
    # (no API-key alternative, e.g. the ``anthropic`` setup-token provider).
    auth_scheme: AuthScheme = "api_key"
    # ``anthropic-beta`` flag to add when authenticating via an OAuth account token.
    oauth_beta: str | None = None
    # When True, the provider can opt into the 1M-token context window beta
    # (gated by the ``anthropic_context_1m`` setting).
    supports_context_1m: bool = False
    # When True, an empty configured credential is allowed because the provider
    # sources its token elsewhere at runtime (e.g. reads a browser-login file).
    credential_optional: bool = False


PROVIDER_CATALOG: dict[str, ProviderDescriptor] = {
    "anthropic": ProviderDescriptor(
        provider_id="anthropic",
        transport_type="anthropic_messages",
        # Credential is the Claude *account* OAuth token the ``claude`` CLI stores
        # after a browser ``/login`` (Pro/Max subscription) — auto-refreshed, NOT a
        # paid console API key and NOT an admin-UI field. ``ANTHROPIC_OAUTH_TOKEN``
        # (a ``claude setup-token``) stays supported as an optional .env override via
        # ``credential_attr``; with no env var here the credential is optional.
        credential_attr="anthropic_oauth_token",
        credential_optional=True,
        default_base_url=ANTHROPIC_DEFAULT_BASE,
        proxy_attr="anthropic_proxy",
        auth_scheme="oauth",
        oauth_beta=ANTHROPIC_OAUTH_BETA,
        supports_context_1m=True,
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "oauth",
            "rate_limit",
        ),
    ),
    "nvidia_nim": ProviderDescriptor(
        provider_id="nvidia_nim",
        transport_type="openai_chat",
        credential_env="NVIDIA_NIM_API_KEY",
        credential_url="https://build.nvidia.com/settings/api-keys",
        credential_attr="nvidia_nim_api_key",
        default_base_url=NVIDIA_NIM_DEFAULT_BASE,
        proxy_attr="nvidia_nim_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "freellm": ProviderDescriptor(
        provider_id="freellm",
        transport_type="openai_chat",
        credential_env="FREELLM_API_KEY",
        credential_url="http://localhost:3001/keys",
        credential_attr="freellm_api_key",
        default_base_url=FREELLM_DEFAULT_BASE,
        proxy_attr="freellm_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "open_router": ProviderDescriptor(
        provider_id="open_router",
        transport_type="anthropic_messages",
        credential_env="OPENROUTER_API_KEY",
        credential_url="https://openrouter.ai/keys",
        credential_attr="open_router_api_key",
        oauth_env="OPENROUTER_OAUTH_TOKEN",
        oauth_attr="open_router_oauth_token",
        default_base_url=OPENROUTER_DEFAULT_BASE,
        proxy_attr="open_router_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
    "gemini": ProviderDescriptor(
        provider_id="gemini",
        transport_type="openai_chat",
        credential_env="GEMINI_API_KEY",
        credential_url="https://aistudio.google.com/apikey",
        credential_attr="gemini_api_key",
        default_base_url=GEMINI_DEFAULT_BASE,
        proxy_attr="gemini_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "deepseek": ProviderDescriptor(
        provider_id="deepseek",
        transport_type="anthropic_messages",
        credential_env="DEEPSEEK_API_KEY",
        credential_url="https://platform.deepseek.com/api_keys",
        credential_attr="deepseek_api_key",
        oauth_env="DEEPSEEK_OAUTH_TOKEN",
        oauth_attr="deepseek_oauth_token",
        default_base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
    "mistral": ProviderDescriptor(
        provider_id="mistral",
        transport_type="openai_chat",
        credential_env="MISTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="mistral_api_key",
        default_base_url=MISTRAL_DEFAULT_BASE,
        proxy_attr="mistral_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "mistral_codestral": ProviderDescriptor(
        provider_id="mistral_codestral",
        transport_type="openai_chat",
        credential_env="CODESTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="codestral_api_key",
        default_base_url=CODESTRAL_DEFAULT_BASE,
        proxy_attr="codestral_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "opencode": ProviderDescriptor(
        provider_id="opencode",
        transport_type="openai_chat",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_DEFAULT_BASE,
        proxy_attr="opencode_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "opencode_go": ProviderDescriptor(
        provider_id="opencode_go",
        transport_type="openai_chat",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_GO_DEFAULT_BASE,
        proxy_attr="opencode_go_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "wafer": ProviderDescriptor(
        provider_id="wafer",
        transport_type="anthropic_messages",
        credential_env="WAFER_API_KEY",
        credential_url="https://www.wafer.ai/pass",
        credential_attr="wafer_api_key",
        oauth_env="WAFER_OAUTH_TOKEN",
        oauth_attr="wafer_oauth_token",
        default_base_url=WAFER_DEFAULT_BASE,
        proxy_attr="wafer_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
    "kimi": ProviderDescriptor(
        provider_id="kimi",
        transport_type="anthropic_messages",
        credential_env="KIMI_API_KEY",
        credential_url="https://platform.moonshot.cn/console/api-keys",
        credential_attr="kimi_api_key",
        oauth_env="KIMI_OAUTH_TOKEN",
        oauth_attr="kimi_oauth_token",
        default_base_url=KIMI_DEFAULT_BASE,
        proxy_attr="kimi_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
        ),
    ),
    "kimi_code": ProviderDescriptor(
        provider_id="kimi_code",
        transport_type="anthropic_messages",
        # No API key: credential is a self-refreshing OAuth account token the `kimi`
        # CLI stores on disk. The provider reads + refreshes it; nothing to configure.
        static_credential="kimi-code-oauth",
        credential_url="run `kimi` and log in (Kimi Code subscription)",
        default_base_url=KIMI_CODE_DEFAULT_BASE,
        auth_scheme="oauth",
        oauth_beta=ANTHROPIC_OAUTH_BETA,
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "oauth",
        ),
    ),
    "cerebras": ProviderDescriptor(
        provider_id="cerebras",
        transport_type="openai_chat",
        credential_env="CEREBRAS_API_KEY",
        credential_url="https://cloud.cerebras.ai",
        credential_attr="cerebras_api_key",
        default_base_url=CEREBRAS_DEFAULT_BASE,
        proxy_attr="cerebras_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "groq": ProviderDescriptor(
        provider_id="groq",
        transport_type="openai_chat",
        credential_env="GROQ_API_KEY",
        credential_url="https://console.groq.com/keys",
        credential_attr="groq_api_key",
        default_base_url=GROQ_DEFAULT_BASE,
        proxy_attr="groq_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "fireworks": ProviderDescriptor(
        provider_id="fireworks",
        transport_type="anthropic_messages",
        credential_env="FIREWORKS_API_KEY",
        credential_url="https://fireworks.ai/account/api-keys",
        credential_attr="fireworks_api_key",
        oauth_env="FIREWORKS_OAUTH_TOKEN",
        oauth_attr="fireworks_oauth_token",
        default_base_url=FIREWORKS_DEFAULT_BASE,
        proxy_attr="fireworks_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "rate_limit",
        ),
    ),
    "zai": ProviderDescriptor(
        provider_id="zai",
        transport_type="anthropic_messages",
        credential_env="ZAI_API_KEY",
        credential_attr="zai_api_key",
        oauth_env="ZAI_OAUTH_TOKEN",
        oauth_attr="zai_oauth_token",
        default_base_url=ZAI_DEFAULT_BASE,
        proxy_attr="zai_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "rate_limit",
        ),
    ),
    "lmstudio": ProviderDescriptor(
        provider_id="lmstudio",
        transport_type="anthropic_messages",
        static_credential="lm-studio",
        default_base_url=LMSTUDIO_DEFAULT_BASE,
        base_url_attr="lm_studio_base_url",
        proxy_attr="lmstudio_proxy",
        capabilities=("chat", "streaming", "tools", "native_anthropic", "local"),
    ),
    "llamacpp": ProviderDescriptor(
        provider_id="llamacpp",
        transport_type="anthropic_messages",
        static_credential="llamacpp",
        default_base_url=LLAMACPP_DEFAULT_BASE,
        base_url_attr="llamacpp_base_url",
        proxy_attr="llamacpp_proxy",
        capabilities=("chat", "streaming", "tools", "native_anthropic", "local"),
    ),
    "ollama": ProviderDescriptor(
        provider_id="ollama",
        transport_type="anthropic_messages",
        static_credential="ollama",
        default_base_url=OLLAMA_DEFAULT_BASE,
        base_url_attr="ollama_base_url",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "local",
        ),
    ),
}

# Key order:
# NVIDIA NIM first (README default), DeepSeek fourth, Wafer ninth / Kimi tenth; then cerebras /
# groq / fireworks overlap; remainder and locals last per project plan (
# github.com/cheahjs/free-llm-api-resources Free Providers TOC as rough guide beyond fixed slots).
# ``SUPPORTED_PROVIDER_IDS`` inherits this insertion order for UI and error-message listing.
SUPPORTED_PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_CATALOG.keys())

if len(set(SUPPORTED_PROVIDER_IDS)) != len(SUPPORTED_PROVIDER_IDS):
    raise AssertionError("Duplicate provider ids in PROVIDER_CATALOG key order")
