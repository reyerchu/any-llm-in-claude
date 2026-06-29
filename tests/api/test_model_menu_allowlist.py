"""The /model menu can be restricted to an allowlist of providers.

``MODEL_MENU_PROVIDERS`` filters which providers' *discovered* models appear in
the model list; the routed MODEL/MODEL_* targets and the built-in Claude models
are always listed.
"""

from __future__ import annotations

from api.model_catalog import build_models_list_response
from config.settings import Settings
from providers.model_listing import ProviderModelInfo


class _FakeRegistry:
    def __init__(self, infos):
        self._infos = tuple(infos)

    def cached_model_supports_thinking(self, provider_id, model_id):
        return None

    def cached_prefixed_model_infos(self):
        return self._infos


def test_allowlist_blank_is_none(monkeypatch):
    monkeypatch.setenv("MODEL_MENU_PROVIDERS", "")
    assert Settings().model_menu_provider_allowlist() is None


def test_allowlist_splits_and_strips(monkeypatch):
    monkeypatch.setenv("MODEL_MENU_PROVIDERS", " anthropic , freellm ,kimi_code ")
    assert Settings().model_menu_provider_allowlist() == frozenset(
        {"anthropic", "freellm", "kimi_code"}
    )


def test_menu_filters_discovered_providers_keeps_routes_and_claude(monkeypatch):
    monkeypatch.setenv("MODEL", "kimi_code/kimi-for-coding")
    monkeypatch.setenv("MODEL_OPUS", "anthropic/claude-opus-4-8")
    monkeypatch.setenv("MODEL_MENU_PROVIDERS", "anthropic,freellm,kimi_code")
    registry = _FakeRegistry(
        [
            ProviderModelInfo("freellm/auto", supports_thinking=True),
            ProviderModelInfo("nvidia_nim/nemotron", supports_thinking=False),
            ProviderModelInfo("open_router/some-model", supports_thinking=False),
            ProviderModelInfo("kimi_code/kimi-for-coding", supports_thinking=True),
        ]
    )
    resp = build_models_list_response(Settings(), registry)
    names = [m.display_name for m in resp.data]
    ids = {m.id for m in resp.data}

    # allowed discovered provider kept; disallowed ones dropped
    assert any(n.startswith("freellm/auto") for n in names)
    assert not any("nvidia_nim" in n for n in names)
    assert not any("open_router" in n for n in names)
    # routed targets always present (even though anthropic discovery is empty)
    assert any(n.startswith("anthropic/claude-opus-4-8") for n in names)
    # built-in Claude models always present
    assert "claude-opus-4-20250514" in ids


def test_menu_no_allowlist_lists_all_discovered(monkeypatch):
    monkeypatch.setenv("MODEL", "kimi_code/kimi-for-coding")
    monkeypatch.setenv("MODEL_MENU_PROVIDERS", "")
    registry = _FakeRegistry(
        [ProviderModelInfo("nvidia_nim/nemotron", supports_thinking=False)]
    )
    resp = build_models_list_response(Settings(), registry)
    assert any("nvidia_nim" in m.display_name for m in resp.data)


def test_discovery_skips_disallowed_providers(monkeypatch):
    from providers.registry import _model_list_provider_ids_for_settings

    monkeypatch.setenv("MODEL", "anthropic/claude-opus-4-8")
    monkeypatch.setenv("MODEL_MENU_PROVIDERS", "anthropic,freellm,kimi_code")
    ids = _model_list_provider_ids_for_settings(Settings())
    assert "nvidia_nim" not in ids
    assert "open_router" not in ids
