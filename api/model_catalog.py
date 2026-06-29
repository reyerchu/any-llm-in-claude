"""Model-list response construction for Claude-compatible clients."""

from __future__ import annotations

from config.settings import Settings
from providers.registry import ProviderRegistry

from .gateway_model_ids import gateway_model_id, no_thinking_gateway_model_id
from .models.responses import ModelResponse, ModelsListResponse

DISCOVERED_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


def build_models_list_response(
    settings: Settings, provider_registry: ProviderRegistry | None
) -> ModelsListResponse:
    """Return configured, cached, and compatibility model ids.

    With ``MODEL_MENU_PROVIDERS`` set, only the listed providers' routed and
    discovered models are surfaced, the built-in Claude models lead the menu,
    and provider entries follow in the configured order. With it unset, the
    historical order is preserved (routed + discovered, then Claude last) and no
    provider is filtered out.
    """
    models: list[ModelResponse] = []
    seen: set[str] = set()
    allowlist = settings.model_menu_provider_allowlist()
    order = settings.model_menu_provider_order()

    # (provider_id, model_ref, supports_thinking) for routed + discovered models.
    entries: list[tuple[str, str, bool | None]] = []
    for ref in settings.configured_chat_model_refs():
        if allowlist is not None and ref.provider_id not in allowlist:
            continue
        supports_thinking = None
        if provider_registry is not None:
            supports_thinking = provider_registry.cached_model_supports_thinking(
                ref.provider_id, ref.model_id
            )
        entries.append((ref.provider_id, ref.model_ref, supports_thinking))
    if provider_registry is not None:
        for model_info in provider_registry.cached_prefixed_model_infos():
            provider_id = model_info.model_id.split("/", 1)[0]
            if allowlist is not None and provider_id not in allowlist:
                continue
            entries.append(
                (provider_id, model_info.model_id, model_info.supports_thinking)
            )

    def _emit_entries() -> None:
        for _provider_id, model_ref, supports_thinking in entries:
            _append_provider_model_variants(
                models, seen, model_ref, supports_thinking=supports_thinking
            )

    def _emit_claude() -> None:
        for model in SUPPORTED_CLAUDE_MODELS:
            _append_unique_model(models, seen, model)

    if order:
        # Built-in Claude first, then provider entries grouped in configured order
        # (stable sort keeps routed-before-discovered within each provider).
        rank = {provider_id: i for i, provider_id in enumerate(order)}
        entries.sort(key=lambda e: rank.get(e[0], len(order)))
        _emit_claude()
        _emit_entries()
    else:
        _emit_entries()
        _emit_claude()

    return ModelsListResponse(
        data=models,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


def _discovered_model_response(model_id: str, *, display_name: str) -> ModelResponse:
    return ModelResponse(
        id=model_id,
        display_name=display_name,
        created_at=DISCOVERED_MODEL_CREATED_AT,
    )


def _append_unique_model(
    models: list[ModelResponse], seen: set[str], model: ModelResponse
) -> None:
    if model.id in seen:
        return
    seen.add(model.id)
    models.append(model)


def _append_provider_model_variants(
    models: list[ModelResponse],
    seen: set[str],
    provider_model_ref: str,
    *,
    supports_thinking: bool | None = None,
) -> None:
    if supports_thinking is not False:
        _append_unique_model(
            models,
            seen,
            _discovered_model_response(
                gateway_model_id(provider_model_ref),
                display_name=provider_model_ref,
            ),
        )
    _append_unique_model(
        models,
        seen,
        _discovered_model_response(
            no_thinking_gateway_model_id(provider_model_ref),
            display_name=f"{provider_model_ref} (no thinking)",
        ),
    )
