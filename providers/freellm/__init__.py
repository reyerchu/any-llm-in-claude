"""FreeLLMAPI provider package.

tashfeenahmed/freellmapi — an OpenAI-compatible unified proxy that rotates the
free tiers of 16 LLM providers behind one /v1 endpoint. We expose it as a thin
``openai_chat`` adapter so the existing ``NvidiaNimProvider``-style transport
handles streaming / tools / thinking uniformly.
"""

from providers.defaults import FREELLM_DEFAULT_BASE

from .client import FreeLlmProvider

__all__ = ["FREELLM_DEFAULT_BASE", "FreeLlmProvider"]
