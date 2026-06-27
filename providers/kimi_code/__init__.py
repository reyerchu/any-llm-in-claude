"""Kimi Code (subscription OAuth) provider exports."""

from config.provider_catalog import KIMI_CODE_DEFAULT_BASE

from .client import KimiCodeProvider

__all__ = [
    "KIMI_CODE_DEFAULT_BASE",
    "KimiCodeProvider",
]
