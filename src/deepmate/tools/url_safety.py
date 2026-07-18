"""Compatibility imports for the shared public URL safety helpers."""

from deepmate.url_safety import (
    PublicUrlResolution,
    public_url_resolution,
    resolved_addresses,
    validate_public_url,
)

__all__ = [
    "PublicUrlResolution",
    "public_url_resolution",
    "resolved_addresses",
    "validate_public_url",
]
