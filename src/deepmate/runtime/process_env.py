"""Process environment helpers for Deepmate-managed subprocesses."""

from __future__ import annotations

import os
from collections.abc import Mapping

_UTF8_LOCALE = "en_US.UTF-8"


def subprocess_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment suitable for Deepmate-managed subprocesses."""
    env = dict(os.environ if base is None else base)
    lang = _utf8_or_default(env.get("LANG"))
    env["LANG"] = lang
    env["LC_ALL"] = _utf8_or_default(env.get("LC_ALL"), fallback=lang)
    env["LC_CTYPE"] = _utf8_or_default(env.get("LC_CTYPE"), fallback=env["LC_ALL"])
    return env


def _utf8_or_default(value: str | None, *, fallback: str = _UTF8_LOCALE) -> str:
    clean = (value or "").strip()
    upper = clean.upper()
    if not clean or upper in {"C", "POSIX", "C.UTF-8", "C.UTF8"}:
        return fallback
    if "UTF-8" in upper or "UTF8" in upper:
        return clean
    return fallback
