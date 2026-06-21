"""Preview deploy health checks."""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """Result of checking a preview URL."""

    ok: bool
    status_code: int = 0
    message: str = ""


def check_url(url: str, *, timeout_seconds: float = 2.0) -> HealthCheck:
    """Return whether a URL responds with a non-error HTTP status."""
    clean = url.strip()
    if not clean:
        return HealthCheck(False, message="URL is empty")
    request = urllib.request.Request(
        clean,
        headers={"User-Agent": "deepmate-preview-check/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 0) or 0)
            if 200 <= status < 400:
                return HealthCheck(True, status_code=status, message="ok")
            return HealthCheck(False, status_code=status, message=f"HTTP {status}")
    except urllib.error.HTTPError as exc:
        return HealthCheck(False, status_code=exc.code, message=f"HTTP {exc.code}")
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return HealthCheck(False, message=str(exc))
