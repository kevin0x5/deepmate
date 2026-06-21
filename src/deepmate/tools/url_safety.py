"""Shared public URL validation for network-facing tools."""

from __future__ import annotations

import ipaddress
import socket
import threading
from dataclasses import dataclass
from time import monotonic
from urllib.parse import urlparse

DNS_CACHE_TTL_SECONDS = 30.0
MAX_DNS_CACHE_ENTRIES = 256
_DNS_CACHE: dict[tuple[str, int], tuple[float, frozenset[str]]] = {}
_DNS_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class PublicUrlResolution:
    """Validated public network target for one URL."""

    hostname: str
    port: int
    addresses: tuple[str, ...]

    def first_address(self) -> str:
        if not self.addresses:
            raise ValueError(f"could not resolve hostname: {self.hostname}")
        return self.addresses[0]


def validate_public_url(url: str) -> None:
    """Raise when a URL points at local, private, reserved, or invalid hosts."""
    public_url_resolution(url)


def public_url_resolution(url: str) -> PublicUrlResolution:
    """Resolve and validate a public URL target."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must use http or https and include a hostname")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("local network URLs are not allowed")
    port = parsed.port or _default_port(parsed.scheme)
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if not literal_ip.is_global:
            raise ValueError("private, loopback, link-local, and reserved URLs are not allowed")
        return PublicUrlResolution(
            hostname=hostname,
            port=port,
            addresses=(str(literal_ip),),
        )
    try:
        addresses = resolved_addresses(hostname, port)
    except OSError as exc:
        raise ValueError(f"could not resolve hostname: {hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private, loopback, link-local, and reserved URLs are not allowed")
    return PublicUrlResolution(
        hostname=hostname,
        port=port,
        addresses=tuple(sorted(addresses)),
    )


def resolved_addresses(hostname: str, port: int) -> frozenset[str]:
    """Return cached DNS addresses for one host and port."""
    key = (hostname, port)
    now = monotonic()
    with _DNS_CACHE_LOCK:
        cached = _DNS_CACHE.get(key)
        if cached and now - cached[0] <= DNS_CACHE_TTL_SECONDS:
            return cached[1]
    addresses = frozenset(entry[4][0] for entry in socket.getaddrinfo(hostname, port))
    with _DNS_CACHE_LOCK:
        _DNS_CACHE[key] = (now, addresses)
        if len(_DNS_CACHE) > MAX_DNS_CACHE_ENTRIES:
            stale_keys = sorted(_DNS_CACHE, key=lambda item: _DNS_CACHE[item][0])[
                : len(_DNS_CACHE) - MAX_DNS_CACHE_ENTRIES
            ]
            for stale_key in stale_keys:
                _DNS_CACHE.pop(stale_key, None)
    return addresses


def _default_port(scheme: str) -> int:
    return 80 if scheme == "http" else 443
