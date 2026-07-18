"""Shared SSRF-resistant HTTP requests for explicitly public URLs."""

from __future__ import annotations

import http.client
import socket
import ssl
from urllib.parse import urlparse
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from deepmate.url_safety import public_url_resolution, validate_public_url

MAX_PUBLIC_REDIRECTS = 5
_CROSS_ORIGIN_SENSITIVE_HEADERS = (
    "Authorization",
    "Proxy-Authorization",
    "Cookie",
)


def open_public_request(request: Request, *, timeout: float = 20):
    """Open one public HTTP request with redirect and DNS rebinding defenses."""
    validate_public_url(request.full_url)
    return build_opener(
        ProxyHandler({}),
        _PublicHTTPHandler(),
        _PublicHTTPSHandler(),
        _PublicRedirectHandler(),
    ).open(request, timeout=timeout)


class _PublicHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that connects to a prevalidated public IP."""

    _scheme = "http"

    def connect(self) -> None:
        resolution = public_url_resolution(
            _connection_url(self._scheme, self.host, self.port)
        )
        self.sock = socket.create_connection(
            (resolution.first_address(), self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to a validated public IP with the original SNI."""

    _scheme = "https"

    def __init__(self, *args, context=None, check_hostname=None, **kwargs) -> None:
        if context is None:
            context = ssl._create_default_https_context()
        super().__init__(*args, context=context, check_hostname=check_hostname, **kwargs)

    def connect(self) -> None:
        resolution = public_url_resolution(
            _connection_url(self._scheme, self.host, self.port)
        )
        sock = socket.create_connection(
            (resolution.first_address(), self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            server_hostname = self._tunnel_host
        else:
            server_hostname = self.host
        self.sock = self._context.wrap_socket(
            sock,
            server_hostname=server_hostname if ssl.HAS_SNI else None,
        )


class _PublicHTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PublicHTTPConnection, req)


class _PublicHTTPSHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PublicHTTPSConnection, req)


class _PublicRedirectHandler(HTTPRedirectHandler):
    max_repeats = MAX_PUBLIC_REDIRECTS
    max_redirections = MAX_PUBLIC_REDIRECTS

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        redirected = super().redirect_request(
            request,
            fp,
            code,
            msg,
            headers,
            newurl,
        )
        if redirected is not None and _origin(request.full_url) != _origin(newurl):
            for header in _CROSS_ORIGIN_SENSITIVE_HEADERS:
                redirected.remove_header(header)
        return redirected


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    port = parsed.port or (80 if scheme == "http" else 443)
    return scheme, (parsed.hostname or "").lower().rstrip("."), port


def _connection_url(scheme: str, host: str, port: int) -> str:
    hostname = host.strip("[]")
    display_host = (
        f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    )
    return f"{scheme}://{display_host}:{port}/"
