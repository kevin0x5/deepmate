"""Preview relay provider and lightweight tunnel worker."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from deepmate.runtime.process_env import subprocess_environment

DEFAULT_RELAY_TIMEOUT_SECONDS = 5.0
DEFAULT_TUNNEL_POLL_SECONDS = 1.0
MAX_TUNNEL_REQUEST_BODY_BYTES = 1_000_000
MAX_TUNNEL_RESPONSE_BODY_BYTES = 5_000_000
ALLOWED_TUNNEL_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
FORWARDED_REQUEST_HEADERS_DENYLIST = frozenset(
    {
        "authorization",
        "cookie",
        "host",
        "connection",
        "content-length",
        "proxy-authorization",
        "x-api-key",
        "x-auth-token",
    }
)


class PreviewTunnelError(RuntimeError):
    """Raised when an external preview tunnel cannot be created."""


@dataclass(frozen=True, slots=True)
class PreviewLease:
    """Relay lease assigned to one temporary preview."""

    lease_id: str
    public_url: str
    tunnel_url: str = ""
    tunnel_token: str = ""
    expires_at: str = ""


@dataclass(frozen=True, slots=True)
class PreviewTunnel:
    """Local process details for an opened preview tunnel."""

    lease: PreviewLease
    public_url: str
    tunnel_pid: int = 0
    process_ids: tuple[int, ...] = ()
    process: Any | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class PreviewTunnelStatus:
    """Relay status for one preview lease."""

    ok: bool
    status: str = ""
    public_url: str = ""
    message: str = ""


class PreviewTunnelProvider:
    """Provider boundary for public preview tunnels."""

    provider_name = "deepmate-preview-relay"

    def create_lease(self, project_slug: str, ttl_seconds: int) -> PreviewLease:
        raise NotImplementedError

    def open_tunnel(self, local_url: str, lease: PreviewLease) -> PreviewTunnel:
        raise NotImplementedError

    def close_tunnel(self, lease_id: str) -> None:
        raise NotImplementedError

    def status(self, lease_id: str) -> PreviewTunnelStatus:
        raise NotImplementedError


class UnconfiguredPreviewTunnelProvider(PreviewTunnelProvider):
    """Provider used when the Deepmate preview relay endpoint is unavailable."""

    def create_lease(self, project_slug: str, ttl_seconds: int) -> PreviewLease:
        raise PreviewTunnelError(
            "Deepmate preview relay is not configured in this build."
        )

    def open_tunnel(self, local_url: str, lease: PreviewLease) -> PreviewTunnel:
        raise PreviewTunnelError(
            "Deepmate preview relay is not configured in this build."
        )

    def close_tunnel(self, lease_id: str) -> None:
        return

    def status(self, lease_id: str) -> PreviewTunnelStatus:
        return PreviewTunnelStatus(
            ok=False,
            status="unconfigured",
            message="Deepmate preview relay is not configured in this build.",
        )


class HttpPreviewRelayProvider(PreviewTunnelProvider):
    """HTTP client for the Deepmate preview relay control plane."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout_seconds: float = DEFAULT_RELAY_TIMEOUT_SECONDS,
    ) -> None:
        clean = base_url.strip().rstrip("/")
        if not clean:
            raise ValueError("preview relay base URL is required")
        self.base_url = clean
        self.token = token.strip()
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> PreviewTunnelProvider:
        """Return the configured relay provider, or an unconfigured provider."""
        base_url = os.environ.get("DEEPMATE_PREVIEW_RELAY_URL", "").strip()
        if not base_url:
            return UnconfiguredPreviewTunnelProvider()
        return cls(
            base_url,
            token=os.environ.get("DEEPMATE_PREVIEW_RELAY_TOKEN", "").strip(),
        )

    def create_lease(self, project_slug: str, ttl_seconds: int) -> PreviewLease:
        payload = self._request_json(
            "POST",
            "/v1/preview/leases",
            {
                "project_slug": project_slug,
                "ttl_seconds": max(1, int(ttl_seconds)),
            },
        )
        lease_id = str(payload.get("lease_id") or payload.get("id") or "").strip()
        public_url = str(payload.get("public_url") or "").strip()
        if not lease_id or not public_url:
            raise PreviewTunnelError(
                "preview relay lease response is missing lease_id or public_url"
            )
        return PreviewLease(
            lease_id=lease_id,
            public_url=public_url,
            tunnel_url=str(payload.get("tunnel_url") or "").strip(),
            tunnel_token=str(payload.get("tunnel_token") or payload.get("token") or "").strip(),
            expires_at=str(payload.get("expires_at") or "").strip(),
        )

    def open_tunnel(self, local_url: str, lease: PreviewLease) -> PreviewTunnel:
        if not lease.tunnel_url:
            payload = self._request_json(
                "POST",
                f"/v1/preview/leases/{_quote_path(lease.lease_id)}/tunnel",
                {"local_url": local_url, "lease_id": lease.lease_id},
            )
            tunnel_url = str(payload.get("tunnel_url") or "").strip()
            tunnel_token = str(
                payload.get("tunnel_token") or payload.get("token") or ""
            ).strip()
            lease = PreviewLease(
                lease_id=lease.lease_id,
                public_url=lease.public_url,
                tunnel_url=tunnel_url,
                tunnel_token=tunnel_token or lease.tunnel_token,
                expires_at=lease.expires_at,
            )
        if not lease.tunnel_url:
            raise PreviewTunnelError("preview relay did not return a tunnel endpoint")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "deepmate.preview_deploy.tunnel",
                "--worker",
                "--tunnel-url",
                lease.tunnel_url,
                "--local-url",
                local_url,
                "--lease-id",
                lease.lease_id,
                "--token",
                lease.tunnel_token,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=subprocess_environment(),
            start_new_session=True,
            text=True,
        )
        return PreviewTunnel(
            lease=lease,
            public_url=lease.public_url,
            tunnel_pid=process.pid,
            process_ids=(process.pid,),
            process=process,
            message="preview relay tunnel started",
        )

    def close_tunnel(self, lease_id: str) -> None:
        clean = lease_id.strip()
        if not clean:
            return
        try:
            self._request_json("DELETE", f"/v1/preview/leases/{_quote_path(clean)}")
        except PreviewTunnelError:
            return

    def status(self, lease_id: str) -> PreviewTunnelStatus:
        clean = lease_id.strip()
        if not clean:
            return PreviewTunnelStatus(ok=False, status="missing_lease")
        try:
            payload = self._request_json(
                "GET", f"/v1/preview/leases/{_quote_path(clean)}"
            )
        except PreviewTunnelError as exc:
            return PreviewTunnelStatus(ok=False, status="unknown", message=str(exc))
        status = str(payload.get("status") or "").strip()
        public_url = str(payload.get("public_url") or "").strip()
        message = str(payload.get("message") or "").strip()
        return PreviewTunnelStatus(
            ok=status in {"running", "ready", "healthy"},
            status=status,
            public_url=public_url,
            message=message,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        url = self.base_url + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise PreviewTunnelError(
                _relay_error_message(f"preview relay returned HTTP {exc.code}", body)
            ) from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise PreviewTunnelError(f"preview relay request failed: {exc}") from exc
        if not body.strip():
            return {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PreviewTunnelError("preview relay returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise PreviewTunnelError("preview relay response must be a JSON object")
        return parsed

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "deepmate-preview-relay/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


def configured_tunnel_provider() -> PreviewTunnelProvider:
    """Return the preview tunnel provider configured for this process."""
    return HttpPreviewRelayProvider.from_environment()


def run_tunnel_worker(
    *,
    tunnel_url: str,
    local_url: str,
    lease_id: str,
    token: str = "",
    poll_seconds: float = DEFAULT_TUNNEL_POLL_SECONDS,
) -> int:
    """Run a small long-poll tunnel worker for relay-managed preview traffic."""
    endpoint = tunnel_url.rstrip("/")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "deepmate-preview-tunnel/1.0",
    }
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    while True:
        try:
            request_payload = _worker_request(
                f"{endpoint}/poll",
                headers,
                {"lease_id": lease_id, "local_url": local_url},
            )
            if not request_payload:
                time.sleep(max(0.1, poll_seconds))
                continue
            response_payload = _forward_tunnel_request(local_url, request_payload)
            response_payload["lease_id"] = lease_id
            _worker_request(f"{endpoint}/respond", headers, response_payload)
        except KeyboardInterrupt:
            return 0
        except Exception:
            time.sleep(max(0.5, poll_seconds))


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the tunnel worker subprocess."""
    parser = argparse.ArgumentParser(prog="deepmate-preview-tunnel")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tunnel-url", required=True)
    parser.add_argument("--local-url", required=True)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("--worker is required")
    return run_tunnel_worker(
        tunnel_url=args.tunnel_url,
        local_url=args.local_url,
        lease_id=args.lease_id,
        token=args.token,
    )


def _forward_tunnel_request(
    local_url: str,
    payload: dict[str, object],
) -> dict[str, object]:
    request_id = str(payload.get("request_id") or "").strip()
    method = str(payload.get("method") or "GET").strip().upper() or "GET"
    if method not in ALLOWED_TUNNEL_METHODS:
        return _tunnel_error_response(
            request_id,
            405,
            f"Preview tunnel blocked unsupported method: {method}",
        )
    path = str(payload.get("path") or "/").strip() or "/"
    body_text = str(payload.get("body_base64") or "")
    try:
        body = base64.b64decode(body_text, validate=True) if body_text else None
    except (ValueError, binascii.Error):
        return _tunnel_error_response(request_id, 400, "Invalid request body encoding")
    if body is not None and len(body) > MAX_TUNNEL_REQUEST_BODY_BYTES:
        return _tunnel_error_response(request_id, 413, "Preview request body is too large")
    target = urllib.parse.urljoin(local_url.rstrip("/") + "/", path.lstrip("/"))
    request_data = body if method not in {"GET", "HEAD"} else None
    request = urllib.request.Request(target, data=request_data, method=method)
    headers = payload.get("headers")
    if isinstance(headers, dict):
        for key, value in headers.items():
            clean_key = str(key).strip()
            if clean_key.lower() in FORWARDED_REQUEST_HEADERS_DENYLIST:
                continue
            request.add_header(clean_key, str(value))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = _read_bounded_response(response)
            status_code = int(getattr(response, "status", 200) or 200)
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        response_body = _read_bounded_response(exc)
        status_code = int(exc.code)
        response_headers = dict(exc.headers.items())
    except Exception as exc:
        response_body = str(exc).encode("utf-8", errors="replace")
        status_code = 502
        response_headers = {"Content-Type": "text/plain; charset=utf-8"}
    return {
        "request_id": request_id,
        "status_code": status_code,
        "headers": response_headers,
        "body_base64": base64.b64encode(response_body).decode("ascii"),
    }


def _read_bounded_response(response) -> bytes:
    body = response.read(MAX_TUNNEL_RESPONSE_BODY_BYTES + 1)
    if len(body) > MAX_TUNNEL_RESPONSE_BODY_BYTES:
        return body[:MAX_TUNNEL_RESPONSE_BODY_BYTES]
    return body


def _tunnel_error_response(
    request_id: str,
    status_code: int,
    message: str,
) -> dict[str, object]:
    body = message.encode("utf-8", errors="replace")
    return {
        "request_id": request_id,
        "status_code": status_code,
        "headers": {"Content-Type": "text/plain; charset=utf-8"},
        "body_base64": base64.b64encode(body).decode("ascii"),
    }


def _worker_request(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return {}
        raise
    if not body.strip():
        return {}
    parsed = json.loads(body)
    return parsed if isinstance(parsed, dict) else {}


def _quote_path(value: str) -> str:
    return urllib.parse.quote(value.strip(), safe="")


def _relay_error_message(prefix: str, body: str) -> str:
    clean = " ".join(body.split())
    if not clean:
        return prefix
    return f"{prefix}: {clean[:500]}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
