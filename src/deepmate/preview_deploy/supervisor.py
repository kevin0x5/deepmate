"""Long-lived local supervisor for static preview deploys."""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import signal
import socket
import sys
import threading
from pathlib import Path

from deepmate.preview_deploy.state import (
    PreviewDeployState,
    PreviewDeployStore,
    expires_at_iso,
    now_iso,
    parse_datetime,
)
from deepmate.runtime.wakelock import RuntimeWakeSession, WakeConfig


class _QuietStaticHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler without noisy stdout logging."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def main(argv: list[str] | None = None) -> int:
    """Run a local static preview until stopped or expired."""
    parser = argparse.ArgumentParser(prog="deepmate-preview-supervisor")
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--target-kind", required=True)
    parser.add_argument("--serve-root", required=True)
    parser.add_argument("--entry-path", default="")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--project-slug", required=True)
    parser.add_argument("--ttl-seconds", type=int, required=True)
    parser.add_argument("--owner-session-id", default="")
    parser.add_argument("--owner-session-workspace", default="")
    parser.add_argument("--public-url", default="")
    args = parser.parse_args(argv)

    store = PreviewDeployStore(args.state_path)
    serve_root = Path(args.serve_root).resolve()
    target_path = Path(args.target_path).resolve()
    if not serve_root.exists() or not serve_root.is_dir():
        state = _state_from_args(args, status="stale", local_url="")
        store.save(state.with_status("stale", message="static preview root is missing"))
        return 1

    handler = functools.partial(_QuietStaticHandler, directory=str(serve_root))
    try:
        server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), handler)
    except OSError as exc:
        state = _state_from_args(args, status="stale", local_url="")
        store.save(
            state.with_status(
                "stale",
                message=f"static preview bind failed: {exc}",
            )
        )
        return 1
    server.daemon_threads = True
    local_url = _local_url(server.server_port, args.entry_path)
    lan_url = _lan_url(server.server_port, args.entry_path)
    wake_session = RuntimeWakeSession(
        f"Deepmate preview deploy: {args.project_name}",
        WakeConfig(enabled=_wake_enabled(), post_turn_grace_minutes=0),
    )

    stopped = threading.Event()

    expired = threading.Event()

    def stop(signum: int | None = None, frame: object | None = None) -> None:
        if stopped.is_set():
            return
        stopped.set()
        threading.Thread(target=_shutdown_server, args=(server,), daemon=True).start()

    def expire() -> None:
        expired.set()
        stop()

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop)

    state = _state_from_args(
        args,
        status="running",
        local_url=local_url,
        lan_url=lan_url,
    )
    store.save(state)
    threading.Thread(
        target=_watch_ttl,
        args=(state.expires_at, stopped, expire),
        daemon=True,
    ).start()
    wake_session.start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stopped.set()
        wake_session.release()
        server.server_close()
        current = store.load()
        if current is not None and current.supervisor_pid == os.getpid():
            final_status = (
                "expired"
                if expired.is_set() or not _still_within_ttl(current)
                else "stopped"
            )
            store.save(current.with_status(final_status))
    return 0


def _state_from_args(
    args: argparse.Namespace,
    *,
    status: str,
    local_url: str,
    lan_url: str = "",
) -> PreviewDeployState:
    return PreviewDeployState(
        status=status,
        owner_session_id=args.owner_session_id,
        owner_session_workspace=args.owner_session_workspace,
        target_path=str(Path(args.target_path).resolve()),
        target_kind=args.target_kind,
        project_name=args.project_name,
        project_slug=args.project_slug,
        local_url=local_url,
        lan_url=lan_url,
        public_url=args.public_url,
        provider="local" if not args.public_url else "deepmate-preview-relay",
        started_at=now_iso(),
        expires_at=expires_at_iso(args.ttl_seconds),
        supervisor_pid=os.getpid(),
        process_ids=(os.getpid(),),
        local_service_owner="deepmate",
        wake_lock_id="preview_deploy",
        message="" if args.public_url else "Deepmate preview relay is not configured.",
    )


def _local_url(port: int, entry_path: str) -> str:
    clean_entry = entry_path.strip().lstrip("/")
    if not clean_entry:
        return f"http://127.0.0.1:{port}/"
    quoted = "/".join(_quote_segment(segment) for segment in clean_entry.split("/"))
    return f"http://127.0.0.1:{port}/{quoted}"


def _lan_url(port: int, entry_path: str) -> str:
    ip = _lan_ip()
    if not ip:
        return ""
    clean_entry = entry_path.strip().lstrip("/")
    if not clean_entry:
        return f"http://{ip}:{port}/"
    quoted = "/".join(_quote_segment(segment) for segment in clean_entry.split("/"))
    return f"http://{ip}:{port}/{quoted}"


def _lan_ip() -> str:
    """Best-effort LAN IP discovery without requiring an outbound connection."""
    candidates: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidates.append(sock.getsockname()[0])
    except OSError:
        pass
    try:
        candidates.extend(
            ip
            for ip in socket.gethostbyname_ex(socket.gethostname())[2]
            if isinstance(ip, str)
        )
    except OSError:
        pass
    for ip in candidates:
        clean = ip.strip()
        if clean and not clean.startswith("127."):
            return clean
    return ""


def _quote_segment(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _wake_enabled() -> bool:
    return os.environ.get("DEEPMATE_PREVIEW_WAKE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _shutdown_server(server: http.server.ThreadingHTTPServer) -> None:
    try:
        server.shutdown()
    except OSError:
        pass


def _watch_ttl(expires_at: str, stopped: threading.Event, expire) -> None:
    while not stopped.wait(0.25):
        parsed = parse_datetime(expires_at)
        if parsed is None or not _still_before(parsed):
            expire()
            return


def _still_before(expires_at) -> bool:
    from datetime import datetime

    return datetime.now().astimezone() < expires_at


def _still_within_ttl(state: PreviewDeployState) -> bool:
    from datetime import datetime

    expires_at = parse_datetime(state.expires_at)
    if expires_at is None:
        return False
    return datetime.now().astimezone() < expires_at


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
