"""User-facing /deploy command handling."""

from __future__ import annotations

import json
import os
import re
import select
import secrets
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from deepmate.foundation import display_path
from deepmate.preview_deploy.health import HealthCheck, check_url
from deepmate.preview_deploy.state import (
    PreviewDeployState,
    PreviewDeployStore,
    expires_at_iso,
    now_iso,
    parse_datetime,
)
from deepmate.preview_deploy.tunnel import (
    PreviewTunnelError,
    PreviewTunnelProvider,
    configured_tunnel_provider,
)
from deepmate.runtime.process_env import subprocess_environment

COMMAND_ALIASES = {"/deploy", "/deployment", "/deploment"}
DEFAULT_TTL_SECONDS = 2 * 60 * 60
MAX_TTL_SECONDS = 24 * 60 * 60
COMMON_DEV_SERVER_PORTS = (5173, 3000, 4173, 8000, 8080)
PUBLIC_PREVIEW_CHECK_TIMEOUT_SECONDS = 10.0
_SUPERVISOR_PROCESSES: dict[int, subprocess.Popen] = {}


@dataclass(frozen=True, slots=True)
class _DeployArgs:
    action: str
    target: str = ""
    path: str = ""
    url: str = ""
    name: str = ""
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    replace: bool = False


@dataclass(frozen=True, slots=True)
class _Target:
    target_path: Path
    target_kind: str
    serve_root: Path | None = None
    entry_path: str = ""
    local_url: str = ""
    local_service_owner: str = "deepmate"


def is_deploy_command(prompt: str) -> bool:
    """Return whether text starts with the preview deploy command."""
    clean = prompt.strip()
    if not clean:
        return False
    return clean.split(maxsplit=1)[0] in COMMAND_ALIASES


def handle_deploy_command(
    prompt: str,
    *,
    workspace: str | Path,
    data_dir: str | Path,
    owner_session_id: str = "",
    owner_session_workspace: str | Path | None = None,
    tunnel_provider: PreviewTunnelProvider | None = None,
) -> str:
    """Handle one /deploy command and return user-facing text."""
    args = _parse_args(prompt)
    store = PreviewDeployStore.in_data_dir(data_dir)
    provider = tunnel_provider or configured_tunnel_provider()
    if args.action == "status":
        state = _refresh_state(store, tunnel_provider=provider)
        return _format_status(state, Path(workspace))
    if args.action == "stop":
        return _stop_preview(store, Path(workspace), tunnel_provider=provider)

    active = _refresh_state(store, tunnel_provider=provider)
    if active is not None and active.is_active() and not args.replace:
        return _format_existing_preview(active, args, Path(workspace))
    if active is not None and active.is_active() and args.replace:
        _stop_state(store, active, tunnel_provider=provider)

    target = _resolve_target(args, Path(workspace))
    project_name = _project_name(args, target, Path(workspace))
    project_slug = _slugify(project_name)
    if target.target_kind in {"static_file", "static_dir"}:
        state = _start_static_preview(
            store=store,
            target=target,
            project_name=project_name,
            project_slug=project_slug,
            ttl_seconds=args.ttl_seconds,
            owner_session_id=owner_session_id,
            owner_session_workspace=owner_session_workspace or workspace,
        )
        try:
            state = _open_external_preview(
                store=store,
                state=state,
                ttl_seconds=args.ttl_seconds,
                tunnel_provider=provider,
            )
        except RuntimeError as exc:
            state = _keep_local_preview_after_relay_failure(store, state, exc)
    elif target.target_kind == "external_url":
        health = check_url(target.local_url)
        if not health.ok:
            raise RuntimeError(
                _external_deploy_failure_message(
                    RuntimeError(f"Local service check failed: {health.message}")
                )
            )
        status = "running" if health.ok else "unhealthy"
        state = store.save(
            PreviewDeployState(
                status=status,
                owner_session_id=owner_session_id.strip(),
                owner_session_workspace=str(
                    Path(owner_session_workspace or workspace).resolve()
                ),
                target_path=str(target.target_path.resolve()),
                target_kind="external_url",
                project_name=project_name,
                project_slug=project_slug,
                local_url=target.local_url,
                provider="local",
                started_at=now_iso(),
                expires_at=expires_at_iso(args.ttl_seconds),
                local_service_owner="external",
                message=(
                    ""
                    if health.ok
                    else f"Local service check failed: {health.message}"
                ),
            )
        )
        if health.ok:
            try:
                state = _open_external_preview(
                    store=store,
                    state=state,
                    ttl_seconds=args.ttl_seconds,
                    tunnel_provider=provider,
                )
            except RuntimeError as exc:
                state = _keep_local_preview_after_relay_failure(store, state, exc)
    else:
        raise ValueError(f"unsupported deploy target: {target.target_kind}")
    return _format_created(state, Path(workspace))


def _parse_args(prompt: str) -> _DeployArgs:
    parts = _split_prompt(prompt)
    if not parts or parts[0] not in COMMAND_ALIASES:
        raise ValueError("not a deploy command")
    action = "create"
    replace = False
    index = 1
    if index < len(parts) and parts[index] in {"status", "stop", "replace"}:
        action = parts[index]
        replace = action == "replace"
        if replace:
            action = "create"
        index += 1
    target = ""
    path = ""
    url = ""
    name = ""
    ttl_seconds = DEFAULT_TTL_SECONDS
    while index < len(parts):
        part = parts[index]
        if part == "--path":
            index += 1
            path = _require_value(parts, index, "--path")
        elif part == "--url":
            index += 1
            url = _require_value(parts, index, "--url")
        elif part == "--name":
            index += 1
            name = _require_value(parts, index, "--name")
        elif part == "--ttl":
            index += 1
            ttl_seconds = _parse_ttl(_require_value(parts, index, "--ttl"))
        elif part.startswith("--"):
            raise ValueError(f"unknown /deploy option: {part}")
        elif not target:
            target = part
        else:
            raise ValueError(f"unexpected /deploy argument: {part}")
        index += 1
    if url and not _is_http_url(url):
        raise ValueError("--url must be an http:// or https:// URL")
    return _DeployArgs(
        action=action,
        target=target,
        path=path,
        url=url,
        name=name,
        ttl_seconds=ttl_seconds,
        replace=replace,
    )


def _split_prompt(prompt: str) -> list[str]:
    try:
        return shlex.split(prompt.strip())
    except ValueError as exc:
        raise ValueError(f"invalid /deploy command: {exc}") from exc


def _require_value(parts: list[str], index: int, option: str) -> str:
    if index >= len(parts) or not parts[index].strip():
        raise ValueError(f"{option} requires a value")
    return parts[index]


def _parse_ttl(value: str) -> int:
    clean = value.strip().lower()
    if not clean:
        raise ValueError("--ttl requires a duration such as 30m or 2h")
    unit = clean[-1] if clean[-1] in {"s", "m", "h"} else "m"
    number_text = clean[:-1] if clean[-1] in {"s", "m", "h"} else clean
    if not number_text.isdigit():
        raise ValueError("--ttl must be a duration such as 30m or 2h")
    amount = int(number_text)
    if unit == "h":
        seconds = amount * 60 * 60
    elif unit == "m":
        seconds = amount * 60
    else:
        seconds = amount
    if seconds <= 0:
        raise ValueError("--ttl must be positive")
    if seconds > MAX_TTL_SECONDS:
        raise ValueError("--ttl must be 24h or less")
    return seconds


def _resolve_target(args: _DeployArgs, workspace: Path) -> _Target:
    if args.url:
        target_path = _resolve_path(args.path or args.target or ".", workspace)
        return _Target(
            target_path=target_path,
            target_kind="external_url",
            local_url=args.url.strip(),
            local_service_owner="external",
        )
    raw_target = args.path or args.target or "."
    target_path = _resolve_path(raw_target, workspace)
    _require_workspace_path(target_path, workspace)
    if target_path.is_file():
        if target_path.suffix.lower() != ".html":
            raise ValueError("only .html files can be deployed directly")
        return _Target(
            target_path=target_path,
            target_kind="static_file",
            serve_root=target_path.parent,
            entry_path=target_path.name,
            local_service_owner="deepmate",
        )
    if target_path.is_dir():
        if (target_path / "index.html").is_file():
            return _Target(
                target_path=target_path,
                target_kind="static_dir",
                serve_root=target_path,
                local_service_owner="deepmate",
            )
        if raw_target in {".", ""}:
            discovered = _discover_static_target(target_path)
            if discovered is not None:
                return discovered
            local_url = _discover_local_server()
            if local_url:
                return _Target(
                    target_path=target_path,
                    target_kind="external_url",
                    local_url=local_url,
                    local_service_owner="external",
                )
            raise ValueError(_no_default_preview_target_message(target_path))
        raise ValueError(_missing_index_message(target_path))
    raise ValueError(_missing_target_message(target_path))


def _discover_static_target(workspace: Path) -> _Target | None:
    for relative in ("dist", "build", "public"):
        candidate = workspace / relative
        if (candidate / "index.html").is_file():
            return _Target(
                target_path=candidate,
                target_kind="static_dir",
                serve_root=candidate,
                local_service_owner="deepmate",
            )
    html_files = sorted(workspace.glob("*.html"))
    if len(html_files) == 1:
        html = html_files[0]
        return _Target(
            target_path=html,
            target_kind="static_file",
            serve_root=html.parent,
            entry_path=html.name,
            local_service_owner="deepmate",
        )
    return None


def _no_default_preview_target_message(workspace: Path) -> str:
    return "\n".join(
        (
            "没有找到可直接预览的页面。",
            "",
            "Deepmate 已自动检查：",
            "- dist/index.html",
            "- build/index.html",
            "- public/index.html",
            "- 当前目录下唯一的 .html 文件",
            "- 常见本地开发端口 5173、3000、4173、8000、8080",
            "",
            "最简单的处理方式：先在项目里生成或启动页面，然后重新执行 /deploy。",
            "如果页面已经在浏览器里打开，可以用 /deploy --url http://127.0.0.1:<端口>。",
        )
    )


def _missing_index_message(target_path: Path) -> str:
    return "\n".join(
        (
            "这个目录暂时不能直接预览，因为里面没有 index.html。",
            "",
            f"目录: {target_path}",
            "",
            "可以改用包含 index.html 的目录，或指定单个 HTML 文件。",
        )
    )


def _missing_target_message(target_path: Path) -> str:
    return "\n".join(
        (
            "没有找到要预览的文件或目录。",
            "",
            f"路径: {target_path}",
            "",
            "请确认页面已经生成，或直接使用当前运行中的本地地址：",
            "/deploy --url http://127.0.0.1:<端口>",
        )
    )


def _discover_local_server() -> str:
    checks: dict[Future, str] = {}
    with ThreadPoolExecutor(max_workers=len(COMMON_DEV_SERVER_PORTS)) as executor:
        for port in COMMON_DEV_SERVER_PORTS:
            url = f"http://127.0.0.1:{port}/"
            checks[executor.submit(check_url, url, timeout_seconds=0.3)] = url
        for future in as_completed(checks):
            url = checks[future]
            if future.result().ok:
                return url
    return ""


def _resolve_path(value: str, workspace: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _require_workspace_path(path: Path, workspace: Path) -> None:
    try:
        path.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError("preview target must be inside the workspace") from exc


def _project_name(args: _DeployArgs, target: _Target, workspace: Path) -> str:
    if args.name.strip():
        return args.name.strip()
    base = (
        target.target_path
        if target.target_path.is_dir()
        else target.target_path.parent
    )
    configured = _configured_project_name(base)
    if configured:
        return configured
    if target.target_path.is_file():
        return target.target_path.stem
    if target.target_path != workspace:
        return target.target_path.name
    return workspace.name or "preview"


def _configured_project_name(root: Path) -> str:
    package_json = root / "package.json"
    if package_json.is_file():
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            name = str(payload.get("name", "")).strip()
            if name:
                return name
    for filename in ("pyproject.toml", "Cargo.toml"):
        candidate = root / filename
        if not candidate.is_file():
            continue
        name = _toml_name(candidate)
        if name:
            return name
    go_mod = root / "go.mod"
    if go_mod.is_file():
        try:
            first = go_mod.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            first = ""
        if first.startswith("module "):
            return first.split("/")[-1].strip()
    return ""


def _toml_name(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    in_package = path.name == "Cargo.toml"
    for line in lines:
        clean = line.strip()
        if clean == "[package]":
            in_package = True
            continue
        if clean.startswith("[") and clean != "[package]":
            in_package = False
        if in_package and clean.startswith("name"):
            _, separator, value = clean.partition("=")
            if separator:
                return value.strip().strip("\"'")
    return ""


def _slugify(value: str) -> str:
    base = value.strip().lower().replace("@", "")
    base = re.sub(r"[^a-z0-9]+", "-", base)
    base = re.sub(r"-{2,}", "-", base).strip("-")
    base = (base or "preview")[:32].strip("-") or "preview"
    suffix = "".join(
        secrets.choice("abcdefghijklmnopqrstuvwxyz234567") for _ in range(6)
    )
    return f"{base}-{suffix}"


def _start_static_preview(
    *,
    store: PreviewDeployStore,
    target: _Target,
    project_name: str,
    project_slug: str,
    ttl_seconds: int,
    owner_session_id: str,
    owner_session_workspace: str | Path,
) -> PreviewDeployState:
    if target.serve_root is None:
        raise ValueError("static preview requires a serve root")
    command = [
        sys.executable,
        "-m",
        "deepmate.preview_deploy.supervisor",
        "--state-path",
        str(store.path),
        "--target-path",
        str(target.target_path),
        "--target-kind",
        target.target_kind,
        "--serve-root",
        str(target.serve_root),
        "--entry-path",
        target.entry_path,
        "--project-name",
        project_name,
        "--project-slug",
        project_slug,
        "--ttl-seconds",
        str(ttl_seconds),
        "--owner-session-id",
        owner_session_id.strip(),
        "--owner-session-workspace",
        str(Path(owner_session_workspace).resolve()),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=subprocess_environment(),
        start_new_session=True,
        text=True,
    )
    _SUPERVISOR_PROCESSES[process.pid] = process
    deadline = time.monotonic() + 5
    last_state: PreviewDeployState | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            last_state = store.load()
        except (OSError, ValueError, json.JSONDecodeError):
            last_state = None
        if (
            last_state is not None
            and last_state.supervisor_pid == process.pid
            and last_state.local_url
            and last_state.status == "running"
        ):
            _close_process_stderr(process)
            return last_state
        time.sleep(0.05)
    stderr_text = _read_process_stderr(process)
    _terminate_pid(process.pid)
    if last_state is not None and last_state.message:
        raise ValueError(_append_supervisor_stderr(last_state.message, stderr_text))
    raise RuntimeError(
        _append_supervisor_stderr("preview supervisor did not start", stderr_text)
    )


def _open_external_preview(
    *,
    store: PreviewDeployStore,
    state: PreviewDeployState,
    ttl_seconds: int,
    tunnel_provider: PreviewTunnelProvider,
) -> PreviewDeployState:
    """Create a public relay lease and attach the local preview to it."""
    if not state.local_url.strip():
        raise RuntimeError("local preview URL is missing")
    try:
        lease = tunnel_provider.create_lease(state.project_slug, ttl_seconds)
        tunnel = tunnel_provider.open_tunnel(state.local_url, lease)
    except PreviewTunnelError as exc:
        raise RuntimeError(str(exc)) from exc
    _register_tunnel_process(tunnel)
    public_url = (tunnel.public_url or lease.public_url).strip()
    if not public_url:
        tunnel_provider.close_tunnel(lease.lease_id)
        _terminate_pid(tunnel.tunnel_pid)
        raise RuntimeError("preview relay did not return an external URL")
    public_health = _check_public_url_until(public_url)
    if not public_health.ok:
        tunnel_provider.close_tunnel(lease.lease_id)
        _terminate_pid(tunnel.tunnel_pid)
        raise RuntimeError(
            f"external preview check failed: {public_health.message}"
        )
    updated = replace(
        state,
        status="running",
        public_url=public_url,
        lease_id=lease.lease_id,
        provider=tunnel_provider.provider_name,
        tunnel_pid=tunnel.tunnel_pid,
        process_ids=_merge_process_ids(state.process_ids, tunnel.process_ids),
        message=tunnel.message,
    )
    return store.save(updated)


def _register_tunnel_process(tunnel: object) -> None:
    process = getattr(tunnel, "process", None)
    pid = _int_process_pid(getattr(tunnel, "tunnel_pid", 0))
    if process is None or pid <= 0:
        return
    _SUPERVISOR_PROCESSES[pid] = process


def _int_process_pid(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _keep_local_preview_after_relay_failure(
    store: PreviewDeployStore,
    state: PreviewDeployState,
    exc: BaseException,
) -> PreviewDeployState:
    reason = _relay_failure_reason(exc)
    message = (
        "公开分享链接暂不可用。本地预览已继续运行。"
        f" 原因：{reason}"
    )
    return store.save(
        replace(
            state,
            status="running",
            public_url="",
            lease_id="",
            provider="local",
            tunnel_pid=0,
            message=message,
        )
    )


def _check_public_url_until(url: str) -> HealthCheck:
    deadline = time.monotonic() + PUBLIC_PREVIEW_CHECK_TIMEOUT_SECONDS
    last = check_url(url, timeout_seconds=2)
    while not last.ok and time.monotonic() < deadline:
        time.sleep(0.25)
        last = check_url(url, timeout_seconds=2)
    return last


def _merge_process_ids(
    current: tuple[int, ...],
    extra: tuple[int, ...],
) -> tuple[int, ...]:
    output: list[int] = []
    for pid in (*current, *extra):
        if pid > 0 and pid not in output:
            output.append(pid)
    return tuple(output)


def _refresh_state(
    store: PreviewDeployStore,
    *,
    tunnel_provider: PreviewTunnelProvider | None = None,
) -> PreviewDeployState | None:
    state = store.load()
    if state is None or not state.is_active():
        return state
    expires_at = parse_datetime(state.expires_at)
    if expires_at is not None and datetime.now().astimezone() >= expires_at:
        _stop_state(store, state, status="expired", tunnel_provider=tunnel_provider)
        return store.load()
    if state.supervisor_pid and not _pid_exists(state.supervisor_pid):
        process = _SUPERVISOR_PROCESSES.pop(state.supervisor_pid, None)
        if process is not None:
            _close_process_stderr(process)
        return _stop_state(
            store,
            state,
            status="stale",
            tunnel_provider=tunnel_provider,
            message="preview supervisor exited",
        )
    if state.local_url:
        health = check_url(state.local_url)
        if not health.ok and state.status != "unhealthy":
            unhealthy = state.with_status(
                "unhealthy",
                message=f"Local service check failed: {health.message}",
            )
            store.save(unhealthy)
            return unhealthy
        if health.ok and state.status == "unhealthy":
            running = state.with_status("running", message=state.message)
            store.save(running)
            return running
    if state.lease_id and state.public_url and tunnel_provider is not None:
        relay = tunnel_provider.status(state.lease_id)
        if not relay.ok and state.status != "unhealthy":
            unhealthy = state.with_status(
                "unhealthy",
                message=relay.message or relay.status or "preview relay is unavailable",
            )
            store.save(unhealthy)
            return unhealthy
        if relay.ok and state.status == "unhealthy":
            running = state.with_status("running", message=relay.message)
            if relay.public_url and relay.public_url != running.public_url:
                running = replace(running, public_url=relay.public_url)
            store.save(running)
            return running
    return state


def _stop_preview(
    store: PreviewDeployStore,
    workspace: Path,
    *,
    tunnel_provider: PreviewTunnelProvider | None = None,
) -> str:
    state = store.load()
    if state is None or not state.is_active():
        return "当前没有运行中的临时预览。"
    stopped = _stop_state(store, state, tunnel_provider=tunnel_provider)
    return _format_stopped(stopped, workspace)


def _stop_state(
    store: PreviewDeployStore,
    state: PreviewDeployState,
    *,
    status: str = "stopped",
    tunnel_provider: PreviewTunnelProvider | None = None,
    message: str = "",
) -> PreviewDeployState:
    if state.lease_id and tunnel_provider is not None:
        tunnel_provider.close_tunnel(state.lease_id)
    pids = [state.supervisor_pid, state.tunnel_pid]
    if state.local_service_owner == "deepmate":
        pids.extend(state.process_ids)
    for pid in sorted({pid for pid in pids if pid > 0}, reverse=True):
        _terminate_pid(pid, require_registered=True)
    stopped = state.with_status(status, message=message)
    store.save(stopped)
    return stopped


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, *, require_registered: bool = False) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    process = _SUPERVISOR_PROCESSES.get(pid)
    if require_registered and process is None:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        if process is not None:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
            _close_process_stderr(process)
            _SUPERVISOR_PROCESSES.pop(pid, None)
        return
    except PermissionError:
        return
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            if process is not None:
                try:
                    process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    pass
                _close_process_stderr(process)
                _SUPERVISOR_PROCESSES.pop(pid, None)
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        if process is not None:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
            _close_process_stderr(process)
            _SUPERVISOR_PROCESSES.pop(pid, None)
        return
    except (AttributeError, PermissionError):
        return
    if process is not None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        _close_process_stderr(process)
        _SUPERVISOR_PROCESSES.pop(pid, None)


def _close_process_stderr(process: subprocess.Popen) -> None:
    stderr = process.stderr
    if stderr is None or not hasattr(stderr, "close"):
        return
    try:
        stderr.close()
    except OSError:
        pass


def _read_process_stderr(process: subprocess.Popen) -> str:
    if process.stderr is None:
        return ""
    if process.poll() is None:
        return _read_live_process_stderr(process.stderr)
    try:
        _stdout, stderr = process.communicate(timeout=0.2)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace").strip()
    return str(stderr or "").strip()


def _read_live_process_stderr(stderr: object) -> str:
    fileno = getattr(stderr, "fileno", None)
    if not callable(fileno):
        return ""
    try:
        fd = fileno()
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            return ""
        data = os.read(fd, 4096)
    except (OSError, ValueError):
        return ""
    return data.decode("utf-8", errors="replace").strip()


def _append_supervisor_stderr(message: str, stderr_text: str) -> str:
    clean = " ".join(stderr_text.split())
    if not clean:
        return message
    return f"{message}\nsupervisor stderr: {clean[:500]}"


def _relay_failure_reason(exc: BaseException) -> str:
    reason = str(exc).strip() or "preview relay unavailable"
    if "not configured" in reason.lower():
        return "公开分享链接暂未配置"
    return reason


def _format_created(state: PreviewDeployState, workspace: Path) -> str:
    title = "外部临时预览已开启" if state.public_url else "本地临时预览已开启"
    lines = [
        title,
        "",
        f"项目: {state.project_name or state.project_slug}",
        f"目标路径: {_target_path_text(state, workspace)}",
        f"外部链接: {_public_url_text(state)}",
        f"本地调试地址: {state.local_url or '(unknown)'}",
    ]
    if state.lan_url:
        lines.append(f"局域网地址: {state.lan_url}")
    lines.extend(
        [
            "",
            f"有效期: {_remaining_text(state)}",
            "电脑唤醒: 已开启" if state.wake_lock_id else "电脑唤醒: 未开启",
        ]
    )
    if state.message and not state.public_url:
        lines.extend(["", state.message])
    lines.extend(["", "查看状态: /deploy status", "关闭预览: /deploy stop"])
    return "\n".join(lines)


def _format_status(state: PreviewDeployState | None, workspace: Path) -> str:
    if state is None or not state.status:
        return "当前没有运行中的临时预览。"
    if state.status in {"stopped", "expired", "stale"}:
        label = {
            "stopped": "临时预览已关闭",
            "expired": "临时预览已过期",
            "stale": "临时预览已失效",
        }.get(state.status, "临时预览不可用")
        lines = [label]
        if state.message:
            lines.extend(["", state.message])
        lines.extend(["", "可以重新执行 /deploy 创建新的临时预览。"])
        return "\n".join(lines)
    if state.status == "running":
        title = "外部临时预览运行中" if state.public_url else "本地临时预览运行中"
    else:
        title = "外部临时预览异常" if state.public_url else "本地临时预览异常"
    lines = [
        title,
        "",
        f"项目: {state.project_name or state.project_slug}",
        f"目标路径: {_target_path_text(state, workspace)}",
        f"外部链接: {_public_url_text(state)}",
        f"本地调试地址: {state.local_url or '(unknown)'}",
    ]
    if state.lan_url:
        lines.append(f"局域网地址: {state.lan_url}")
    lines.extend(
        [
            f"剩余时间: {_remaining_text(state)}",
            f"最近检查: {'正常' if state.status == 'running' else state.message or state.status}",
        ]
    )
    lines.extend(["", "关闭预览: /deploy stop"])
    return "\n".join(lines)


def _format_existing_preview(
    state: PreviewDeployState,
    args: _DeployArgs,
    workspace: Path,
) -> str:
    lines = [
        "当前已有临时预览在运行",
        "",
        f"正在预览: {state.project_name or state.project_slug}",
        f"目标路径: {_target_path_text(state, workspace)}",
        f"外部链接: {_public_url_text(state)}",
    ]
    if state.lan_url:
        lines.append(f"局域网地址: {state.lan_url}")
    lines.extend([f"剩余时间: {_remaining_text(state)}", ""])
    requested = args.path or args.target
    if requested:
        replace_command = "/deploy replace"
        if args.path:
            replace_command += f" --path {shlex.quote(args.path)}"
        else:
            replace_command += f" {shlex.quote(args.target)}"
        lines.extend(
            [
                "如果要替换当前预览，请使用:",
                replace_command,
            ]
        )
    else:
        lines.extend(["查看状态: /deploy status", "关闭预览: /deploy stop"])
    return "\n".join(lines)


def _format_stopped(state: PreviewDeployState, workspace: Path) -> str:
    return "\n".join(
        [
            "临时预览已关闭",
            "",
            f"项目: {state.project_name or state.project_slug}",
            f"目标路径: {_target_path_text(state, workspace)}",
        ]
    )


def _target_path_text(state: PreviewDeployState, workspace: Path) -> str:
    if state.target_kind == "external_url":
        return "N/A（外部 URL）"
    return display_path(state.target_path, workspace)


def _public_url_text(state: PreviewDeployState) -> str:
    if state.public_url:
        return state.public_url
    if state.message:
        return "暂不可用"
    return "未创建"


def _remaining_text(state: PreviewDeployState) -> str:
    expires_at = parse_datetime(state.expires_at)
    if expires_at is None:
        return "unknown"
    remaining = int((expires_at - datetime.now().astimezone()).total_seconds())
    if remaining <= 0:
        return "已到期"
    hours, rest = divmod(remaining, 3600)
    minutes = rest // 60
    if hours > 0 and minutes > 0:
        return f"{hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h"
    if minutes > 0:
        return f"{minutes}m"
    return f"{remaining}s"


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
