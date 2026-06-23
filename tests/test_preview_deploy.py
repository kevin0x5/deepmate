from __future__ import annotations

import os
import base64
import socket
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import deepmate.preview_deploy.commands as deploy_commands
from deepmate.preview_deploy import PreviewDeployStore, handle_deploy_command
from deepmate.preview_deploy.health import HealthCheck
from deepmate.preview_deploy.state import PreviewDeployState
from deepmate.preview_deploy.tunnel import (
    PreviewLease,
    PreviewTunnel,
    PreviewTunnelError,
    PreviewTunnelProvider,
    PreviewTunnelStatus,
    _forward_tunnel_request,
)


def _local_socket_available() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except OSError:
        return False
    return True


_REQUIRES_LOCAL_SOCKET = unittest.skipUnless(
    _local_socket_available(),
    "local socket binding is not available in this environment",
)


class _FakeTunnelProvider(PreviewTunnelProvider):
    def __init__(self, *, fail: str = "") -> None:
        self.fail = fail
        self.created: list[tuple[str, int]] = []
        self.opened: list[tuple[str, str]] = []
        self.closed: list[str] = []

    def create_lease(self, project_slug: str, ttl_seconds: int) -> PreviewLease:
        if self.fail:
            raise PreviewTunnelError(self.fail)
        self.created.append((project_slug, ttl_seconds))
        return PreviewLease(
            lease_id=f"lease_{project_slug}",
            public_url=f"https://{project_slug}.preview.deepmate.dev",
            tunnel_url="https://relay.example/tunnel",
            tunnel_token="token",
        )

    def open_tunnel(self, local_url: str, lease: PreviewLease) -> PreviewTunnel:
        self.opened.append((local_url, lease.lease_id))
        return PreviewTunnel(
            lease=lease,
            public_url=lease.public_url,
            tunnel_pid=0,
            process_ids=(),
            message="fake relay tunnel opened",
        )

    def close_tunnel(self, lease_id: str) -> None:
        self.closed.append(lease_id)

    def status(self, lease_id: str) -> PreviewTunnelStatus:
        return PreviewTunnelStatus(ok=True, status="running")


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.waited = False
        self.stderr = None

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


class _ProcessTunnelProvider(_FakeTunnelProvider):
    def __init__(self, process: _FakeProcess) -> None:
        super().__init__()
        self.process = process

    def open_tunnel(self, local_url: str, lease: PreviewLease) -> PreviewTunnel:
        self.opened.append((local_url, lease.lease_id))
        return PreviewTunnel(
            lease=lease,
            public_url=lease.public_url,
            tunnel_pid=self.process.pid,
            process_ids=(self.process.pid,),
            process=self.process,
            message="fake relay tunnel opened",
        )


class PreviewDeployTests(unittest.TestCase):
    def test_static_preview_rejects_target_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            data_dir = root / "var"
            outside = root / "outside.html"
            workspace.mkdir()
            outside.write_text("<!doctype html><title>Outside</title>", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inside the workspace"):
                handle_deploy_command(
                    f"/deploy --path {outside}",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=_FakeTunnelProvider(),
                )

    def test_stop_state_does_not_kill_unregistered_process_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "var"
            store = PreviewDeployStore.in_data_dir(data_dir)
            state = PreviewDeployState(
                status="running",
                project_slug="demo",
                target_path=str(root / "index.html"),
                target_kind="static_file",
                local_url="http://127.0.0.1:8000/",
                local_service_owner="deepmate",
                supervisor_pid=12345,
                process_ids=(23456,),
            )
            killed: list[int] = []

            with patch("deepmate.preview_deploy.commands.os.kill") as kill:
                kill.side_effect = lambda pid, _signal: killed.append(pid)
                deploy_commands._stop_state(store, state)

            self.assertEqual(killed, [])

    def test_stop_state_terminates_registered_tunnel_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "var"
            store = PreviewDeployStore.in_data_dir(data_dir)
            process = _FakeProcess(34567)
            deploy_commands._SUPERVISOR_PROCESSES[process.pid] = process  # type: ignore[assignment]
            state = PreviewDeployState(
                status="running",
                project_slug="demo",
                target_path=str(root / "index.html"),
                target_kind="static_file",
                local_url="http://127.0.0.1:8000/",
                local_service_owner="deepmate",
                tunnel_pid=process.pid,
                process_ids=(process.pid,),
            )

            killed: list[tuple[int, int]] = []

            def fake_kill(pid: int, sig: int) -> None:
                killed.append((pid, sig))
                if sig == deploy_commands.signal.SIGTERM:
                    raise ProcessLookupError

            with patch("deepmate.preview_deploy.commands.os.kill", side_effect=fake_kill):
                deploy_commands._stop_state(store, state)

            self.assertEqual(killed, [(process.pid, deploy_commands.signal.SIGTERM)])
            self.assertTrue(process.waited)
            self.assertNotIn(process.pid, deploy_commands._SUPERVISOR_PROCESSES)

    @_REQUIRES_LOCAL_SOCKET
    def test_deploy_static_html_status_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "pricing-page"
            data_dir = root / "var"
            workspace.mkdir()
            (workspace / "report.html").write_text(
                "<!doctype html><title>Preview</title><h1>Deepmate Preview</h1>",
                encoding="utf-8",
            )
            provider = _FakeTunnelProvider()
            with (
                patch.dict(os.environ, {"DEEPMATE_PREVIEW_WAKE": "0"}),
                patch(
                    "deepmate.preview_deploy.commands.check_url",
                    side_effect=lambda url, **kwargs: HealthCheck(
                        True,
                        status_code=200,
                        message="ok",
                    ),
                ),
            ):
                try:
                    created = handle_deploy_command(
                        "/deploy report.html --ttl 5m",
                        workspace=workspace,
                        data_dir=data_dir,
                        owner_session_id="session_1",
                        owner_session_workspace=workspace,
                        tunnel_provider=provider,
                    )
                    state = PreviewDeployStore.in_data_dir(data_dir).load()
                    self.assertIsNotNone(state)
                    assert state is not None
                    self.assertIn("外部临时预览已开启", created)
                    self.assertIn("外部链接: https://report-", created)
                    self.assertIn("本地调试地址: http://127.0.0.1:", created)
                    self.assertEqual(state.status, "running")
                    self.assertEqual(state.target_kind, "static_file")
                    self.assertEqual(state.provider, "deepmate-preview-relay")
                    self.assertTrue(state.public_url.endswith(".preview.deepmate.dev"))
                    self.assertTrue(state.lease_id.startswith("lease_report-"))
                    self.assertTrue(state.local_url.endswith("/report.html"))
                    body = urllib.request.urlopen(
                        state.local_url,
                        timeout=2,
                    ).read()
                    self.assertIn(b"Deepmate Preview", body)

                    status = handle_deploy_command(
                        "/deploy status",
                        workspace=workspace,
                        data_dir=data_dir,
                        tunnel_provider=provider,
                    )
                    self.assertIn("外部临时预览运行中", status)
                    self.assertIn("report.html", status)
                finally:
                    stopped = handle_deploy_command(
                        "/deploy stop",
                        workspace=workspace,
                        data_dir=data_dir,
                        tunnel_provider=provider,
                    )
            self.assertIn("临时预览已关闭", stopped)
            self.assertTrue(provider.closed)

    def test_format_created_and_status_show_lan_url_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            state = PreviewDeployState(
                status="running",
                project_name="demo",
                project_slug="demo",
                target_path=str(workspace / "index.html"),
                target_kind="static_file",
                local_url="http://127.0.0.1:5173/",
                lan_url="http://192.0.2.20:5173/",
                provider="local",
                started_at="2026-06-20T00:00:00+00:00",
                expires_at="2099-01-01T00:00:00+00:00",
            )

            created = deploy_commands._format_created(state, workspace)
            status = deploy_commands._format_status(state, workspace)

        self.assertIn("本地调试地址: http://127.0.0.1:5173/", created)
        self.assertIn("局域网地址: http://192.0.2.20:5173/", created)
        self.assertIn("局域网地址: http://192.0.2.20:5173/", status)

    def test_external_preview_registers_tunnel_process_for_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "var"
            store = PreviewDeployStore.in_data_dir(data_dir)
            process = _FakeProcess(45678)
            provider = _ProcessTunnelProvider(process)
            state = PreviewDeployState(
                status="running",
                project_slug="demo",
                target_path=str(root / "index.html"),
                target_kind="static_file",
                local_url="http://127.0.0.1:8000/",
                provider="local",
            )

            with patch(
                "deepmate.preview_deploy.commands._check_public_url_until",
                return_value=HealthCheck(True, status_code=200, message="ok"),
            ):
                updated = deploy_commands._open_external_preview(
                    store=store,
                    state=state,
                    ttl_seconds=300,
                    tunnel_provider=provider,
                )

            self.assertEqual(updated.tunnel_pid, process.pid)
            self.assertIs(deploy_commands._SUPERVISOR_PROCESSES[process.pid], process)

            deploy_commands._SUPERVISOR_PROCESSES.pop(process.pid, None)

    @_REQUIRES_LOCAL_SOCKET
    def test_deploy_does_not_replace_existing_preview_without_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "project"
            data_dir = root / "var"
            workspace.mkdir()
            (workspace / "report.html").write_text("one", encoding="utf-8")
            dist = workspace / "dist"
            dist.mkdir()
            (dist / "index.html").write_text("two", encoding="utf-8")
            provider = _FakeTunnelProvider()
            with (
                patch.dict(os.environ, {"DEEPMATE_PREVIEW_WAKE": "0"}),
                patch(
                    "deepmate.preview_deploy.commands.check_url",
                    return_value=HealthCheck(True, status_code=200, message="ok"),
                ),
            ):
                try:
                    handle_deploy_command(
                        "/deploy report.html --ttl 5m",
                        workspace=workspace,
                        data_dir=data_dir,
                        tunnel_provider=provider,
                    )
                    conflict = handle_deploy_command(
                        "/deploy dist/",
                        workspace=workspace,
                        data_dir=data_dir,
                        tunnel_provider=provider,
                    )
                finally:
                    handle_deploy_command(
                        "/deploy stop",
                        workspace=workspace,
                        data_dir=data_dir,
                        tunnel_provider=provider,
                    )
            self.assertIn("当前已有临时预览在运行", conflict)
            self.assertIn("/deploy replace dist/", conflict)

    def test_deploy_external_url_uses_external_target_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "project"
            data_dir = root / "var"
            workspace.mkdir()

            with patch(
                "deepmate.preview_deploy.commands.check_url",
                return_value=HealthCheck(True, status_code=200, message="ok"),
            ):
                provider = _FakeTunnelProvider()
                created = handle_deploy_command(
                    "/deploy --url http://127.0.0.1:5173 --name demo",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )
                status = handle_deploy_command(
                    "/deploy status",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )
                running_state = PreviewDeployStore.in_data_dir(data_dir).load()
                stopped = handle_deploy_command(
                    "/deploy stop",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )

            store = PreviewDeployStore.in_data_dir(data_dir)
            state = store.load()
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.target_kind, "external_url")
            self.assertEqual(state.provider, "deepmate-preview-relay")
            self.assertRegex(
                state.public_url,
                r"^https://demo-[a-z2-7]{6}\.preview\.deepmate\.dev$",
            )
            self.assertIn("目标路径: N/A（外部 URL）", created)
            self.assertIn("本地调试地址: http://127.0.0.1:5173", created)
            self.assertIn("目标路径: N/A（外部 URL）", status)
            self.assertIn("目标路径: N/A（外部 URL）", stopped)

    def test_tunnel_worker_blocks_mutating_methods(self) -> None:
        result = _forward_tunnel_request(
            "http://127.0.0.1:3000",
            {
                "request_id": "req_1",
                "method": "POST",
                "path": "/api/delete",
                "body_base64": base64.b64encode(b"{}").decode("ascii"),
            },
        )

        self.assertEqual(result["request_id"], "req_1")
        self.assertEqual(result["status_code"], 405)
        body = base64.b64decode(str(result["body_base64"])).decode("utf-8")
        self.assertIn("blocked unsupported method", body)

    @_REQUIRES_LOCAL_SOCKET
    def test_relay_failure_keeps_local_preview_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "project"
            data_dir = root / "var"
            workspace.mkdir()
            (workspace / "index.html").write_text("preview", encoding="utf-8")
            provider = _FakeTunnelProvider(fail="relay unavailable")

            with (
                patch.dict(os.environ, {"DEEPMATE_PREVIEW_WAKE": "0"}),
                patch(
                    "deepmate.preview_deploy.commands.check_url",
                    return_value=HealthCheck(True, status_code=200, message="ok"),
                ),
            ):
                created = handle_deploy_command(
                    "/deploy --ttl 5m",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )
                status = handle_deploy_command(
                    "/deploy status",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )
                running_state = PreviewDeployStore.in_data_dir(data_dir).load()
                stopped = handle_deploy_command(
                    "/deploy stop",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )
                status_after_stop = handle_deploy_command(
                    "/deploy status",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )

            self.assertIsNotNone(running_state)
            assert running_state is not None
            self.assertEqual(running_state.status, "running")
            self.assertEqual(running_state.provider, "local")
            self.assertFalse(running_state.public_url)
            self.assertIn("本地临时预览已开启", created)
            self.assertIn("公开分享链接暂不可用", created)
            self.assertIn("本地调试地址: http://127.0.0.1:", created)
            self.assertIn("本地临时预览运行中", status)
            self.assertIn("外部链接: 暂不可用", status)
            self.assertIn("临时预览已关闭", stopped)
            self.assertIn("临时预览已关闭", status_after_stop)
            self.assertNotIn("本地预览已继续运行", status_after_stop)

    @_REQUIRES_LOCAL_SOCKET
    def test_exited_supervisor_is_marked_stale_and_process_ref_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "project"
            data_dir = root / "var"
            workspace.mkdir()
            (workspace / "index.html").write_text("preview", encoding="utf-8")

            provider = _FakeTunnelProvider()
            with (
                patch.dict(os.environ, {"DEEPMATE_PREVIEW_WAKE": "0"}),
                patch(
                    "deepmate.preview_deploy.commands.check_url",
                    return_value=HealthCheck(True, status_code=200, message="ok"),
                ),
            ):
                handle_deploy_command(
                    "/deploy --ttl 5m",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )
                state = PreviewDeployStore.in_data_dir(data_dir).load()
                self.assertIsNotNone(state)
                assert state is not None
                process = deploy_commands._SUPERVISOR_PROCESSES[state.supervisor_pid]
                process.kill()
                process.wait(timeout=3)

                status = handle_deploy_command(
                    "/deploy status",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )

            self.assertIn("临时预览已失效", status)
            self.assertNotIn(state.supervisor_pid, deploy_commands._SUPERVISOR_PROCESSES)

    @_REQUIRES_LOCAL_SOCKET
    def test_static_preview_expires_by_wall_clock_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "project"
            data_dir = root / "var"
            workspace.mkdir()
            (workspace / "index.html").write_text("preview", encoding="utf-8")

            provider = _FakeTunnelProvider()
            with (
                patch.dict(os.environ, {"DEEPMATE_PREVIEW_WAKE": "0"}),
                patch(
                    "deepmate.preview_deploy.commands.check_url",
                    return_value=HealthCheck(True, status_code=200, message="ok"),
                ),
            ):
                handle_deploy_command(
                    "/deploy --ttl 1s",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )
                time.sleep(1.4)
                status = handle_deploy_command(
                    "/deploy status",
                    workspace=workspace,
                    data_dir=data_dir,
                    tunnel_provider=provider,
                )

            self.assertIn("临时预览已过期", status)

    def test_supervisor_start_failure_includes_stderr(self) -> None:
        class _FailedProcess:
            pid = 99_999_999
            stderr = object()

            def poll(self):
                return 1

            def communicate(self, timeout=0):
                return None, "bind failed: address already in use\n"

            def wait(self, timeout=0):
                return 1

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "index.html").write_text("preview", encoding="utf-8")
            store = PreviewDeployStore.in_data_dir(workspace / "var")
            target = deploy_commands._Target(
                target_path=workspace / "index.html",
                target_kind="static_file",
                serve_root=workspace,
                entry_path="index.html",
            )
            process = _FailedProcess()
            try:
                with patch(
                    "deepmate.preview_deploy.commands.subprocess.Popen",
                    return_value=process,
                ):
                    with self.assertRaisesRegex(RuntimeError, "address already in use"):
                        deploy_commands._start_static_preview(
                            store=store,
                            target=target,
                            project_name="demo",
                            project_slug="demo-123",
                            ttl_seconds=60,
                            owner_session_id="",
                            owner_session_workspace=workspace,
                        )
            finally:
                deploy_commands._SUPERVISOR_PROCESSES.pop(process.pid, None)

    def test_deploy_rejects_ttl_over_24h(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "index.html").write_text("ok", encoding="utf-8")
            with self.assertRaises(ValueError):
                handle_deploy_command(
                    "/deploy --ttl 25h",
                    workspace=workspace,
                    data_dir=workspace / "var",
                )

    def test_deploy_no_default_target_explains_auto_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            with self.assertRaisesRegex(ValueError, "Deepmate 已自动检查") as raised:
                handle_deploy_command(
                    "/deploy",
                    workspace=workspace,
                    data_dir=workspace / "var",
                )

        message = str(raised.exception)
        self.assertIn("dist/index.html", message)
        self.assertIn("/deploy --url http://127.0.0.1:<端口>", message)

    def test_deploy_directory_without_index_has_user_facing_repair_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            output_dir = workspace / "output"
            output_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "没有 index.html") as raised:
                handle_deploy_command(
                    "/deploy output",
                    workspace=workspace,
                    data_dir=workspace / "var",
                )

        self.assertIn("指定单个 HTML 文件", str(raised.exception))

    def test_state_parses_integer_float_pids_without_starting_active_state(self) -> None:
        state = PreviewDeployState.from_record(
            {
                "status": "running",
                "supervisor_pid": 12345.0,
                "process_ids": [12345.0, "12346"],
            }
        )

        self.assertTrue(state.is_active())
        self.assertEqual(state.supervisor_pid, 12345)
        self.assertEqual(state.process_ids, (12345, 12346))
        self.assertFalse(PreviewDeployState(status="starting").is_active())


if __name__ == "__main__":
    unittest.main()
