from __future__ import annotations

import unittest
import tempfile
import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from deepmate.local import (
    LocalModelCancellationToken,
    LocalModelInstallResult,
    LocalModelStateStore,
    OllamaLocalRuntime,
    local_model_by_id,
    ollama_api_url_from_provider_base_url,
)
from deepmate.local.ollama import LocalModelStatus
from deepmate.providers import ModelResponse


class LocalRuntimeTests(unittest.TestCase):
    def test_running_service_with_installed_model_needs_no_cli_or_download(self) -> None:
        preset = local_model_by_id("qwen3-local")
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                requests.append(self.path)
                payload = {"version": "fixture"} if self.path == "/api/version" else {"models": [{"name": preset.runtime_name}]}
                self.reply(payload)

            def do_POST(self):
                requests.append(self.path)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body["model"] != preset.runtime_name:
                    self.send_error(400)
                    return
                self.reply({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})

            def reply(self, payload):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as tmp:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                runtime = OllamaLocalRuntime(api_url=f"http://127.0.0.1:{server.server_port}")
                store = LocalModelStateStore(tmp)
                with patch("deepmate.local.ollama.shutil.which", return_value=None):
                    self.assertTrue(runtime.status().available)
                    self.assertEqual(runtime.prepared_model(start_server=False), preset)
                    result = runtime.prepare_model(preset, state_store=store, install_missing_runtime=False)
                self.assertTrue(result.ok)
                self.assertTrue(store.load().is_ready())
                self.assertEqual(requests.count("/v1/chat/completions"), 1)
                self.assertEqual(set(requests), {"/api/version", "/api/tags", "/v1/chat/completions"})
            finally:
                server.shutdown()
                worker.join()

    def test_prepare_model_skips_alias_when_runtime_name_matches_ollama_ref(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()

        with (
            patch.object(
                runtime,
                "ensure_ready",
                return_value=LocalModelStatus(
                    available=True,
                    installed=True,
                    running=True,
                ),
            ),
            patch.object(runtime, "has_model", return_value=True),
            patch.object(runtime, "has_runtime_model", return_value=False),
            patch.object(
                runtime,
                "health_check",
                return_value=LocalModelInstallResult(
                    ok=True,
                    preset=preset,
                    message="verified",
                ),
            ),
            patch.object(runtime, "has_runtime_model") as has_runtime_model,
            patch.object(runtime, "create_runtime_alias") as create_runtime_alias,
        ):
            result = runtime.prepare_model(preset)

        self.assertTrue(result.ok)
        has_runtime_model.assert_not_called()
        create_runtime_alias.assert_not_called()

    def test_create_runtime_alias_uses_ollama_copy(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        alias_preset = type(preset)(
            id=preset.id,
            label=preset.label,
            short_label=preset.short_label,
            runtime_name="deepmate-qwen3-4b",
            ollama_ref=preset.ollama_ref,
            size_label=preset.size_label,
            min_memory_gb=preset.min_memory_gb,
            recommended_memory_gb=preset.recommended_memory_gb,
            effective_context_tokens=preset.effective_context_tokens,
            response_token_reserve=preset.response_token_reserve,
            safety_margin_tokens=preset.safety_margin_tokens,
            max_tokens=preset.max_tokens,
            description=preset.description,
        )

        with patch("deepmate.local.ollama.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""

            result = OllamaLocalRuntime().create_runtime_alias(alias_preset)

        self.assertTrue(result.ok)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command, ("ollama", "cp", "qwen3:4b", "deepmate-qwen3-4b"))

    def test_prepare_model_guides_user_when_ollama_is_missing_without_brew(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()

        with (
            patch.object(
                runtime,
                "ensure_ready",
                return_value=LocalModelStatus(
                    available=False,
                    installed=False,
                    running=False,
                ),
            ),
            patch("deepmate.local.ollama.platform.system", return_value="Darwin"),
            patch("deepmate.local.ollama.shutil.which", return_value=None),
        ):
            result = runtime.prepare_model(preset)

        self.assertFalse(result.ok)
        self.assertIn("Ollama", result.message)
        self.assertIn("https://ollama.com/download", result.message)

    def test_prepare_model_can_skip_runtime_install(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()

        with (
            patch.object(
                runtime,
                "ensure_ready",
                return_value=LocalModelStatus(
                    available=False,
                    installed=False,
                    running=False,
                ),
            ),
            patch.object(runtime, "install_runtime") as install_runtime,
        ):
            result = runtime.prepare_model(preset, install_missing_runtime=False)

        self.assertFalse(result.ok)
        self.assertIn("Ollama", result.message)
        install_runtime.assert_not_called()

    def test_prepare_model_installs_ollama_with_homebrew_when_available(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()
        statuses = [
            LocalModelStatus(available=False, installed=False, running=False),
            LocalModelStatus(available=True, installed=True, running=True),
        ]

        with (
            patch.object(runtime, "ensure_ready", side_effect=statuses),
            patch.object(runtime, "has_model", return_value=True),
            patch.object(
                runtime,
                "health_check",
                return_value=LocalModelInstallResult(
                    ok=True,
                    preset=preset,
                    message="verified",
                ),
            ),
            patch("deepmate.local.ollama.platform.system", return_value="Darwin"),
            patch("deepmate.local.ollama.shutil.which", return_value="/opt/homebrew/bin/brew"),
            patch("deepmate.local.ollama.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            run.return_value.stdout = ""
            run.return_value.stderr = ""

            result = runtime.prepare_model(preset)

        self.assertTrue(result.ok)
        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0],
            ("/opt/homebrew/bin/brew", "install", "ollama"),
        )

    def test_prepare_model_emits_progress_and_fails_closed_when_health_check_fails(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()
        progress = []

        with (
            patch.object(
                runtime,
                "ensure_ready",
                return_value=LocalModelStatus(
                    available=True,
                    installed=True,
                    running=True,
                ),
            ),
            patch.object(runtime, "has_model", return_value=True),
            patch.object(runtime, "has_runtime_model", return_value=False),
            patch.object(
                runtime,
                "health_check",
                return_value=LocalModelInstallResult(
                    ok=False,
                    preset=preset,
                    message="failed",
                ),
            ),
            patch.object(runtime, "_start_server") as start_server,
        ):
            result = runtime.prepare_model(preset, progress=progress.append)

        self.assertFalse(result.ok)
        self.assertTrue(any(item.stage == "checking" for item in progress))
        self.assertTrue(any(item.stage == "verifying" for item in progress))
        start_server.assert_called_once()

    def test_prepare_model_fails_closed_when_installed_model_health_check_fails(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()

        with (
            patch.object(
                runtime,
                "ensure_ready",
                return_value=LocalModelStatus(
                    available=True,
                    installed=True,
                    running=True,
                ),
            ),
            patch.object(runtime, "has_model", return_value=True),
            patch.object(runtime, "has_runtime_model", return_value=True),
            patch.object(
                runtime,
                "health_check",
                return_value=LocalModelInstallResult(
                    ok=False,
                    preset=preset,
                    message="failed",
                ),
            ),
            patch.object(runtime, "_start_server") as start_server,
        ):
            result = runtime.prepare_model(preset)

        self.assertFalse(result.ok)
        self.assertIn("连续两次验证失败", result.message)
        start_server.assert_called_once()

    def test_pull_timeout_kills_and_waits_for_process(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)

        class Stdout:
            closed = False
            released = threading.Event()

            def readline(self):
                self.released.wait()
                return ""

            def close(self):
                self.closed = True
                self.released.set()

        class Process:
            stdout = Stdout()
            returncode = None
            killed = False
            wait_calls = 0

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.wait_calls += 1
                self.returncode = -9
                return self.returncode

        process = Process()
        runtime = OllamaLocalRuntime()

        with (
            patch("deepmate.local.ollama.subprocess.Popen", return_value=process),
            patch(
                "deepmate.local.ollama.time.monotonic",
                side_effect=(0.0, 3601.0),
            ),
        ):
            result = runtime.pull(preset)

        self.assertFalse(result.ok)
        self.assertTrue(process.killed)
        self.assertEqual(process.wait_calls, 1)
        self.assertTrue(process.stdout.closed)

    def test_prepare_model_persists_prepare_state(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalModelStateStore(Path(tmp))

            with (
                patch.object(
                    runtime,
                    "ensure_ready",
                    return_value=LocalModelStatus(
                        available=True,
                        installed=True,
                        running=True,
                    ),
                ),
                patch.object(runtime, "has_model", return_value=True),
                patch.object(
                    runtime,
                    "health_check",
                    return_value=LocalModelInstallResult(
                        ok=True,
                        preset=preset,
                        message="verified",
                    ),
                ),
            ):
                result = runtime.prepare_model(preset, state_store=store)

            saved = store.load()

        self.assertTrue(result.ok)
        self.assertIsNotNone(saved)
        self.assertEqual(saved.model_id, "qwen3-local")
        self.assertEqual(saved.status, "ready")
        self.assertEqual(saved.stage, "ready")

    def test_local_model_state_concurrent_saves_remain_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalModelStateStore(Path(tmp))
            failures: list[Exception] = []

            def save(index: int) -> None:
                try:
                    store.record(
                        model_id=f"model-{index}",
                        stage="checking",
                        message=f"attempt {index}",
                    )
                except Exception as exc:  # pragma: no cover - assertion captures it.
                    failures.append(exc)

            workers = [threading.Thread(target=save, args=(index,)) for index in range(20)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            saved = store.load()

        self.assertEqual(failures, [])
        self.assertIsNotNone(saved)
        self.assertTrue(saved.model_id.startswith("model-") if saved else False)

    def test_disk_preflight_rejects_insufficient_space(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        usage = type("DiskUsage", (), {"free": 128 * 1024 * 1024})()

        with patch("deepmate.local.ollama.shutil.disk_usage", return_value=usage):
            result = OllamaLocalRuntime().disk_preflight(preset)

        self.assertIsNotNone(result)
        self.assertFalse(result.ok if result else True)
        self.assertIn("磁盘空间不足", result.message if result else "")

    def test_prepare_model_resumes_incomplete_state(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()
        progress = []
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalModelStateStore(Path(tmp))
            store.record(
                model_id=preset.id,
                stage="downloading",
                message="partial",
                status="failed",
                failure_kind="download_failed",
            )
            with (
                patch.object(
                    runtime,
                    "ensure_ready",
                    return_value=LocalModelStatus(True, True, True),
                ),
                patch.object(runtime, "has_model", return_value=True),
                patch.object(
                    runtime,
                    "health_check",
                    return_value=LocalModelInstallResult(True, preset, "verified"),
                ),
            ):
                result = runtime.prepare_model(
                    preset,
                    progress=progress.append,
                    state_store=store,
                )

        self.assertTrue(result.ok)
        self.assertEqual(progress[0].stage, "resuming")

    def test_pull_cancellation_kills_process_and_preserves_resume_state(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        token = LocalModelCancellationToken()
        token.cancel()

        class Process:
            stdout = type(
                "Stdout",
                (),
                {"readline": lambda self: "", "close": lambda self: None},
            )()
            returncode = None
            killed = False

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                self.returncode = -9
                return self.returncode

        process = Process()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalModelStateStore(Path(tmp))
            with patch("deepmate.local.ollama.subprocess.Popen", return_value=process):
                result = OllamaLocalRuntime().pull(
                    preset,
                    state_store=store,
                    cancellation_token=token,
                )
            saved = store.load()

        self.assertFalse(result.ok)
        self.assertTrue(process.killed)
        self.assertEqual(saved.failure_kind if saved else "", "cancelled")

    def test_concurrent_prepare_calls_share_completed_result(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        runtime = OllamaLocalRuntime()
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalModelStateStore(Path(tmp))
            verified = LocalModelInstallResult(True, preset, "verified")
            pulled = LocalModelInstallResult(True, preset, "downloaded")
            with (
                patch.object(
                    runtime,
                    "ensure_ready",
                    return_value=LocalModelStatus(True, True, True),
                ),
                patch.object(runtime, "has_model", side_effect=(False, True)),
                patch.object(runtime, "disk_preflight", return_value=None),
                patch.object(runtime, "pull", return_value=pulled) as pull,
                patch.object(runtime, "health_check", return_value=verified) as health,
            ):
                workers = [
                    threading.Thread(
                        target=lambda: results.append(
                            runtime.prepare_model(preset, state_store=store)
                        )
                    )
                    for _ in range(2)
                ]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.ok for result in results))
        pull.assert_called_once()
        self.assertEqual(health.call_count, 2)

    def test_prepared_model_prefers_recommended_installed_model(self) -> None:
        runtime = OllamaLocalRuntime()

        with (
            patch.object(
                runtime,
                "ensure_ready",
                return_value=LocalModelStatus(
                    available=True,
                    installed=True,
                    running=True,
                ),
            ),
            patch.object(runtime, "installed_models", return_value=("qwen3:4b", "qwen3:8b")),
            patch("deepmate.local.presets.platform.machine", return_value="arm64"),
            patch("deepmate.local.presets._memory_gb", return_value=24),
        ):
            preset = runtime.prepared_model()

        self.assertIsNotNone(preset)
        self.assertEqual(preset.id, "qwen3-balanced")

    def test_prepared_model_status_mode_does_not_start_server(self) -> None:
        runtime = OllamaLocalRuntime()

        with (
            patch.object(
                runtime,
                "status",
                return_value=LocalModelStatus(
                    available=False,
                    installed=True,
                    running=False,
                ),
            ),
            patch.object(runtime, "ensure_ready") as ensure_ready,
        ):
            preset = runtime.prepared_model(start_server=False)

        self.assertIsNone(preset)
        ensure_ready.assert_not_called()

    def test_ollama_runtime_rejects_non_http_api_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "http"):
            OllamaLocalRuntime(api_url="file:///tmp/ollama.sock")

    def test_health_check_uses_custom_api_url_provider_base(self) -> None:
        preset = local_model_by_id("qwen3-local")
        self.assertIsNotNone(preset)
        created = []

        class FakeProvider:
            def __init__(self, base_url: str, api_key: str, timeout: int) -> None:
                created.append((base_url, api_key, timeout))

            def complete(self, _request):
                return ModelResponse(content="ok")

        runtime = OllamaLocalRuntime(api_url="http://127.0.0.1:11555")
        with patch("deepmate.local.ollama.ChatCompletionsProvider", FakeProvider):
            result = runtime.health_check(preset)

        self.assertTrue(result.ok)
        self.assertEqual(created, [("http://127.0.0.1:11555/v1", "ollama", 20)])
        self.assertEqual(result.provider_base_url, "http://127.0.0.1:11555/v1")

    def test_provider_base_url_converts_to_ollama_api_url(self) -> None:
        self.assertEqual(
            ollama_api_url_from_provider_base_url("http://127.0.0.1:11555/v1"),
            "http://127.0.0.1:11555",
        )
        self.assertEqual(
            ollama_api_url_from_provider_base_url("http://127.0.0.1:11555"),
            "http://127.0.0.1:11555",
        )

    def test_get_json_rejects_non_json_response(self) -> None:
        class FakeHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        class FakeResponse:
            headers = FakeHeaders({"Content-Type": "text/plain"})

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _size=-1):
                return b"ok"

        runtime = OllamaLocalRuntime()
        with patch("deepmate.local.ollama.urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(ValueError, "JSON"):
                runtime._get_json("/api/version", timeout=1)

    def test_get_json_rejects_oversized_response(self) -> None:
        class FakeHeaders(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        class FakeResponse:
            headers = FakeHeaders({"Content-Type": "application/json"})

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, size=-1):
                return b" " * size

        runtime = OllamaLocalRuntime()
        with patch("deepmate.local.ollama.urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(ValueError, "too large"):
                runtime._get_json("/api/version", timeout=1)


if __name__ == "__main__":
    unittest.main()
