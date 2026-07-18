"""Ollama-backed local model runtime."""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from deepmate.domain import Message, MessageRole
from deepmate.local.presets import LOCAL_PROVIDER_BASE_URL, LocalModelPreset
from deepmate.local.state import LocalModelStateStore
from deepmate.providers import ChatCompletionsProvider, ModelConversationItem, ModelRequest
from deepmate.storage import file_lock

OLLAMA_API_URL = "http://127.0.0.1:11434"
MAX_OLLAMA_JSON_RESPONSE_BYTES = 2 * 1024 * 1024
LOCAL_MODEL_DISK_HEADROOM_BYTES = 512 * 1024 * 1024
_PREPARE_LOCKS: dict[str, threading.Lock] = {}
_PREPARE_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class LocalModelStatus:
    """Current local runtime status."""

    available: bool
    installed: bool
    running: bool
    version: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class LocalModelInstallResult:
    """Outcome of preparing one local model."""

    ok: bool
    preset: LocalModelPreset
    message: str
    provider_base_url: str = LOCAL_PROVIDER_BASE_URL


@dataclass(frozen=True, slots=True)
class LocalModelProgress:
    """User-facing progress for local model preparation."""

    stage: str
    message: str
    percent: int | None = None


ProgressCallback = Callable[[LocalModelProgress], None]


class LocalModelCancellationToken:
    """Thread-safe cancellation signal for local-model preparation."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class OllamaLocalRuntime:
    """Small wrapper around the Ollama CLI and HTTP API."""

    def __init__(self, *, api_url: str = OLLAMA_API_URL) -> None:
        clean = api_url.rstrip("/")
        parsed = urlparse(clean)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama api_url must be an http(s) URL")
        self._api_url = clean

    def status(self) -> LocalModelStatus:
        """Return whether Ollama is installed and serving requests."""
        installed = shutil.which("ollama") is not None
        version = ""
        running = False
        message = ""
        try:
            raw = self._get_json("/api/version", timeout=1)
            version = str(raw.get("version", "")).strip()
            running = True
        except (
            OSError,
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            if installed:
                message = "Ollama is installed but not running."
            else:
                message = "Ollama is not installed."
            if str(exc).strip():
                message += f" {exc}"
        return LocalModelStatus(
            available=installed and running,
            installed=installed,
            running=running,
            version=version,
            message=message,
        )

    def ensure_ready(self) -> LocalModelStatus:
        """Start Ollama when installed, then return current status."""
        status = self.status()
        if status.running or not status.installed:
            return status
        self._start_server()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = self.status()
            if status.running:
                return status
            time.sleep(0.3)
        return self.status()

    def installed_models(self) -> tuple[str, ...]:
        """Return locally available Ollama model names."""
        try:
            raw = self._get_json("/api/tags", timeout=2)
        except (
            OSError,
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return ()
        models = raw.get("models")
        if not isinstance(models, list):
            return ()
        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        return tuple(names)

    def prepared_model(self, *, start_server: bool = True) -> LocalModelPreset | None:
        """Return an already installed Deepmate local model without downloading."""
        status = self.ensure_ready() if start_server else self.status()
        if not status.installed or not status.running:
            return None
        from deepmate.local.presets import local_model_presets, recommended_local_model

        installed = {name.lower() for name in self.installed_models()}
        prepared: list[LocalModelPreset] = []
        for preset in local_model_presets():
            names = {
                preset.runtime_name.lower(),
                preset.ollama_ref.lower(),
                _ollama_short_name(preset.ollama_ref).lower(),
            }
            if installed.intersection(names):
                prepared.append(preset)
        if not prepared:
            return None
        recommended = recommended_local_model()
        for preset in prepared:
            if preset.id == recommended.id:
                return preset
        return prepared[-1]

    def prepare_model(
        self,
        preset: LocalModelPreset,
        *,
        progress: ProgressCallback | None = None,
        state_store: LocalModelStateStore | None = None,
        install_missing_runtime: bool = True,
        cancellation_token: LocalModelCancellationToken | None = None,
    ) -> LocalModelInstallResult:
        """Prepare one model once across concurrent callers and processes."""
        lock_key = f"{self._api_url}:{preset.id}"
        with _prepare_lock(lock_key):
            lock_context = nullcontext()
            if state_store is not None and state_store.path is not None:
                lock_context = file_lock(
                    state_store.path.with_name(f".{preset.id}.prepare")
                )
            with lock_context:
                return self._prepare_model_unlocked(
                    preset,
                    progress=progress,
                    state_store=state_store,
                    install_missing_runtime=install_missing_runtime,
                    cancellation_token=cancellation_token,
                )

    def _prepare_model_unlocked(
        self,
        preset: LocalModelPreset,
        *,
        progress: ProgressCallback | None = None,
        state_store: LocalModelStateStore | None = None,
        install_missing_runtime: bool = True,
        cancellation_token: LocalModelCancellationToken | None = None,
    ) -> LocalModelInstallResult:
        """Ensure Ollama is running and the requested model is available."""
        previous = state_store.load() if state_store is not None else None
        if (
            previous is not None
            and previous.model_id == preset.id
            and previous.is_ready()
        ):
            status = self.ensure_ready()
            if status.running and self.has_model(preset):
                verified = self.health_check(preset)
                if verified.ok:
                    return LocalModelInstallResult(
                        ok=True,
                        preset=preset,
                        message=f"{preset.label} 已就绪。",
                        provider_base_url=self.provider_base_url,
                    )
        if _cancelled(cancellation_token):
            return _cancelled_result(preset, state_store)
        resuming = (
            previous is not None
            and previous.model_id == preset.id
            and previous.is_incomplete()
        )
        _emit_progress(
            progress,
            "resuming" if resuming else "checking",
            (
                f"正在继续上次的本地模型准备 · {preset.short_label}"
                if resuming
                else f"正在检查本地环境 · {preset.short_label}"
            ),
            preset=preset,
            state_store=state_store,
        )
        status = self.ensure_ready()
        if _cancelled(cancellation_token):
            return _cancelled_result(preset, state_store)
        if not status.installed and not install_missing_runtime:
            result = LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=(
                    "需要先安装本地模型运行组件 Ollama：https://ollama.com/download"
                ),
            )
            _record_result(state_store, result)
            return result
        if not status.installed:
            _record_state(
                state_store,
                preset,
                "installing",
                f"正在准备本地模型运行组件 · {preset.short_label}",
            )
            installed = self.install_runtime(preset)
            if not installed.ok:
                _record_result(state_store, installed)
                return installed
            _emit_progress(
                progress,
                "starting",
                f"正在启动本地模型服务 · {preset.short_label}",
                preset=preset,
                state_store=state_store,
            )
            status = self.ensure_ready()
            if _cancelled(cancellation_token):
                return _cancelled_result(preset, state_store)
        if not status.running:
            _record_state(
                state_store,
                preset,
                "failed",
                "本地模型服务暂时没有启动成功，请稍后重试 /local。",
                status="failed",
                failure_kind="service_start_failed",
            )
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message="本地模型服务暂时没有启动成功，请稍后重试 /local。",
            )
        if not self.has_model(preset):
            disk_result = self.disk_preflight(preset)
            if disk_result is not None:
                _record_result(state_store, disk_result)
                return disk_result
            pulled = self.pull(
                preset,
                progress=progress,
                state_store=state_store,
                cancellation_token=cancellation_token,
            )
            if not pulled.ok:
                _record_result(state_store, pulled)
                return pulled
        if _cancelled(cancellation_token):
            return _cancelled_result(preset, state_store)
        if preset.runtime_name != preset.ollama_ref and not self.has_runtime_model(preset):
            created = self.create_runtime_alias(preset)
            if not created.ok:
                _record_result(state_store, created)
                return created
        _emit_progress(
            progress,
            "verifying",
            f"正在验证本地模型 · {preset.short_label}",
            preset=preset,
            state_store=state_store,
        )
        verified = self.health_check(preset)
        if not verified.ok:
            self._start_server()
            time.sleep(0.5)
            verified = self.health_check(preset)
            if not verified.ok:
                result = LocalModelInstallResult(
                    ok=False,
                    preset=preset,
                    message=(
                        f"{preset.label} 已安装，但本地推理接口连续两次验证失败。"
                        "已继续使用当前模型；确认 Ollama 可正常生成回复后再重试 /local。"
                    ),
                )
                _record_state(
                    state_store,
                    preset,
                    "failed",
                    result.message,
                    status="failed",
                    failure_kind="health_check_failed",
                )
                return result
        _record_state(
            state_store,
            preset,
            "ready",
            f"{preset.label} 已就绪。",
            status="ready",
        )
        return LocalModelInstallResult(
            ok=True,
            preset=preset,
            message=f"{preset.label} 已就绪。",
            provider_base_url=self.provider_base_url,
        )

    def install_runtime(self, preset: LocalModelPreset) -> LocalModelInstallResult:
        """Install Ollama through the simplest available macOS path."""
        if platform.system().lower() != "darwin":
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=(
                    "需要先安装本地模型运行组件 Ollama。安装后重新运行 /local 即可继续："
                    "https://ollama.com/download"
                ),
            )
        brew = shutil.which("brew")
        if brew is None:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=(
                    "需要先安装本地模型运行组件 Ollama。请打开安装页完成一次安装，"
                    "然后重新运行 /local：https://ollama.com/download"
                ),
            )
        try:
            result = subprocess.run(
                (brew, "install", "ollama"),
                check=False,
                capture_output=True,
                text=True,
                timeout=20 * 60,
            )
        except subprocess.TimeoutExpired:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message="Ollama 自动安装还没有完成，请稍后重试 /local。",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=f"Ollama 自动安装没有完成：{exc}",
            )
        if result.returncode != 0:
            output = (result.stderr or result.stdout).strip()
            detail = f"：{output}" if output else ""
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=(
                    "Ollama 自动安装没有完成"
                    f"{detail}。也可以打开 https://ollama.com/download 完成安装。"
                ),
            )
        return LocalModelInstallResult(
            ok=True,
            preset=preset,
            message="Ollama 已安装。",
        )

    def has_model(self, preset: LocalModelPreset) -> bool:
        """Return whether the preset's model is already installed."""
        names = {name.lower() for name in self.installed_models()}
        return (
            preset.ollama_ref.lower() in names
            or preset.runtime_name.lower() in names
            or _ollama_short_name(preset.ollama_ref).lower() in names
        )

    def has_runtime_model(self, preset: LocalModelPreset) -> bool:
        """Return whether Deepmate's stable local model name exists."""
        names = {name.lower() for name in self.installed_models()}
        return preset.runtime_name.lower() in names

    def create_runtime_alias(self, preset: LocalModelPreset) -> LocalModelInstallResult:
        """Create Deepmate's stable local model alias from the pulled model."""
        try:
            result = subprocess.run(
                ("ollama", "cp", preset.ollama_ref, preset.runtime_name),
                check=False,
                capture_output=True,
                text=True,
                timeout=10 * 60,
            )
        except FileNotFoundError:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message="需要先安装本地模型运行组件 Ollama：https://ollama.com/download",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=f"{preset.label} 本地别名创建失败：{exc}",
            )
        if result.returncode != 0:
            output = (result.stderr or result.stdout).strip()
            detail = f"：{output}" if output else ""
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=f"{preset.label} 本地别名创建失败{detail}",
            )
        return LocalModelInstallResult(
            ok=True,
            preset=preset,
            message=f"{preset.label} 已注册。",
        )

    def disk_preflight(
        self,
        preset: LocalModelPreset,
    ) -> LocalModelInstallResult | None:
        """Return a failure when the Ollama model volume lacks safe headroom."""
        required = _preset_download_bytes(preset) + LOCAL_MODEL_DISK_HEADROOM_BYTES
        model_root = Path(os.environ.get("OLLAMA_MODELS", "")).expanduser()
        if not str(model_root).strip() or str(model_root) == ".":
            model_root = Path.home() / ".ollama" / "models"
        probe = model_root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free = shutil.disk_usage(probe).free
        except OSError:
            return None
        if free >= required:
            return None
        return LocalModelInstallResult(
            ok=False,
            preset=preset,
            message=(
                f"磁盘空间不足，{preset.label} 预计需要至少 "
                f"{_format_gib(required)} 可用空间，当前约 {_format_gib(free)}。"
            ),
        )

    def pull(
        self,
        preset: LocalModelPreset,
        *,
        progress: ProgressCallback | None = None,
        state_store: LocalModelStateStore | None = None,
        cancellation_token: LocalModelCancellationToken | None = None,
    ) -> LocalModelInstallResult:
        """Download one model through Ollama."""
        command = ("ollama", "pull", preset.ollama_ref)
        _emit_progress(
            progress,
            "downloading",
            f"正在下载本地模型 · {preset.short_label}",
            preset=preset,
            state_store=state_store,
        )
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message="需要先安装本地模型运行组件 Ollama：https://ollama.com/download",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=f"{preset.label} 下载没有完成：{exc}",
            )
        output_lines: deque[str] = deque(maxlen=20)
        deadline = time.monotonic() + 60 * 60
        stream_events: queue.Queue[str | None] = queue.Queue(maxsize=128)
        assert process.stdout is not None

        def read_output() -> None:
            try:
                while True:
                    line = process.stdout.readline()
                    if line == "":
                        break
                    stream_events.put(line)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    stream_events.put_nowait(None)
                except queue.Full:
                    pass

        reader = threading.Thread(
            target=read_output,
            name="deepmate-ollama-pull-output",
            daemon=True,
        )
        reader.start()
        stream_ended = False
        try:
            while True:
                if _cancelled(cancellation_token):
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
                        pass
                    return _cancelled_result(preset, state_store)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
                        pass
                    result = LocalModelInstallResult(
                        ok=False,
                        preset=preset,
                        message="模型下载还没有完成，请稍后重试 /local。",
                    )
                    _record_state(
                        state_store,
                        preset,
                        "failed",
                        result.message,
                        status="failed",
                        failure_kind="download_timeout",
                    )
                    return result
                try:
                    event = stream_events.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    if process.poll() is not None and stream_events.empty():
                        break
                    continue
                if event is None:
                    stream_ended = True
                else:
                    line = event
                    clean = line.strip()
                    if clean:
                        output_lines.append(clean)
                    percent = _progress_percent(line)
                    if percent is not None:
                        _emit_progress(
                            progress,
                            "downloading",
                            f"正在下载本地模型 · {preset.short_label} · {percent}%",
                            percent=percent,
                            preset=preset,
                            state_store=state_store,
                        )
                if stream_ended and process.poll() is not None and stream_events.empty():
                    break
        finally:
            if process.stdout is not None:
                process.stdout.close()
            reader.join(timeout=0.2)
        returncode = process.wait()
        if returncode != 0:
            output = "\n".join(tuple(output_lines)[-3:]).strip()
            detail = f"：{output}" if output else ""
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message=f"{preset.label} 下载失败{detail}",
            )
        return LocalModelInstallResult(
            ok=True,
            preset=preset,
            message=f"{preset.label} 下载完成。",
        )

    def health_check(self, preset: LocalModelPreset) -> LocalModelInstallResult:
        """Verify the local OpenAI-compatible chat endpoint can answer."""
        try:
            provider = ChatCompletionsProvider(
                base_url=self.provider_base_url,
                api_key="ollama",
                timeout=20,
            )
            response = provider.complete(
                ModelRequest(
                    model=preset.runtime_name,
                    conversation=(
                        ModelConversationItem.from_message(
                            Message(
                                role=MessageRole.USER,
                                content="Reply with ok.",
                            )
                        ),
                    ),
                    options={"max_tokens": 8},
                )
            )
        except Exception:
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message="本地模型暂时没有准备成功，已继续使用当前模型。",
            )
        if not response.has_output():
            return LocalModelInstallResult(
                ok=False,
                preset=preset,
                message="本地模型暂时没有准备成功，已继续使用当前模型。",
            )
        return LocalModelInstallResult(
            ok=True,
            preset=preset,
            message=f"{preset.label} 已验证。",
            provider_base_url=self.provider_base_url,
        )

    @property
    def provider_base_url(self) -> str:
        """Return the OpenAI-compatible base URL for this Ollama API."""
        return f"{self._api_url}/v1"

    def _get_json(self, path: str, *, timeout: float) -> dict[str, object]:
        request = Request(f"{self._api_url}{path}", method="GET")
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if content_type and "json" not in content_type.lower():
                raise ValueError("Ollama response must be JSON")
            raw = response.read(MAX_OLLAMA_JSON_RESPONSE_BYTES + 1)
            if len(raw) > MAX_OLLAMA_JSON_RESPONSE_BYTES:
                raise ValueError("Ollama response is too large")
            payload = raw.decode("utf-8")
        data = json.loads(payload)
        return data if isinstance(data, dict) else {}

    def _start_server(self) -> None:
        try:
            subprocess.Popen(
                ("ollama", "serve"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return


def ollama_api_url_from_provider_base_url(base_url: str) -> str:
    """Return the Ollama management API origin for an OpenAI-compatible base URL."""
    clean = base_url.strip().rstrip("/")
    if not clean:
        return OLLAMA_API_URL
    if clean.endswith("/v1"):
        return clean[: -len("/v1")]
    return clean


def _ollama_short_name(value: str) -> str:
    clean = value.strip()
    if clean.startswith("hf.co/"):
        return clean.rsplit("/", 1)[-1]
    return clean


def _prepare_lock(key: str) -> threading.Lock:
    with _PREPARE_LOCKS_GUARD:
        lock = _PREPARE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PREPARE_LOCKS[key] = lock
        return lock


def _preset_download_bytes(preset: LocalModelPreset) -> int:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(gb|mb)", preset.size_label, re.IGNORECASE)
    if match is None:
        return 1024**3
    value = float(match.group(1))
    multiplier = 1024**3 if match.group(2).lower() == "gb" else 1024**2
    return max(1, int(value * multiplier * 1.15))


def _format_gib(value: int) -> str:
    return f"{value / (1024**3):.1f} GiB"


def _cancelled(token: LocalModelCancellationToken | None) -> bool:
    return token is not None and token.is_cancelled()


def _cancelled_result(
    preset: LocalModelPreset,
    state_store: LocalModelStateStore | None,
) -> LocalModelInstallResult:
    message = "本地模型准备已取消；再次输入 /local 会从 Ollama 已下载的分层继续。"
    _record_state(
        state_store,
        preset,
        "cancelled",
        message,
        status="failed",
        failure_kind="cancelled",
    )
    return LocalModelInstallResult(ok=False, preset=preset, message=message)


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    *,
    percent: int | None = None,
    preset: LocalModelPreset | None = None,
    state_store: LocalModelStateStore | None = None,
) -> None:
    if preset is not None:
        _record_state(state_store, preset, stage, message)
    if callback is None:
        return
    callback(LocalModelProgress(stage=stage, message=message, percent=percent))


def _record_state(
    state_store: LocalModelStateStore | None,
    preset: LocalModelPreset,
    stage: str,
    message: str,
    *,
    status: str = "running",
    failure_kind: str = "",
) -> None:
    if state_store is None:
        return
    state_store.record(
        model_id=preset.id,
        stage=stage,
        message=message,
        status=status,
        failure_kind=failure_kind,
    )


def _record_result(
    state_store: LocalModelStateStore | None,
    result: LocalModelInstallResult,
) -> None:
    if result.ok:
        _record_state(
            state_store,
            result.preset,
            "ready",
            result.message,
            status="ready",
        )
        return
    _record_state(
        state_store,
        result.preset,
        "failed",
        result.message,
        status="failed",
        failure_kind=_failure_kind(result.message),
    )


def _failure_kind(message: str) -> str:
    text = message.lower()
    if "download" in text or "下载" in message:
        return "download_failed"
    if "install" in text or "安装" in message:
        return "runtime_missing"
    if "启动" in message or "serve" in text:
        return "service_start_failed"
    if "验证" in message:
        return "health_check_failed"
    return "prepare_failed"


def _progress_percent(line: str) -> int | None:
    match = re.search(r"(\d{1,3})%", line)
    if match is None:
        return None
    value = int(match.group(1))
    if value < 0 or value > 100:
        return None
    return value
