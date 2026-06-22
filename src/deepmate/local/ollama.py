"""Ollama-backed local model runtime."""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from deepmate.domain import Message, MessageRole
from deepmate.local.presets import LOCAL_PROVIDER_BASE_URL, LocalModelPreset
from deepmate.local.state import LocalModelStateStore
from deepmate.providers import ChatCompletionsProvider, ModelConversationItem, ModelRequest

OLLAMA_API_URL = "http://127.0.0.1:11434"
MAX_OLLAMA_JSON_RESPONSE_BYTES = 2 * 1024 * 1024


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
    ) -> LocalModelInstallResult:
        """Ensure Ollama is running and the requested model is available."""
        _emit_progress(
            progress,
            "checking",
            f"正在检查本地环境 · {preset.short_label}",
            preset=preset,
            state_store=state_store,
        )
        status = self.ensure_ready()
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
            pulled = self.pull(preset, progress=progress, state_store=state_store)
            if not pulled.ok:
                _record_result(state_store, pulled)
                return pulled
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
                    message="本地模型暂时没有准备成功，已继续使用当前模型。稍后可以再次输入 /local。",
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

    def pull(
        self,
        preset: LocalModelPreset,
        *,
        progress: ProgressCallback | None = None,
        state_store: LocalModelStateStore | None = None,
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
        output_lines: list[str] = []
        deadline = time.monotonic() + 60 * 60
        try:
            assert process.stdout is not None
            while True:
                line = process.stdout.readline()
                if line:
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
                if line == "" and process.poll() is not None:
                    break
                if time.monotonic() > deadline:
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
        finally:
            if process.stdout is not None:
                process.stdout.close()
        returncode = process.wait()
        if returncode != 0:
            output = "\n".join(output_lines[-3:]).strip()
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
