"""Built-in local model presets for Deepmate Local."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

from deepmate.providers.messages import ModelCapabilities

LOCAL_PROVIDER_NAME = "local"
LOCAL_PROVIDER_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_PROVIDER_API_KEY = "ollama"


@dataclass(frozen=True, slots=True)
class LocalModelPreset:
    """One user-facing local model choice."""

    id: str
    label: str
    short_label: str
    runtime_name: str
    ollama_ref: str
    size_label: str
    min_memory_gb: int
    recommended_memory_gb: int
    effective_context_tokens: int
    response_token_reserve: int
    safety_margin_tokens: int
    max_tokens: int
    description: str
    supports_tools: bool = True
    supports_thinking: bool = False
    supports_stream_usage: bool = False
    supports_assistant_reasoning_replay: bool = False
    supports_image_input: bool = False

    def capabilities(self) -> ModelCapabilities:
        """Return the request capabilities for this local model."""
        return ModelCapabilities(
            supports_tools=self.supports_tools,
            supports_thinking=self.supports_thinking,
            supports_stream_usage=self.supports_stream_usage,
            supports_assistant_reasoning_replay=self.supports_assistant_reasoning_replay,
            supports_image_input=self.supports_image_input,
        )


_LOCAL_MODEL_PRESETS: tuple[LocalModelPreset, ...] = (
    LocalModelPreset(
        id="qwen3-lite",
        label="Qwen3 1.7B",
        short_label="轻量",
        runtime_name="qwen3:1.7b",
        ollama_ref="qwen3:1.7b",
        size_label="about 2GB",
        min_memory_gb=8,
        recommended_memory_gb=8,
        effective_context_tokens=12_288,
        response_token_reserve=2_048,
        safety_margin_tokens=1_024,
        max_tokens=2_048,
        description="fastest local option for older Macs and simple work",
    ),
    LocalModelPreset(
        id="qwen3-local",
        label="Qwen3 4B Instruct",
        short_label="标准",
        runtime_name="qwen3:4b",
        ollama_ref="qwen3:4b",
        size_label="about 2.5GB",
        min_memory_gb=8,
        recommended_memory_gb=16,
        effective_context_tokens=24_576,
        response_token_reserve=4_096,
        safety_margin_tokens=2_048,
        max_tokens=4_096,
        description="recommended default for most Macs",
    ),
    LocalModelPreset(
        id="qwen3-balanced",
        label="Qwen3 8B",
        short_label="增强",
        runtime_name="qwen3:8b",
        ollama_ref="qwen3:8b",
        size_label="about 5GB",
        min_memory_gb=16,
        recommended_memory_gb=24,
        effective_context_tokens=32_768,
        response_token_reserve=4_096,
        safety_margin_tokens=2_048,
        max_tokens=4_096,
        description="stronger local quality for Apple Silicon Macs",
    ),
    LocalModelPreset(
        id="qwen3-coder-strong",
        label="Qwen3 Coder 30B-A3B",
        short_label="代码强力",
        runtime_name="qwen3-coder:30b",
        ollama_ref="qwen3-coder:30b",
        size_label="about 19GB",
        min_memory_gb=32,
        recommended_memory_gb=64,
        effective_context_tokens=65_536,
        response_token_reserve=8_192,
        safety_margin_tokens=4_096,
        max_tokens=8_192,
        description="high-end local coding model for large-memory Macs",
    ),
)


def local_model_presets() -> tuple[LocalModelPreset, ...]:
    """Return Deepmate's built-in local model choices."""
    return _LOCAL_MODEL_PRESETS


def local_model_by_id(value: str) -> LocalModelPreset | None:
    """Return a preset by stable id or user-facing alias."""
    clean = value.strip().lower()
    aliases = {
        "lite": "qwen3-lite",
        "light": "qwen3-lite",
        "fast": "qwen3-lite",
        "轻量": "qwen3-lite",
        "1.7b": "qwen3-lite",
        "default": "qwen3-local",
        "recommended": "qwen3-local",
        "recommend": "qwen3-local",
        "standard": "qwen3-local",
        "标准": "qwen3-local",
        "4b": "qwen3-local",
        "balanced": "qwen3-balanced",
        "增强": "qwen3-balanced",
        "strong": "qwen3-balanced",
        "8b": "qwen3-balanced",
        "coder": "qwen3-coder-strong",
        "30b": "qwen3-coder-strong",
        "30b-a3b": "qwen3-coder-strong",
        "high": "qwen3-coder-strong",
        "代码强力": "qwen3-coder-strong",
    }
    clean = aliases.get(clean, clean)
    for preset in _LOCAL_MODEL_PRESETS:
        if clean in {
            preset.id,
            preset.label.lower(),
            preset.short_label.lower(),
            preset.runtime_name.lower(),
        }:
            return preset
    return None


def local_model_by_runtime_name(value: str) -> LocalModelPreset | None:
    """Return a preset by the model name sent to the local provider."""
    clean = value.strip().lower()
    for preset in _LOCAL_MODEL_PRESETS:
        if clean in {preset.runtime_name.lower(), preset.ollama_ref.lower()}:
            return preset
    return None


def local_model_capabilities(value: str) -> ModelCapabilities:
    """Return model capabilities inferred from a local runtime model name."""
    preset = local_model_by_runtime_name(value) or local_model_by_id(value)
    if preset is None:
        return ModelCapabilities()
    return preset.capabilities()


def recommended_local_model() -> LocalModelPreset:
    """Return a conservative local recommendation for this Mac."""
    memory_gb = _memory_gb()
    machine = platform.machine().lower()
    if memory_gb >= 64 and machine == "arm64":
        return local_model_by_id("qwen3-coder-strong") or _LOCAL_MODEL_PRESETS[-1]
    if memory_gb >= 24 and machine == "arm64":
        return local_model_by_id("qwen3-balanced") or _LOCAL_MODEL_PRESETS[2]
    if memory_gb >= 16 and machine == "arm64":
        return local_model_by_id("qwen3-local") or _LOCAL_MODEL_PRESETS[1]
    return local_model_by_id("qwen3-lite") or _LOCAL_MODEL_PRESETS[0]


def _memory_gb() -> int:
    try:
        result = subprocess.run(
            ("sysctl", "-n", "hw.memsize"),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return 8
    try:
        bytes_value = int(result.stdout.strip())
    except ValueError:
        return 8
    return max(1, round(bytes_value / (1024**3)))
