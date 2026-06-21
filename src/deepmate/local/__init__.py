"""Local model runtime helpers."""

from deepmate.local.ollama import (
    LocalModelInstallResult,
    LocalModelProgress,
    LocalModelStatus,
    OllamaLocalRuntime,
    ollama_api_url_from_provider_base_url,
)
from deepmate.local.presets import (
    LOCAL_PROVIDER_API_KEY,
    LOCAL_PROVIDER_BASE_URL,
    LOCAL_PROVIDER_NAME,
    LocalModelPreset,
    local_model_by_id,
    local_model_capabilities,
    local_model_by_runtime_name,
    local_model_presets,
    recommended_local_model,
)
from deepmate.local.state import LocalModelPrepareState, LocalModelStateStore

__all__ = [
    "LOCAL_PROVIDER_NAME",
    "LOCAL_PROVIDER_BASE_URL",
    "LOCAL_PROVIDER_API_KEY",
    "LocalModelInstallResult",
    "LocalModelPreset",
    "LocalModelProgress",
    "LocalModelPrepareState",
    "LocalModelStateStore",
    "LocalModelStatus",
    "OllamaLocalRuntime",
    "ollama_api_url_from_provider_base_url",
    "local_model_by_id",
    "local_model_capabilities",
    "local_model_by_runtime_name",
    "local_model_presets",
    "recommended_local_model",
]
