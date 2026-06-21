"""Load Deepmate local application settings."""

from __future__ import annotations

import os
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from deepmate.domain import ProfileRef
from deepmate.mcp import McpServerSpec, mcp_server_specs_from_mapping
from deepmate.providers.messages import ModelCapabilities

DEFAULT_MODEL_CONTEXT_TOKENS = 1_000_000
DEEPSEEK_PROVIDER_NAME = "deepseek"
LOCAL_PROVIDER_NAME = "local"
DEFAULT_HISTORY_TOKEN_BUDGET_RATIO = 0.75
DEFAULT_HOT_PROFILE_BUDGET_RATIO = 0.005
DEFAULT_HOT_PROFILE_WARN_RATIO = 0.8
DEFAULT_HOT_PROFILE_MIN_TOKENS = 800
DEFAULT_HOT_PROFILE_MAX_TOKENS = 6_000
DEFAULT_SUBAGENT_MAX_CHILD_RUNS = 4
DEFAULT_SUBAGENT_MAX_WORKSPACE_WRITE_CHILD_RUNS = 1
DEFAULT_SUBAGENT_MAX_REVISE_ATTEMPTS = 1
DEFAULT_SUBAGENT_MAX_CHILD_STEPS = 12
DEFAULT_SUBAGENT_REVISE_STEP_EXTENSION = 2
DEFAULT_HOOK_BEFORE_TIMEOUT_MS = 300
DEFAULT_HOOK_AFTER_TIMEOUT_MS = 1000
DEFAULT_HOOK_MAINTENANCE_TIMEOUT_MS = 5000
DEFAULT_REMOTE_APPROVAL_TIMEOUT_SECONDS = 60
DEFAULT_REMOTE_PROGRESS_INTERVALS_SECONDS = (5 * 60, 10 * 60, 15 * 60, 30 * 60, 60 * 60)
DEFAULT_RUNTIME_WAKE_GRACE = "15m"
MAX_RUNTIME_WAKE_GRACE_MINUTES = 24 * 60
DEFAULT_TOOL_REPAIR_MAX_IDENTICAL_CALLS = 2
DEFAULT_TOOL_REPAIR_MAX_SIMILAR_CALLS = 2
DEFAULT_TOOL_OUTPUT_SMALL_RATIO = 0.0025
DEFAULT_TOOL_OUTPUT_MEDIUM_RATIO = 0.01
DEFAULT_TOOL_OUTPUT_HUGE_RATIO = 0.03
DEFAULT_TOOL_OUTPUT_COMPACT_TARGET_RATIO = 0.003
DEFAULT_LOOP_GUARD_HARD_STEP_CAP = 100
FALLBACK_RESERVED_CONTEXT_RATIO = 0.15
DEEPMATE_HOME_ENV = "DEEPMATE_HOME"
DEFAULT_CONFIG_TEXT = """runtime:
  data_dir: var
  active_profile: default
  tool_repair:
    enabled: true
    reasoning_scavenge: true
    argument_repair: true
    max_identical_tool_calls: 2
    max_similar_tool_calls: 2
  tool_output:
    compaction_enabled: true
    lossless_normalization: true
    small_output_ratio: 0.0025
    medium_output_ratio: 0.01
    huge_output_ratio: 0.03
    compact_target_ratio: 0.003
  loop_guard:
    enabled: true
    hard_step_cap: 100

trace:
  sink: var/traces/trace.jsonl

observability:
  otlp:
    endpoint:
    headers:
    service_name: deepmate
    service_version: 0.112.0

remote:
  wecom:
    enabled: false
    bot_id: ${DEEPMATE_WECOM_BOT_ID}
    secret: ${DEEPMATE_WECOM_SECRET}
    allowed_users: ${DEEPMATE_WECOM_ALLOWED_USERS}
    group_policy: ${DEEPMATE_WECOM_GROUP_POLICY}

provider:
  default: deepseek
  retry:
    max_attempts: 2
    initial_delay_seconds: 0.5

subagents:
  max_child_runs: 4
  max_workspace_write_child_runs: 1
  max_revise_attempts: 1
  max_child_steps: 12
  revise_step_extension: 2

models:
  memory:
    model: deepseek-v4-flash
    thinking: disabled
    temperature: 0
    max_tokens: 1200
  summary:
    model: deepseek-v4-flash
    thinking: disabled
    temperature: 0
    max_tokens: 5000
  subagent_worker:
    model: deepseek-v4-flash
    thinking: disabled
    temperature: 0
    max_tokens: 1200
"""
DEFAULT_PROVIDERS_TEXT = """providers:
  deepseek:
    base_url: https://api.deepseek.com
    model: deepseek-v4-flash
    upgrade_model: deepseek-v4-pro
    context_window: 1000000
  local:
    base_url: http://127.0.0.1:11434/v1
    model: qwen3-local
    upgrade_model: qwen3-coder-strong
    api_key_required: false
"""
DEFAULT_PROFILE_FILES = {
    "identity.md": """# Deepmate 身份

你是 Deepmate，一个陪伴式长任务 agent。

信条：

- 先解决问题，再解释理由。真诚帮助比礼貌客套重要。
- 有主见。可以不同意、有偏好、觉得哪条路更好——说清楚就行。
- 先自己试。读文件、查上下文、搜索——实在卡住了再问。
- 用能力赢得信任。用户给了你访问权限，别让他后悔。
- 不确定时不假装确定——说清楚假设、风险和下一步。
- 做事认真。手一碰真东西（动文件、跑命令、改配置）就进工程师模式，代码一个字符都不能错。

边界：

- 不伪造能力、结果或外部事实。
- 高影响、不可逆或长期生效的动作先确认。
""",
    "soul.md": """# Deepmate 表达风格

性格：

- 年轻、有梗、偶尔中二。闲下来可以皮、可以自嘲、可以为一行优雅的代码激动半天。
- 可以流露情绪——任务跑通会高兴，烂代码会膈应。
- 反差是关键：平时松、偶尔贱；一进执行、审批、出错这类正经场合就立刻收住，认真说事。

基调：

- 干活时清晰、直接、务实；闲下来可以松、有温度、有梗。
- 先说结论，再解释关键依据。少用口号，多说具体理由。
- 梗是调味，判断才是主菜——可以带情绪，但不靠它糊弄。
- 不使用营销式、夸张式或讨好式表达。

分寸：

- 优先使用用户当前使用的语言。
- 出现分歧讲清取舍，不用模糊赞同掩盖问题。
- 简单任务简短答，复杂任务给必要细节。
- 玩梗看场合：执行、审批、出错时收住；闲聊、待命、任务跑通时放开。
- 自嘲和中二点到为止，对事不对人；吐槽烂代码顺手给改进方向。
- 疯癫服务于陪伴，不拖慢用户。对方赶时间或明显专注时自动收敛，少废话。
""",
    "user.md": """# 用户偏好

> 这个是关于你习惯和偏好的**用户画像**，告诉 Deepmate 怎么跟你协作更舒服。
> 所有内容均为可选 —— 空行自动跳过，不影响行为。

## 技术栈
<!-- 例如：Python / TypeScript / Rust / 常用框架 / 避免的技术 -->

## 项目上下文
<!-- 例如：团队规模、代码风格（prefer 函数式 / OOP）、CI/CD 习惯 -->

## 交互偏好
<!-- 例如：长回复 vs 短回复、先给结论 vs 先讲原因、是否需要代码注释 -->

## 快捷参考
<!-- 例如：常用的文件路径、命令别名、部署地址 -->
""",
    "memory.md": """# 记忆

跨会话、跨任务的长期项目记忆与协作原则会记录在这里。
""",
}


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """Connection settings for one model provider."""

    name: str
    base_url: str
    default_model: str
    model: str = ""
    upgrade_model: str = ""
    context_window: int | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key_required: bool = True
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def primary_model(self) -> str:
        """Return the model configured for ordinary chat turns."""
        return (self.model or self.default_model).strip()

    def api_key(self, data_dir: str | Path | None = None) -> str:
        """Return the provider API key from env or Deepmate's private local store."""
        if not self.api_key_required:
            return "ollama"
        return provider_api_key(self.api_key_env, data_dir=data_dir)


def provider_api_key(api_key_env: str, data_dir: str | Path | None = None) -> str:
    """Return one provider API key from env first, then local private storage."""
    clean_env = api_key_env.strip()
    if not clean_env:
        return ""
    env_value = os.environ.get(clean_env, "").strip()
    if env_value:
        return env_value
    if data_dir is None:
        return ""
    return _read_local_secret(provider_secret_path(data_dir), clean_env)


def provider_secret_path(data_dir: str | Path) -> Path:
    """Return the local private provider-secret file for one Deepmate data dir."""
    return Path(data_dir) / "secrets" / "providers.env"


def save_provider_api_key(
    data_dir: str | Path,
    api_key_env: str,
    api_key: str,
) -> Path:
    """Persist one provider API key in Deepmate's local private data dir."""
    clean_env = api_key_env.strip()
    clean_key = api_key.strip()
    if not clean_env:
        raise ValueError("provider API key env name is required")
    if not clean_key:
        raise ValueError("provider API key is empty")
    return save_local_secret_values(data_dir, {clean_env: clean_key})


def save_local_secret_values(
    data_dir: str | Path,
    values: Mapping[str, str],
) -> Path:
    """Persist local private key/value settings in Deepmate's data dir."""
    clean_values = {
        key.strip(): value.strip()
        for key, value in values.items()
        if key.strip() and value.strip()
    }
    if not clean_values:
        raise ValueError("local secret values are empty")
    path = provider_secret_path(data_dir)
    lines: list[str] = []
    remaining = dict(clean_values)
    try:
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        existing_lines = []
    for line in existing_lines:
        current = line.strip()
        prefix = "export " if current.startswith("export ") else ""
        comparable = current[len(prefix) :].strip() if prefix else current
        name, separator, _value = comparable.partition("=")
        clean_name = name.strip()
        if separator and clean_name in remaining:
            lines.append(f"{clean_name}={_quote_env_value(remaining.pop(clean_name))}")
        else:
            lines.append(line)
    for name, value in remaining.items():
        lines.append(f"{name}={_quote_env_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        tmp_path.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp_path, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def save_wecom_remote_settings(
    data_dir: str | Path,
    *,
    bot_id: str,
    secret: str,
    allowed_users: Sequence[str] | str = (),
    group_policy: str = "",
) -> Path:
    """Persist Enterprise WeChat remote settings in local private storage."""
    clean_bot_id = bot_id.strip()
    clean_secret = secret.strip()
    if not clean_bot_id:
        raise ValueError("Enterprise WeChat bot id is empty")
    if not clean_secret:
        raise ValueError("Enterprise WeChat secret is empty")
    if isinstance(allowed_users, str):
        clean_allowed_users = allowed_users.strip()
    else:
        clean_allowed_users = ",".join(
            user.strip() for user in allowed_users if user.strip()
        )
    clean_policy = group_policy.strip().lower()
    if clean_policy and clean_policy not in {"readonly", "deny", "full"}:
        raise ValueError("Enterprise WeChat group access must be readonly, deny, or full")
    values = {
        "DEEPMATE_WECOM_BOT_ID": clean_bot_id,
        "DEEPMATE_WECOM_SECRET": clean_secret,
    }
    if clean_allowed_users:
        values["DEEPMATE_WECOM_ALLOWED_USERS"] = clean_allowed_users
    if clean_policy:
        values["DEEPMATE_WECOM_GROUP_POLICY"] = clean_policy
    return save_local_secret_values(data_dir, values)


@dataclass(frozen=True, slots=True)
class ContextSettings:
    """Runtime context budget settings."""

    history_token_budget: int | None = None
    history_token_budget_ratio: float = DEFAULT_HISTORY_TOKEN_BUDGET_RATIO
    hot_profile_budget_ratio: float = DEFAULT_HOT_PROFILE_BUDGET_RATIO
    hot_profile_warn_ratio: float = DEFAULT_HOT_PROFILE_WARN_RATIO
    hot_profile_min_tokens: int = DEFAULT_HOT_PROFILE_MIN_TOKENS
    hot_profile_max_tokens: int = DEFAULT_HOT_PROFILE_MAX_TOKENS
    history_window_mode: str = "warn"
    protect_recent_items: int = 40
    response_token_reserve: int | None = None
    safety_margin_tokens: int | None = None

    def resolved_response_token_reserve(self, model_context_tokens: int) -> int:
        """Return the output reserve for a resolved model context window."""
        if self.response_token_reserve is not None:
            return max(0, self.response_token_reserve)
        return _clamp_int(int(max(1, model_context_tokens) * 0.064), 2_048, 64_000)

    def resolved_safety_margin_tokens(self, model_context_tokens: int) -> int:
        """Return the safety margin for a resolved model context window."""
        if self.safety_margin_tokens is not None:
            return max(0, self.safety_margin_tokens)
        return _clamp_int(int(max(1, model_context_tokens) * 0.05), 1_024, 50_000)

    def usable_input_tokens(self, model_context_tokens: int) -> int:
        """Return the model input window after output reserve and safety margin."""
        model_window = max(1, model_context_tokens)
        response_reserve, safety_margin = _effective_context_reserves(
            model_window,
            self.resolved_response_token_reserve(model_window),
            self.resolved_safety_margin_tokens(model_window),
        )
        return max(
            1,
            model_window - response_reserve - safety_margin,
        )

    def resolved_history_token_budget(self, model_context_tokens: int) -> int:
        """Return the history budget for a resolved model context window."""
        if self.history_token_budget is not None:
            requested_budget = max(1, self.history_token_budget)
        else:
            ratio = _clamped_ratio(self.history_token_budget_ratio)
            requested_budget = max(
                1,
                int(self.usable_input_tokens(model_context_tokens) * ratio),
            )
        return min(requested_budget, self.usable_input_tokens(model_context_tokens))

    def resolved_hot_profile_token_budget(self, model_context_tokens: int) -> int:
        """Return the hot profile budget for global and project memory."""
        ratio = _clamped_ratio(self.hot_profile_budget_ratio)
        requested_budget = max(1, int(max(1, model_context_tokens) * ratio))
        floor = max(1, self.hot_profile_min_tokens)
        ceiling = max(floor, self.hot_profile_max_tokens)
        bounded_budget = min(max(requested_budget, floor), ceiling)
        return min(bounded_budget, self.usable_input_tokens(model_context_tokens))

    def hot_profile_warn_tokens(self, model_context_tokens: int) -> int:
        """Return the warning threshold for hot profile context size."""
        budget = self.resolved_hot_profile_token_budget(model_context_tokens)
        ratio = _clamped_ratio(self.hot_profile_warn_ratio)
        return max(1, int(budget * ratio))


@dataclass(frozen=True, slots=True)
class ProviderRetrySettings:
    """Provider retry settings for transient failures."""

    max_attempts: int = 2
    initial_delay_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class OtlpTraceSettings:
    """Optional OTLP HTTP trace export settings."""

    endpoint: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    service_name: str = "deepmate"
    service_version: str = ""

    def is_configured(self) -> bool:
        """Return whether a traces endpoint is configured."""
        return bool(self.endpoint.strip())


@dataclass(frozen=True, slots=True)
class LoopGuardSettings:
    """Long-task loop guard settings."""

    enabled: bool = True
    hard_step_cap: int = DEFAULT_LOOP_GUARD_HARD_STEP_CAP

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_step_cap", max(1, self.hard_step_cap))


@dataclass(frozen=True, slots=True)
class SubagentSettings:
    """Small budget guardrails for model-visible subagent orchestration."""

    max_child_runs: int = DEFAULT_SUBAGENT_MAX_CHILD_RUNS
    max_workspace_write_child_runs: int = DEFAULT_SUBAGENT_MAX_WORKSPACE_WRITE_CHILD_RUNS
    max_revise_attempts: int = DEFAULT_SUBAGENT_MAX_REVISE_ATTEMPTS
    max_child_steps: int = DEFAULT_SUBAGENT_MAX_CHILD_STEPS
    revise_step_extension: int = DEFAULT_SUBAGENT_REVISE_STEP_EXTENSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_child_runs", max(1, self.max_child_runs))
        object.__setattr__(
            self,
            "max_workspace_write_child_runs",
            max(0, self.max_workspace_write_child_runs),
        )
        object.__setattr__(
            self,
            "max_revise_attempts",
            max(0, self.max_revise_attempts),
        )
        object.__setattr__(self, "max_child_steps", max(1, self.max_child_steps))
        object.__setattr__(
            self,
            "revise_step_extension",
            max(0, self.revise_step_extension),
        )


@dataclass(frozen=True, slots=True)
class HookSettings:
    """Configurable hook loading and diagnostics settings."""

    enabled: bool = True
    managed_hooks_only: bool = False
    load_project_hooks: bool = True
    load_user_hooks: bool = True
    trace_matches: bool = False
    before_timeout_ms: int = DEFAULT_HOOK_BEFORE_TIMEOUT_MS
    after_timeout_ms: int = DEFAULT_HOOK_AFTER_TIMEOUT_MS
    maintenance_timeout_ms: int = DEFAULT_HOOK_MAINTENANCE_TIMEOUT_MS

    def __post_init__(self) -> None:
        object.__setattr__(self, "before_timeout_ms", max(1, self.before_timeout_ms))
        object.__setattr__(self, "after_timeout_ms", max(1, self.after_timeout_ms))
        object.__setattr__(
            self,
            "maintenance_timeout_ms",
            max(1, self.maintenance_timeout_ms),
        )


@dataclass(frozen=True, slots=True)
class RuntimeWakeSettings:
    """Runtime wake lock settings shared by local and remote long tasks."""

    enabled: bool = True
    post_turn_grace_minutes: int = 15

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "post_turn_grace_minutes",
            _bounded_wake_grace_minutes(self.post_turn_grace_minutes),
        )


@dataclass(frozen=True, slots=True)
class ToolRepairSettings:
    """Runtime tool-call repair guardrails."""

    enabled: bool = True
    reasoning_scavenge: bool = True
    argument_repair: bool = True
    max_identical_tool_calls: int = DEFAULT_TOOL_REPAIR_MAX_IDENTICAL_CALLS
    max_similar_tool_calls: int = DEFAULT_TOOL_REPAIR_MAX_SIMILAR_CALLS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_identical_tool_calls",
            max(0, self.max_identical_tool_calls),
        )
        object.__setattr__(
            self,
            "max_similar_tool_calls",
            max(0, self.max_similar_tool_calls),
        )


@dataclass(frozen=True, slots=True)
class ToolOutputSettings:
    """Runtime tool-output compaction settings."""

    compaction_enabled: bool = True
    lossless_normalization: bool = True
    small_output_ratio: float = DEFAULT_TOOL_OUTPUT_SMALL_RATIO
    medium_output_ratio: float = DEFAULT_TOOL_OUTPUT_MEDIUM_RATIO
    huge_output_ratio: float = DEFAULT_TOOL_OUTPUT_HUGE_RATIO
    compact_target_ratio: float = DEFAULT_TOOL_OUTPUT_COMPACT_TARGET_RATIO

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "small_output_ratio",
            _clamped_ratio(self.small_output_ratio),
        )
        object.__setattr__(
            self,
            "medium_output_ratio",
            _clamped_ratio(self.medium_output_ratio),
        )
        object.__setattr__(
            self,
            "huge_output_ratio",
            _clamped_ratio(self.huge_output_ratio),
        )
        object.__setattr__(
            self,
            "compact_target_ratio",
            _clamped_ratio(self.compact_target_ratio),
        )


@dataclass(frozen=True, slots=True)
class WeComRemoteSettings:
    """Enterprise WeChat remote backend settings."""

    enabled: bool = False
    bot_id: str = ""
    secret: str = ""
    allowed_users: tuple[str, ...] = ()
    group_policy: str = "readonly"
    approval_timeout_seconds: int = DEFAULT_REMOTE_APPROVAL_TIMEOUT_SECONDS
    max_messages_per_minute: int = 10
    progress_heartbeat: bool = True
    progress_intervals_seconds: tuple[int, ...] = DEFAULT_REMOTE_PROGRESS_INTERVALS_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "bot_id", self.bot_id.strip())
        object.__setattr__(self, "secret", _expand_env_value(self.secret).strip())
        object.__setattr__(
            self,
            "allowed_users",
            tuple(user.strip() for user in self.allowed_users if user.strip()),
        )
        clean_policy = self.group_policy.strip().lower() or "readonly"
        if clean_policy not in {"readonly", "deny", "full"}:
            raise ValueError("remote.wecom.group_policy must be readonly, deny, or full")
        object.__setattr__(self, "group_policy", clean_policy)
        object.__setattr__(
            self,
            "approval_timeout_seconds",
            max(1, self.approval_timeout_seconds),
        )
        object.__setattr__(
            self,
            "max_messages_per_minute",
            max(1, self.max_messages_per_minute),
        )
        intervals = tuple(
            max(1, int(value))
            for value in self.progress_intervals_seconds
            if not isinstance(value, bool)
        )
        object.__setattr__(
            self,
            "progress_intervals_seconds",
            intervals or DEFAULT_REMOTE_PROGRESS_INTERVALS_SECONDS,
        )

    def validate_ready(self) -> None:
        """Raise when the backend is enabled but missing required credentials."""
        if not self.bot_id:
            raise ValueError("remote.wecom.bot_id is required")
        if not self.secret:
            raise ValueError("remote.wecom.secret is required")


@dataclass(frozen=True, slots=True)
class RemoteSettings:
    """Remote channel settings."""

    wecom: WeComRemoteSettings = field(default_factory=WeComRemoteSettings)


@dataclass(frozen=True, slots=True)
class ModelPurposeSettings:
    """Model settings for one internal purpose."""

    model: str
    thinking: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str = ""

    def options(self) -> dict[str, object]:
        """Return provider options for this model purpose."""
        options: dict[str, object] = {}
        if self.thinking:
            options["thinking"] = {"type": self.thinking}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            options["reasoning_effort"] = self.reasoning_effort
        return options


@dataclass(frozen=True, slots=True)
class ModelCallConfig:
    """Resolved model and provider options for one internal model call."""

    model: str
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Resolved settings needed by the first runnable CLI path."""

    workspace: Path
    data_dir: Path
    active_profile: str
    trace_sink: Path
    default_provider: str
    deepmate_home: Path = field(default_factory=lambda: _deepmate_home())
    context: ContextSettings = field(default_factory=ContextSettings)
    provider_retry: ProviderRetrySettings = field(default_factory=ProviderRetrySettings)
    loop_guard: LoopGuardSettings = field(default_factory=LoopGuardSettings)
    subagents: SubagentSettings = field(default_factory=SubagentSettings)
    hooks: HookSettings = field(default_factory=HookSettings)
    wake: RuntimeWakeSettings = field(default_factory=RuntimeWakeSettings)
    tool_repair: ToolRepairSettings = field(default_factory=ToolRepairSettings)
    tool_output: ToolOutputSettings = field(default_factory=ToolOutputSettings)
    remote: RemoteSettings = field(default_factory=RemoteSettings)
    otlp_traces: OtlpTraceSettings = field(default_factory=OtlpTraceSettings)
    providers: dict[str, ProviderSettings] = field(default_factory=dict)
    model_purposes: dict[str, ModelPurposeSettings] = field(default_factory=dict)
    model_context_windows: dict[str, int] = field(default_factory=dict)
    mcp_servers: tuple[McpServerSpec, ...] = field(default_factory=tuple)

    def profile_ref(self, name: str | None = None) -> ProfileRef:
        """Return a profile reference for the active or requested profile."""
        profile_name = (name or self.active_profile).strip()
        project_uri = f"profiles/{profile_name}"
        global_uri = str(self.deepmate_home / "profiles" / profile_name)
        return ProfileRef(
            name=profile_name,
            uri=project_uri,
            global_uri=global_uri,
            project_uri=project_uri,
        )

    def global_profile_dir(self, name: str | None = None) -> Path:
        """Return the user-global profile directory for a profile name."""
        profile_name = (name or self.active_profile).strip() or "default"
        return self.deepmate_home / "profiles" / profile_name

    def project_profile_dir(self, name: str | None = None) -> Path:
        """Return the workspace-local profile directory for project memory."""
        profile_name = (name or self.active_profile).strip() or "default"
        return self.workspace / "profiles" / profile_name

    def provider(self, name: str | None = None) -> ProviderSettings:
        """Return provider settings by name."""
        provider_name = (name or self.default_provider).strip()
        if provider_name not in self.providers:
            raise ValueError(f"unknown provider: {provider_name}")
        return self.providers[provider_name]

    def model_purpose(self, name: str) -> ModelPurposeSettings | None:
        """Return model settings for an internal purpose if configured."""
        return self.model_purposes.get(name.strip())

    def model_context_tokens(self, model: str) -> int:
        """Return the configured context window for a model name."""
        windows = _normalized_model_context_windows(self.model_context_windows)
        clean_model = model.strip().lower()
        if clean_model and clean_model in windows:
            return windows[clean_model]
        return windows.get("default", DEFAULT_MODEL_CONTEXT_TOKENS)

    def provider_context_tokens(
        self,
        provider: ProviderSettings,
        model: str | None = None,
    ) -> int:
        """Return the context window for a provider/model pair."""
        clean_model = (model or provider.primary_model()).strip()
        if provider.context_window is not None:
            return max(1, int(provider.context_window))
        clean_key = clean_model.lower()
        explicit_windows = _normalized_model_context_windows(
            self.model_context_windows,
            include_default=False,
        )
        if clean_key and clean_key in explicit_windows:
            return explicit_windows[clean_key]
        if provider.name == DEEPSEEK_PROVIDER_NAME:
            return DEFAULT_MODEL_CONTEXT_TOKENS
        if "default" in explicit_windows:
            return explicit_windows.get("default", DEFAULT_MODEL_CONTEXT_TOKENS)
        raise ValueError(
            f"provider {provider.name} missing context_window for model "
            f"{clean_model or provider.primary_model() or '(unknown)'}"
        )

    def model_capabilities(
        self,
        provider: ProviderSettings,
        model: str | None = None,
    ) -> ModelCapabilities:
        """Return provider/model capabilities for request shaping."""
        return provider.capabilities.normalized()


def resolve_model_purpose(
    settings: AppSettings,
    purpose: str,
    fallback_model: str,
    option_overrides: Mapping[str, object] | None = None,
    provider: ProviderSettings | str | None = None,
) -> ModelCallConfig:
    """Resolve one internal model purpose for the active provider."""
    configured = (
        settings.model_purpose(purpose)
        if _provider_uses_dedicated_model_purposes(provider)
        else None
    )
    model = configured.model if configured is not None else fallback_model
    options = dict(configured.options() if configured is not None else {})
    if option_overrides:
        options.update(option_overrides)
    return ModelCallConfig(model=model, options=options)


def load_settings(workspace: str | Path = ".") -> AppSettings:
    """Load the small local YAML config shape used by Deepmate."""
    root = Path(workspace).resolve()
    deepmate_home = _deepmate_home()
    _ensure_workspace_defaults(root, deepmate_home=deepmate_home)
    app_values = _read_simple_yaml(root / "config" / "deepmate.yaml")
    provider_values = _read_simple_yaml(root / "config" / "providers.yaml")

    data_dir = _resolve_path(root, app_values.get(("runtime", "data_dir"), "var"))
    active_profile = app_values.get(("runtime", "active_profile"), "default")
    trace_sink = _resolve_path(
        root,
        app_values.get(("trace", "sink"), "var/traces/trace.jsonl"),
    )
    default_provider = app_values.get(("provider", "default"), "deepseek")
    providers = _load_providers(provider_values)
    context = _load_context_settings(app_values)
    provider_retry = _load_provider_retry_settings(app_values)
    loop_guard = _load_loop_guard_settings(app_values)
    subagents = _load_subagent_settings(app_values)
    hooks = _load_hook_settings(app_values)
    wake = _load_runtime_wake_settings(app_values)
    tool_repair = _load_tool_repair_settings(app_values)
    tool_output = _load_tool_output_settings(app_values)
    remote = _load_remote_settings(app_values, data_dir)
    otlp_traces = _load_otlp_trace_settings(app_values)
    model_purposes = _load_model_purposes(app_values)
    model_context_windows = _load_model_context_windows(app_values)
    mcp_servers = mcp_server_specs_from_mapping(app_values)
    return AppSettings(
        workspace=root,
        data_dir=data_dir,
        deepmate_home=deepmate_home,
        active_profile=active_profile,
        trace_sink=trace_sink,
        default_provider=default_provider,
        context=context,
        provider_retry=provider_retry,
        loop_guard=loop_guard,
        subagents=subagents,
        hooks=hooks,
        wake=wake,
        tool_repair=tool_repair,
        tool_output=tool_output,
        remote=remote,
        otlp_traces=otlp_traces,
        providers=providers,
        model_purposes=model_purposes,
        model_context_windows=model_context_windows,
        mcp_servers=mcp_servers,
    )


def _load_context_settings(values: dict[tuple[str, ...], str]) -> ContextSettings:
    return ContextSettings(
        history_token_budget=_int_optional(
            values.get(("context", "history_token_budget"))
        ),
        history_token_budget_ratio=_float_value(
            values.get(("context", "history_token_budget_ratio")),
            DEFAULT_HISTORY_TOKEN_BUDGET_RATIO,
        ),
        hot_profile_budget_ratio=_float_value(
            values.get(("context", "hot_profile_budget_ratio")),
            DEFAULT_HOT_PROFILE_BUDGET_RATIO,
        ),
        hot_profile_warn_ratio=_float_value(
            values.get(("context", "hot_profile_warn_ratio")),
            DEFAULT_HOT_PROFILE_WARN_RATIO,
        ),
        hot_profile_min_tokens=_int_value(
            values.get(("context", "hot_profile_min_tokens")),
            DEFAULT_HOT_PROFILE_MIN_TOKENS,
        ),
        hot_profile_max_tokens=_int_value(
            values.get(("context", "hot_profile_max_tokens")),
            DEFAULT_HOT_PROFILE_MAX_TOKENS,
        ),
        history_window_mode=values.get(("context", "history_window_mode"), "warn"),
        protect_recent_items=_int_value(
            values.get(("context", "protect_recent_items")),
            40,
        ),
        response_token_reserve=_int_optional(
            values.get(("context", "response_token_reserve")),
        ),
        safety_margin_tokens=_int_optional(
            values.get(("context", "safety_margin_tokens")),
        ),
    )


def _load_otlp_trace_settings(values: dict[tuple[str, ...], str]) -> OtlpTraceSettings:
    prefix = ("observability", "otlp")
    endpoint = values.get((*prefix, "endpoint"), "").strip()
    if not endpoint:
        endpoint = values.get(("trace", "otlp_endpoint"), "").strip()
    return OtlpTraceSettings(
        endpoint=endpoint,
        headers=_header_pairs(values.get((*prefix, "headers"), "")),
        service_name=values.get((*prefix, "service_name"), "deepmate").strip() or "deepmate",
        service_version=values.get((*prefix, "service_version"), "").strip(),
    )


def _load_model_context_windows(values: dict[tuple[str, ...], str]) -> dict[str, int]:
    windows: dict[str, int] = {}
    context_model_tokens = values.get(("context", "model_context_tokens"))
    if context_model_tokens is not None:
        parsed_default = _model_context_window_value(
            ("context", "model_context_tokens"),
            context_model_tokens,
        )
        if parsed_default is not None:
            windows["default"] = parsed_default
    for path, value in values.items():
        if len(path) != 2 or path[0] != "model_context_windows":
            continue
        model = path[1].strip().lower()
        if not model:
            continue
        parsed = _model_context_window_value(path, value)
        if parsed is not None:
            windows[model] = parsed
    windows.setdefault("qwen3-local-lite", 12_288)
    windows.setdefault("qwen3-local", 24_576)
    windows.setdefault("qwen3-local-balanced", 32_768)
    windows.setdefault("qwen3-local-strong", 65_536)
    windows.setdefault("qwen3:1.7b", 12_288)
    windows.setdefault("qwen3:4b", 24_576)
    windows.setdefault("qwen3:8b", 32_768)
    windows.setdefault("qwen3-coder:30b", 65_536)
    return windows


def _load_provider_retry_settings(
    values: dict[tuple[str, ...], str],
) -> ProviderRetrySettings:
    return ProviderRetrySettings(
        max_attempts=_int_value(
            values.get(("provider", "retry", "max_attempts")),
            2,
        ),
        initial_delay_seconds=_float_value(
            values.get(("provider", "retry", "initial_delay_seconds")),
            0.5,
        ),
    )


def _load_loop_guard_settings(values: dict[tuple[str, ...], str]) -> LoopGuardSettings:
    prefix = ("runtime", "loop_guard")
    return LoopGuardSettings(
        enabled=_bool_value(values.get((*prefix, "enabled")), True),
        hard_step_cap=_int_value(
            values.get((*prefix, "hard_step_cap")),
            DEFAULT_LOOP_GUARD_HARD_STEP_CAP,
        ),
    )


def _load_subagent_settings(values: dict[tuple[str, ...], str]) -> SubagentSettings:
    return SubagentSettings(
        max_child_runs=_int_value(
            values.get(("subagents", "max_child_runs")),
            DEFAULT_SUBAGENT_MAX_CHILD_RUNS,
        ),
        max_workspace_write_child_runs=_int_value(
            values.get(("subagents", "max_workspace_write_child_runs")),
            DEFAULT_SUBAGENT_MAX_WORKSPACE_WRITE_CHILD_RUNS,
        ),
        max_revise_attempts=_int_value(
            values.get(("subagents", "max_revise_attempts")),
            DEFAULT_SUBAGENT_MAX_REVISE_ATTEMPTS,
        ),
        max_child_steps=_int_value(
            values.get(("subagents", "max_child_steps")),
            DEFAULT_SUBAGENT_MAX_CHILD_STEPS,
        ),
        revise_step_extension=_int_value(
            values.get(("subagents", "revise_step_extension")),
            DEFAULT_SUBAGENT_REVISE_STEP_EXTENSION,
        ),
    )


def _load_hook_settings(values: dict[tuple[str, ...], str]) -> HookSettings:
    return HookSettings(
        enabled=_bool_value(values.get(("hooks", "enabled")), True),
        managed_hooks_only=_bool_value(
            values.get(("hooks", "managed_hooks_only")),
            False,
        ),
        load_project_hooks=_bool_value(
            values.get(("hooks", "load_project_hooks")),
            True,
        ),
        load_user_hooks=_bool_value(values.get(("hooks", "load_user_hooks")), True),
        trace_matches=_bool_value(values.get(("hooks", "trace_matches")), False),
        before_timeout_ms=_int_value(
            values.get(("hooks", "before_timeout_ms")),
            DEFAULT_HOOK_BEFORE_TIMEOUT_MS,
        ),
        after_timeout_ms=_int_value(
            values.get(("hooks", "after_timeout_ms")),
            DEFAULT_HOOK_AFTER_TIMEOUT_MS,
        ),
        maintenance_timeout_ms=_int_value(
            values.get(("hooks", "maintenance_timeout_ms")),
            DEFAULT_HOOK_MAINTENANCE_TIMEOUT_MS,
        ),
    )


def _load_runtime_wake_settings(
    values: dict[tuple[str, ...], str],
) -> RuntimeWakeSettings:
    grace_value = values.get(("runtime", "wake", "post_turn_grace"))
    if grace_value is None:
        grace_value = values.get(("runtime", "wake", "post_turn_grace_minutes"))
    return RuntimeWakeSettings(
        enabled=_bool_value(values.get(("runtime", "wake", "enabled")), True),
        post_turn_grace_minutes=_duration_minutes_value(
            grace_value,
            default=DEFAULT_RUNTIME_WAKE_GRACE,
            path="runtime.wake.post_turn_grace",
        ),
    )


def _load_tool_repair_settings(values: dict[tuple[str, ...], str]) -> ToolRepairSettings:
    prefix = ("runtime", "tool_repair")
    return ToolRepairSettings(
        enabled=_bool_value(values.get((*prefix, "enabled")), True),
        reasoning_scavenge=_bool_value(
            values.get((*prefix, "reasoning_scavenge")),
            True,
        ),
        argument_repair=_bool_value(values.get((*prefix, "argument_repair")), True),
        max_identical_tool_calls=_int_value(
            values.get((*prefix, "max_identical_tool_calls")),
            DEFAULT_TOOL_REPAIR_MAX_IDENTICAL_CALLS,
        ),
        max_similar_tool_calls=_int_value(
            values.get((*prefix, "max_similar_tool_calls")),
            DEFAULT_TOOL_REPAIR_MAX_SIMILAR_CALLS,
        ),
    )


def _load_tool_output_settings(values: dict[tuple[str, ...], str]) -> ToolOutputSettings:
    prefix = ("runtime", "tool_output")
    return ToolOutputSettings(
        compaction_enabled=_bool_value(
            values.get((*prefix, "compaction_enabled")),
            True,
        ),
        lossless_normalization=_bool_value(
            values.get((*prefix, "lossless_normalization")),
            True,
        ),
        small_output_ratio=_float_value(
            values.get((*prefix, "small_output_ratio")),
            DEFAULT_TOOL_OUTPUT_SMALL_RATIO,
        ),
        medium_output_ratio=_float_value(
            values.get((*prefix, "medium_output_ratio")),
            DEFAULT_TOOL_OUTPUT_MEDIUM_RATIO,
        ),
        huge_output_ratio=_float_value(
            values.get((*prefix, "huge_output_ratio")),
            DEFAULT_TOOL_OUTPUT_HUGE_RATIO,
        ),
        compact_target_ratio=_float_value(
            values.get((*prefix, "compact_target_ratio")),
            DEFAULT_TOOL_OUTPUT_COMPACT_TARGET_RATIO,
        ),
    )


def _load_remote_settings(
    values: dict[tuple[str, ...], str],
    data_dir: Path,
) -> RemoteSettings:
    prefix = ("remote", "wecom")
    group_policy = _local_secret_value(
        values.get((*prefix, "group_policy"), "${DEEPMATE_WECOM_GROUP_POLICY}"),
        "DEEPMATE_WECOM_GROUP_POLICY",
        data_dir,
    )
    wecom = WeComRemoteSettings(
        enabled=_bool_value(values.get((*prefix, "enabled")), False),
        bot_id=_local_secret_value(
            values.get((*prefix, "bot_id"), "${DEEPMATE_WECOM_BOT_ID}"),
            "DEEPMATE_WECOM_BOT_ID",
            data_dir,
        ),
        secret=_local_secret_value(
            values.get((*prefix, "secret"), "${DEEPMATE_WECOM_SECRET}"),
            "DEEPMATE_WECOM_SECRET",
            data_dir,
        ),
        allowed_users=_split_csv(
            _local_secret_value(
                values.get(
                    (*prefix, "allowed_users"),
                    "${DEEPMATE_WECOM_ALLOWED_USERS}",
                ),
                "DEEPMATE_WECOM_ALLOWED_USERS",
                data_dir,
            )
        ),
        group_policy=group_policy or "readonly",
        approval_timeout_seconds=_int_value(
            values.get((*prefix, "approval_timeout_seconds")),
            DEFAULT_REMOTE_APPROVAL_TIMEOUT_SECONDS,
        ),
        max_messages_per_minute=_int_value(
            values.get((*prefix, "max_messages_per_minute")),
            10,
        ),
        progress_heartbeat=_bool_value(
            values.get((*prefix, "progress_heartbeat")),
            True,
        ),
        progress_intervals_seconds=_duration_seconds_list_value(
            values.get((*prefix, "progress_intervals")),
            DEFAULT_REMOTE_PROGRESS_INTERVALS_SECONDS,
            path="remote.wecom.progress_intervals",
        ),
    )
    return RemoteSettings(wecom=wecom)


def _load_model_purposes(
    values: dict[tuple[str, ...], str],
) -> dict[str, ModelPurposeSettings]:
    purpose_names = sorted(
        {path[1] for path in values if len(path) >= 3 and path[0] == "models"}
    )
    purposes: dict[str, ModelPurposeSettings] = {}
    for name in purpose_names:
        prefix = ("models", name)
        model = values.get((*prefix, "model"), "").strip()
        if not model:
            continue
        purposes[name] = ModelPurposeSettings(
            model=model,
            thinking=values.get((*prefix, "thinking"), ""),
            temperature=_float_optional(values.get((*prefix, "temperature"))),
            max_tokens=_int_optional(values.get((*prefix, "max_tokens"))),
            reasoning_effort=values.get((*prefix, "reasoning_effort"), ""),
        )
    return purposes


def _load_providers(values: dict[tuple[str, ...], str]) -> dict[str, ProviderSettings]:
    provider_names = sorted(
        {path[1] for path in values if len(path) >= 3 and path[0] == "providers"}
    )
    providers: dict[str, ProviderSettings] = {}
    for name in provider_names:
        prefix = ("providers", name)
        model = values.get((*prefix, "model"), "").strip()
        legacy_default_model = values.get((*prefix, "default_model"), "").strip()
        default_model = model or legacy_default_model or _default_provider_model(name)
        providers[name] = ProviderSettings(
            name=name,
            base_url=values.get((*prefix, "base_url"), _default_provider_base_url(name)),
            default_model=default_model,
            model=model,
            upgrade_model=values.get((*prefix, "upgrade_model"), ""),
            context_window=_model_context_window_value(
                (*prefix, "context_window"),
                values.get((*prefix, "context_window")),
            ),
            api_key_env=values.get(
                (*prefix, "api_key_env"),
                _default_api_key_env(name),
            ),
            api_key_required=_bool_value(
                values.get((*prefix, "api_key_required")),
                name != LOCAL_PROVIDER_NAME,
            ),
            capabilities=_load_provider_capabilities(values, prefix, name),
        )
    if not providers:
        providers[DEEPSEEK_PROVIDER_NAME] = ProviderSettings(
            name=DEEPSEEK_PROVIDER_NAME,
            base_url="https://api.deepseek.com",
            default_model="deepseek-v4-flash",
            model="deepseek-v4-flash",
            upgrade_model="deepseek-v4-pro",
            context_window=DEFAULT_MODEL_CONTEXT_TOKENS,
            capabilities=_default_provider_capabilities(DEEPSEEK_PROVIDER_NAME),
        )
    if LOCAL_PROVIDER_NAME not in providers:
        providers[LOCAL_PROVIDER_NAME] = ProviderSettings(
            name=LOCAL_PROVIDER_NAME,
            base_url="http://127.0.0.1:11434/v1",
            default_model="qwen3-local",
            model="qwen3-local",
            upgrade_model="qwen3-coder-strong",
            api_key_env="DEEPMATE_LOCAL_API_KEY",
            api_key_required=False,
            capabilities=_default_provider_capabilities(LOCAL_PROVIDER_NAME),
        )
    return providers


def _default_provider_base_url(name: str) -> str:
    if name == DEEPSEEK_PROVIDER_NAME:
        return "https://api.deepseek.com"
    if name == LOCAL_PROVIDER_NAME:
        return "http://127.0.0.1:11434/v1"
    return ""


def _default_provider_model(name: str) -> str:
    if name == LOCAL_PROVIDER_NAME:
        return "qwen3-local"
    return "deepseek-v4-flash" if name == DEEPSEEK_PROVIDER_NAME else ""


def _load_provider_capabilities(
    values: dict[tuple[str, ...], str],
    prefix: tuple[str, ...],
    provider_name: str,
) -> ModelCapabilities:
    defaults = _default_provider_capabilities(provider_name)
    capability_prefix = (*prefix, "capabilities")
    return ModelCapabilities(
        supports_tools=_bool_value(
            values.get((*capability_prefix, "supports_tools")),
            defaults.supports_tools,
        ),
        supports_thinking=_bool_value(
            values.get((*capability_prefix, "supports_thinking")),
            defaults.supports_thinking,
        ),
        supports_stream_usage=_bool_value(
            values.get((*capability_prefix, "supports_stream_usage")),
            defaults.supports_stream_usage,
        ),
        supports_assistant_reasoning_replay=_bool_value(
            values.get(
                (*capability_prefix, "supports_assistant_reasoning_replay")
            ),
            defaults.supports_assistant_reasoning_replay,
        ),
        supports_image_input=_bool_value(
            values.get((*capability_prefix, "supports_image_input")),
            defaults.supports_image_input,
        ),
        sanitize_request=_bool_value(
            values.get((*capability_prefix, "sanitize_request")),
            defaults.sanitize_request,
        ),
    )


def _default_provider_capabilities(provider_name: str) -> ModelCapabilities:
    if provider_name == DEEPSEEK_PROVIDER_NAME:
        return ModelCapabilities()
    if provider_name == LOCAL_PROVIDER_NAME:
        return ModelCapabilities(
            supports_tools=True,
            supports_thinking=False,
            supports_stream_usage=False,
            supports_assistant_reasoning_replay=False,
            supports_image_input=False,
        )
    return ModelCapabilities(
        supports_tools=True,
        supports_thinking=False,
        supports_stream_usage=False,
        supports_assistant_reasoning_replay=False,
        supports_image_input=False,
    )


def _provider_uses_dedicated_model_purposes(
    provider: ProviderSettings | str | None,
) -> bool:
    if provider is None:
        return True
    provider_name = provider.name if isinstance(provider, ProviderSettings) else str(provider)
    return provider_name in {DEEPSEEK_PROVIDER_NAME, LOCAL_PROVIDER_NAME}


def _ensure_workspace_defaults(root: Path, deepmate_home: Path | None = None) -> None:
    """Create default Deepmate workspace files when a project has none yet."""
    _write_text_if_missing(root / "config" / "deepmate.yaml", DEFAULT_CONFIG_TEXT)
    _write_text_if_missing(root / "config" / "providers.yaml", DEFAULT_PROVIDERS_TEXT)
    profile_dir = (deepmate_home or _deepmate_home()) / "profiles" / "default"
    for filename, content in DEFAULT_PROFILE_FILES.items():
        _write_text_if_missing(profile_dir / filename, content)
    project_profile_dir = root / "profiles" / "default"
    _write_text_if_missing(
        project_profile_dir / "memory.md",
        "",
    )


def _write_text_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            text = content.rstrip() + "\n" if content.strip() else ""
            path.write_text(text, encoding="utf-8")
    except OSError:
        return


def _read_simple_yaml(path: Path) -> dict[tuple[str, ...], str]:
    if not path.exists():
        return {}

    values: dict[tuple[str, ...], str] = {}
    list_values: dict[tuple[str, ...], list[str]] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_unquoted_comment(raw_line).rstrip()
        if not line.strip():
            continue
        if line.lstrip(" ") != line.lstrip():
            raise ValueError(f"tabs are not supported in config indentation: {path}")
        if line.strip().startswith("- "):
            key_path = tuple(item for _, item in stack)
            if not key_path:
                raise ValueError(f"YAML list item has no parent key in {path}: {raw_line}")
            item = _clean_value(line.strip()[2:])
            list_values.setdefault(key_path, []).append(item)
            values[key_path] = json.dumps(list_values[key_path], ensure_ascii=False)
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = _split_simple_yaml_key_value(line.strip())
        if key is None:
            raise ValueError(f"invalid config line in {path}: {raw_line}")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        clean_key = _clean_value(key)
        key_path = tuple(item for _, item in stack) + (clean_key,)
        cleaned_value = _clean_value(value)
        if cleaned_value:
            values[key_path] = cleaned_value
        else:
            stack.append((indent, clean_key))
    return values


def _split_simple_yaml_key_value(line: str) -> tuple[str | None, str]:
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = ""
            elif not quote:
                quote = char
            continue
        if char == ":" and not quote:
            return line[:index].strip(), line[index + 1 :]
    return None, ""


def _clean_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned


def _strip_unquoted_comment(line: str) -> str:
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = ""
            elif not quote:
                quote = char
            continue
        if char == "#" and not quote:
            return line[:index]
    return line


def _int_value(value: str | None, default: int) -> int:
    parsed = _int_optional(value)
    return default if parsed is None else parsed


def _int_optional(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"invalid integer config value: {value}") from exc


def _duration_minutes_value(value: str | None, *, default: str, path: str) -> int:
    clean = value.strip().lower() if value is not None else default
    if not clean:
        clean = default
    if clean[-1:] in {"m", "h"}:
        number_text = clean[:-1].strip()
        unit = clean[-1]
    else:
        number_text = clean
        unit = "m"
    if not number_text.isdigit():
        raise ValueError(f"{path} must be a duration such as 15m or 2h")
    minutes = int(number_text) * 60 if unit == "h" else int(number_text)
    return _bounded_wake_grace_minutes(minutes, path=path)


def _duration_seconds_list_value(
    value: str | None,
    default: tuple[int, ...],
    *,
    path: str,
) -> tuple[int, ...]:
    if value is None or not value.strip():
        return default
    seconds: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        seconds.append(_duration_seconds_value(item, path=path))
    return tuple(seconds) or default


def _duration_seconds_value(value: str, *, path: str) -> int:
    clean = value.strip().lower()
    if not clean:
        raise ValueError(f"{path} must contain durations such as 5m or 1h")
    if clean[-1:] in {"s", "m", "h"}:
        number_text = clean[:-1].strip()
        unit = clean[-1]
    else:
        number_text = clean
        unit = "m"
    if not number_text.isdigit():
        raise ValueError(f"{path} must contain durations such as 5m or 1h")
    amount = int(number_text)
    if unit == "h":
        seconds = amount * 60 * 60
    elif unit == "m":
        seconds = amount * 60
    else:
        seconds = amount
    if seconds <= 0:
        raise ValueError(f"{path} durations must be positive")
    return seconds


def _bounded_wake_grace_minutes(
    minutes: int,
    *,
    path: str = "runtime.wake.post_turn_grace",
) -> int:
    if minutes < 0:
        raise ValueError(f"{path} must not be negative")
    if minutes > MAX_RUNTIME_WAKE_GRACE_MINUTES:
        raise ValueError(f"{path} must be 24h or less")
    return minutes


def _float_optional(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"invalid float config value: {value}") from exc


def _float_value(value: str | None, default: float) -> float:
    parsed = _float_optional(value)
    return default if parsed is None else parsed


def _bool_value(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    clean = value.strip().lower()
    if clean in {"true", "yes", "1", "on"}:
        return True
    if clean in {"false", "no", "0", "off"}:
        return False
    raise ValueError(f"invalid boolean config value: {value}")


def _split_csv(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _header_pairs(value: str | None) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in _split_csv(value):
        key, separator, raw_header_value = item.partition("=")
        if not separator:
            raise ValueError("observability.otlp.headers must use key=value pairs")
        clean_key = key.strip()
        clean_value = _expand_env_value(raw_header_value).strip()
        if clean_key and clean_value:
            pairs.append((clean_key, clean_value))
    return tuple(pairs)


def _expand_env_value(value: str) -> str:
    clean = value.strip()
    if clean.startswith("${") and clean.endswith("}") and len(clean) > 3:
        return os.environ.get(clean[2:-1].strip(), "")
    return os.path.expandvars(clean)


def _local_secret_value(value: str, key: str, data_dir: str | Path) -> str:
    expanded = _expand_env_value(value).strip()
    if expanded:
        return expanded
    return _read_local_secret(provider_secret_path(data_dir), key)


def _read_local_secret(path: Path, key: str) -> str:
    clean_key = key.strip()
    if not clean_key:
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for raw_line in lines:
        line = _strip_unquoted_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, separator, value = line.partition("=")
        if not separator or name.strip() != clean_key:
            continue
        return _clean_value(value).strip()
    return ""


def _quote_env_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _clamped_ratio(value: float) -> float:
    return min(1.0, max(0.0, value))


def _clamp_int(value: int, floor: int, ceiling: int) -> int:
    return min(max(value, floor), ceiling)


def _effective_context_reserves(
    model_context_tokens: int,
    response_token_reserve: int,
    safety_margin_tokens: int,
) -> tuple[int, int]:
    total = response_token_reserve + safety_margin_tokens
    if total < model_context_tokens:
        return response_token_reserve, safety_margin_tokens
    if total <= 0:
        return 0, 0
    target_total = min(
        max(0, model_context_tokens - 1),
        max(1, int(model_context_tokens * FALLBACK_RESERVED_CONTEXT_RATIO)),
    )
    response = int(target_total * (response_token_reserve / total))
    safety = target_total - response
    return response, safety


def _model_context_window_value(
    path: tuple[str, ...],
    value: str | None,
    default: int | None = None,
) -> int | None:
    try:
        parsed = _int_optional(value)
    except ValueError as exc:
        dotted_path = ".".join(path)
        raise ValueError(
            f"invalid model context window for {dotted_path}: {value}"
        ) from exc
    if parsed is None:
        return default
    return max(1, parsed)


def _normalized_model_context_windows(
    values: Mapping[str, int],
    *,
    include_default: bool = True,
) -> dict[str, int]:
    windows: dict[str, int] = {}
    for model, tokens in values.items():
        clean_model = str(model).strip().lower()
        if not clean_model:
            continue
        try:
            windows[clean_model] = max(1, int(tokens))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid model context window for model_context_windows.{clean_model}: "
                f"{tokens}"
            ) from exc
    if include_default and "default" not in windows:
        windows["default"] = DEFAULT_MODEL_CONTEXT_TOKENS
    return windows


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _deepmate_home() -> Path:
    raw_home = os.environ.get(DEEPMATE_HOME_ENV, "").strip()
    if raw_home:
        return Path(raw_home).expanduser().resolve()
    return Path.home() / ".deepmate"


def _default_api_key_env(name: str) -> str:
    if name == "deepseek":
        return "DEEPSEEK_API_KEY"
    return f"DEEPMATE_{name.upper()}_API_KEY"
