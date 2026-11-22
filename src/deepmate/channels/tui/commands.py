"""Slash command handling for the Deepmate TUI."""

from __future__ import annotations

import io
import shlex
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from deepmate.app import save_provider_api_key
from deepmate.channels import interactive as legacy_interactive
from deepmate.channels.session_lineage import (
    SessionLineageCommandResult,
    handle_session_lineage_command,
)
from deepmate.channels.session_maintenance import runtime_conversation_from_store
from deepmate.channels.tui.files import read_workspace_file_preview, workspace_diff
from deepmate.channels.tui.formatters import TuiMessage
from deepmate.channels.tui.state import LocalModelPrepareRequest, TuiRuntimeState
from deepmate.cron import handle_cron_command
from deepmate.qa import handle_qa_command
from deepmate.local import (
    LOCAL_PROVIDER_NAME,
    LocalModelInstallResult,
    LocalModelPreset,
    LocalModelStateStore,
    OllamaLocalRuntime,
    local_model_by_id,
    local_model_capabilities,
    local_model_by_runtime_name,
    local_model_presets,
    ollama_api_url_from_provider_base_url,
    recommended_local_model,
)
from deepmate.pet.state import PET_PRESETS, PetProfile
from deepmate.pet.setup import pet_setup_status
from deepmate.providers import ChatCompletionsProvider
from deepmate.runtime import (
    ConversationBudgetPolicy,
    ToolAccessMode,
    start_runtime_activation,
    start_session_runtime,
)
from deepmate.tools import COMPUTER_TOOL_NAMES
from deepmate.storage import TurnCheckpointStore


@dataclass(frozen=True, slots=True)
class TuiCommandResult:
    """Result of handling a slash command."""

    handled: bool
    exit_requested: bool = False
    messages: tuple[TuiMessage, ...] = ()
    local_prepare: LocalModelPrepareRequest | None = None


def command_suggestions() -> tuple[str, ...]:
    """Return the supported slash command names."""
    return (
        "/commands - 列出可用命令",
        "/status - 查看模型和上下文用量",
        "/task - 查看当前任务",
        "/diff - 查看工作区改动",
        "/compose - 编辑多行草稿",
        "/send - 发送草稿",
        "/cancel-compose - 清空多行草稿",
        "/restore-draft - 恢复上次草稿",
        "/files [query] - 查找可引用的工作区文件",
        "/open <path> [--offset N] [--limit N] - 打开文件，可按字节续读",
        "/find <keyword> - 在当前页查找内容",
        "/workspace <folder> - 切换工作区",
        "/setup-key <api_key> - 为当前工作区保存模型密钥",
        "/task plan <goal> - 创建或更新任务计划",
        "/task execute - 执行当前任务计划",
        "/task status - 查看任务状态和文件",
        "/task checkpoint <note> - 保存任务检查点",
        "/task clear - 清除当前任务的本地运行状态",
        "/cron add <schedule and job> - 创建定时任务草稿",
        "/cron list|status|approve|pause|resume|remove - 管理定时任务",
        "/qa <goal> - 制定验收计划并查看所需权限",
        "/qa run|status|report|list - 运行验收并查看报告",
        "/local - 准备或切换本地模型",
        "/local status - 查看安装状态、模型和体积",
        "/behavior - 管理交互偏好",
        "/computer on|off|status - 切换当前任务的电脑操作权限",
        "/model - 查看或切换模型",
        "/model local - 后续对话使用本地模型",
        "/model remote - 后续对话使用云端模型",
        "/model upgrade - 切换到已配置的增强模型",
        "/model default - 恢复默认模型",
        "/search <query> - 搜索公开网页",
        "/pet - 查看桌宠和可用操作",
        "/pet on - 打开桌宠窗口",
        "/pet setup - 安装或修复桌宠运行环境",
        "/pet select dog|cat|squirrel|penguin - 选择桌宠外观",
        "/pet learning off|low|standard - 设置桌宠学习频率",
        "/pet bubble smart|frugal - 设置气泡生成方式",
        "/deploy [path] - 预览外部访问链接",
        "/deploy status|stop|replace - 管理正在运行的预览",
        "/detail - 重新打开最近的内容页",
        "/close-tab - 关闭当前内容页",
        "/followup <text> - 在下一个安全步骤补充要求",
        "/queue <text> - 把提示加入下一轮队列",
        "/queue - 查看待处理提示",
        "/resume-queue - 继续暂停的队列",
        "/clear-queue - 清空待处理提示",
        "/approvals - 查看本会话的审批记录",
        "/clear - 清空屏幕，保留对话记录",
        "/undo-clear - 恢复刚清空的屏幕",
        "/verbose - 切换思考过程显示",
        "/trust - 切换本会话的写入和命令自动批准",
        "/rewind [turn_id] [--apply] - 预览或恢复到检查点",
        "/undo [turn_id] [--apply] - 恢复到检查点",
        "/session - 查看当前会话",
        "/new - 新建会话",
        "/session tree - 查看会话之间的关系",
        "/session clone [title] - 复制当前会话并切换",
        "/session fork <turn|sequence> [title] - 从指定步骤创建分支会话",
        "/tree, /clone, /fork - 会话分支快捷命令",
        "/remote - 查看远程通道绑定",
        "/remote --wecom - 绑定企业微信",
        "/remote --open wecom - 将当前会话交给企业微信",
        "/remote --close wecom - 回到本地接收消息",
        "/remote --unbind wecom - 解除企业微信绑定",
        "/sessions - 列出会话",
        "/resume <id> - 恢复指定会话",
        "/title <title> - 重命名当前会话",
        "/skills - 查看可用技能",
        "/show-skill <name> - 查看技能说明",
        "/mcp - 查看 MCP 服务",
        "/hooks status|validate|trust|reload - 管理运行钩子",
        "/exit, /quit - 退出 TUI",
    )


def handle_tui_command(
    prompt: str,
    state: TuiRuntimeState,
) -> TuiCommandResult:
    """Handle one TUI slash command, mutating runtime state when needed."""
    text = prompt.strip()
    if not text.startswith("/"):
        return TuiCommandResult(False)
    if text in {"/help", "/?", "/commands"}:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="command palette",
                    body="Commands opened in a content tab.",
                    preview=_commands_preview(),
                ),
            ),
        )
    if text in {"/exit", "/quit"}:
        return TuiCommandResult(True, exit_requested=True)
    if text == "/clear":
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="clear",
                    body=(
                        "Screen chat cleared; transcript and context are unchanged. "
                        "Use /undo-clear in the TUI to restore the visible messages."
                    ),
                ),
            ),
        )
    if text == "/task":
        return TuiCommandResult(True, messages=(_task_message(state),))
    if text == "/status":
        report = _status_report(state)
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="runtime status",
                    body="Runtime status opened in a content tab.",
                    preview=report,
                ),
            ),
        )
    if text == "/diff":
        diff = workspace_diff(state.workspace)
        has_changes = not diff.strip().startswith(
            (
                "No workspace diff.",
                "No git diff is available",
                "Git diff is unavailable",
            )
        )
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="diff",
                    title="workspace diff",
                    body=(
                        "Workspace diff opened in a content tab."
                        if has_changes
                        else "No workspace changes detected."
                    ),
                    preview=diff,
                ),
            ),
        )
    if text.startswith("/open "):
        relative, offset, limit = _parse_open_args(text[len("/open ") :].strip())
        if not relative:
            return TuiCommandResult(
                True,
                messages=(
                    TuiMessage(
                        kind="status",
                        title="/open",
                        body="Usage: /open <path> [--offset N] [--limit N]",
                    ),
                ),
            )
        try:
            preview = read_workspace_file_preview(
                state.workspace,
                relative,
                offset=offset,
                max_bytes=limit,
            )
        except (OSError, ValueError) as exc:
            return TuiCommandResult(
                True,
                messages=(TuiMessage(kind="error", title="/open", body=str(exc)),),
            )
        slice_label = (
            f" bytes {preview.start}-{preview.end} of {preview.bytes_total}"
            if preview.bytes_total
            else ""
        )
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="file",
                    title=relative,
                    body=f"Opened {relative}{slice_label}.",
                    preview=preview.rendered_content(),
                ),
            ),
        )
    if text.startswith("/search "):
        query = text[len("/search ") :].strip()
        if not query:
            return TuiCommandResult(
                True,
                messages=(TuiMessage(kind="status", title="/search", body="Usage: /search <query>"),),
            )
        tool = state.native_tools.get("web_search") if state.native_tools is not None else None
        if tool is None:
            return TuiCommandResult(
                True,
                messages=(
                    TuiMessage(
                        kind="warning",
                        title="/search",
                        body=(
                            "Web search is not available in this TUI session. "
                            "Start Deepmate TUI from the normal interactive entrypoint so "
                            "web_search/web_fetch are registered."
                        ),
                    ),
                ),
            )
        try:
            result = tool.call({"query": query, "max_results": 5})
        except Exception as exc:
            return TuiCommandResult(
                True,
                messages=(TuiMessage(kind="error", title="/search", body=str(exc)),),
            )
        body = result.content.strip() or "No search results found."
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="file",
                    title=f"search: {query}",
                    body="Search results opened in a content tab.",
                    refs=tuple(result.refs),
                    preview=body,
                ),
            ),
        )
    if text == "/setup-key" or text.startswith("/setup-key "):
        return _handle_setup_key_command(text, state)
    if text == "/cron" or text.startswith("/cron "):
        return _handle_cron_command(text, state)
    if text == "/qa" or text.startswith("/qa "):
        return _handle_qa_command(text, state)
    if text == "/local" or text.startswith("/local "):
        return _handle_local_command(text, state)
    if text == "/behavior" or text.startswith("/behavior "):
        return _handle_behavior_command(text, state)
    if text == "/computer" or text.startswith("/computer "):
        return _handle_computer_command(text, state)
    if text == "/model" or text.startswith("/model "):
        return _handle_model_command(text, state)
    if text == "/pet" or text.startswith("/pet "):
        return _handle_pet_command(text, state)
    if _is_session_lineage_command(text):
        return _handle_session_lineage_command(text, state)
    return _handle_legacy_command(text, state)


def _status_report(state: TuiRuntimeState) -> str:
    sections = ["Capabilities", _capability_status_detail(state), "", "Runtime"]
    sections.append(state.runtime_stats.detail_text())
    return "\n".join(sections)


def _capability_status_detail(state: TuiRuntimeState) -> str:
    registry = state.native_tools
    if registry is None:
        return "- tools: unavailable"
    names = {tool.name for tool in registry.list_tools()}
    visible = {
        str(schema.get("name", "")).strip()
        for schema in state.tool_schemas
        if hasattr(schema, "get")
    }
    lines = [
        _capability_detail(
            names,
            visible,
            ("read_text_file", "list_directory"),
            "workspace read",
        ),
        _capability_detail(
            names,
            visible,
            ("write_text_file", "edit_text_file"),
            "workspace write",
            write_policy=(
                state.tool_access_policy.mode
                if state.tool_access_policy is not None
                else None
            ),
        ),
        _capability_detail(
            names,
            visible,
            ("run_shell_command",),
            "shell",
        ),
        _capability_detail(
            names,
            visible,
            ("web_search", "web_fetch"),
            "web",
        ),
    ]
    return "\n".join(lines)


def _capability_detail(
    names: set[str],
    visible: set[str],
    tool_names: tuple[str, ...],
    label: str,
    *,
    write_policy: ToolAccessMode | None = None,
) -> str:
    registered = any(name in names for name in tool_names)
    exposed = any(name in visible for name in tool_names)
    if label == "workspace write" and exposed and write_policy != ToolAccessMode.WORKSPACE_WRITE:
        return f"- {label}: available with approval"
    if label == "shell" and registered and not exposed:
        return f"- {label}: available on explicit shell request with approval"
    if exposed:
        return f"- {label}: enabled"
    if registered:
        return f"- {label}: available on request"
    return f"- {label}: unavailable"


def _parse_open_args(raw: str) -> tuple[str, int, int]:
    """Parse /open args while allowing quoted paths."""
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()
    path_parts: list[str] = []
    offset = 0
    limit = 80_000
    index = 0
    while index < len(parts):
        part = parts[index]
        if part in {"--offset", "--from"} and index + 1 < len(parts):
            offset = _nonnegative_int(parts[index + 1], offset)
            index += 2
            continue
        if part.startswith("--offset="):
            offset = _nonnegative_int(part.partition("=")[2], offset)
            index += 1
            continue
        if part.startswith("--from="):
            offset = _nonnegative_int(part.partition("=")[2], offset)
            index += 1
            continue
        if part in {"--limit", "--max-bytes"} and index + 1 < len(parts):
            limit = _positive_int(parts[index + 1], limit)
            index += 2
            continue
        if part.startswith("--limit="):
            limit = _positive_int(part.partition("=")[2], limit)
            index += 1
            continue
        if part.startswith("--max-bytes="):
            limit = _positive_int(part.partition("=")[2], limit)
            index += 1
            continue
        path_parts.append(part)
        index += 1
    return " ".join(path_parts).strip(), offset, min(limit, 200_000)


def _nonnegative_int(value: str, fallback: int) -> int:
    try:
        return max(0, int(value))
    except ValueError:
        return fallback


def _positive_int(value: str, fallback: int) -> int:
    try:
        return max(1, int(value))
    except ValueError:
        return fallback


def _handle_setup_key_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    value = prompt[len("/setup-key") :].strip()
    if not value:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="/setup-key",
                    body="Paste your model API key after /setup-key to save it locally.",
                ),
            ),
        )
    if state.data_dir is None:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="error",
                    title="/setup-key",
                    body="Deepmate local storage is unavailable in this session.",
                ),
            ),
        )
    try:
        path = save_provider_api_key(
            state.data_dir,
            state.provider_api_key_env,
            value,
        )
    except (OSError, ValueError) as exc:
        return TuiCommandResult(
            True,
            messages=(TuiMessage(kind="error", title="/setup-key", body=str(exc)),),
        )
    state.provider_api_key_available = True
    body = "\n".join(
        (
            "Model API key saved locally.",
            "New model requests in this workspace can use it.",
            f"Storage: {path}",
        )
    )
    return TuiCommandResult(
        True,
        messages=(TuiMessage(kind="status", title="/setup-key", body=body),),
    )


def _handle_cron_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    try:
        body = handle_cron_command(prompt, workspace=state.workspace)
    except (OSError, ValueError) as exc:
        return TuiCommandResult(
            True,
            messages=(TuiMessage(kind="error", title="/cron", body=str(exc)),),
        )
    return TuiCommandResult(
        True,
        messages=(TuiMessage(kind="status", title="/cron", body=body),),
    )


def _handle_qa_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    try:
        body = handle_qa_command(
            prompt,
            workspace=state.workspace,
            provider=state.provider,
            model=state.model,
            options=state.options,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return TuiCommandResult(
            True,
            messages=(TuiMessage(kind="error", title="/qa", body=str(exc)),),
        )
    return TuiCommandResult(
        True,
        messages=(TuiMessage(kind="status", title="/qa", body=body),),
    )


def _handle_local_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    target = prompt[len("/local") :].strip()
    if target.lower() in {"status", "models", "list"}:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="/local",
                    body=_local_status(state),
                ),
            ),
        )
    preset = (local_model_by_id(target) or local_model_by_runtime_name(target)) if target else recommended_local_model()
    if preset is None:
        choices = ", ".join(item.id for item in local_model_presets())
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="warning",
                    title="/local",
                    body=f"没有这个本地模型。可选：{choices}",
                ),
            ),
        )
    return TuiCommandResult(
        True,
        messages=(
            TuiMessage(
                kind="status",
                title="/local",
                body=f"正在准备本地模型 {preset.label}。首次下载约 {preset.size_label.removeprefix('about ')}，Esc 可取消。",
            ),
        ),
        local_prepare=LocalModelPrepareRequest(preset=preset, source="/local"),
    )


def _handle_model_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    raw_target = prompt[len("/model") :].strip()
    if not raw_target:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="/model",
                    body=_model_status(state),
                ),
            ),
        )
    target = raw_target.strip()
    lowered = target.lower()
    if lowered == "local" or lowered.startswith("local "):
        preset_name = target[5:].strip()
        preset = (
            (local_model_by_id(preset_name) or local_model_by_runtime_name(preset_name))
            if preset_name
            else local_model_by_runtime_name(state.local_default_model)
            or local_model_by_id(state.local_default_model)
        )
        if preset is None and preset_name:
            return _handle_local_command(f"/local {preset_name}", state)
        if preset is None:
            preset = recommended_local_model()
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="/model",
                    body=f"正在准备本地模型 {preset.label}。首次下载约 {preset.size_label.removeprefix('about ')}，Esc 可取消。",
                ),
            ),
            local_prepare=LocalModelPrepareRequest(preset=preset, source="/model local"),
        )
    if lowered in {"remote", "cloud"}:
        if state.remote_provider is None:
            return TuiCommandResult(
                True,
                messages=(
                    TuiMessage(
                        kind="warning",
                        title="/model",
                        body="当前 session 没有可恢复的云端模型。",
                    ),
                ),
            )
        _switch_to_remote_model(state)
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title="/model",
                    body="已切回云端模型，后续对话将使用云端。",
                ),
            ),
        )
    if lowered in {"default", "flash"}:
        selected = state.default_model.strip()
        if not selected:
            selected = state.model.strip()
    elif lowered in {"upgrade", "upgraded", "strong", "pro"}:
        selected = state.upgrade_model.strip()
        if not selected:
            return TuiCommandResult(
                True,
                messages=(
                    TuiMessage(
                        kind="warning",
                        title="/model",
                        body="当前 provider 只配置了一个模型。",
                    ),
                ),
            )
        if state.provider_name == LOCAL_PROVIDER_NAME:
            preset = local_model_by_id(selected) or local_model_by_runtime_name(selected)
            if preset is not None:
                selected = preset.runtime_name
    else:
        selected = target
    if not selected:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="error",
                    title="/model",
                    body="Model name is empty.",
                ),
            ),
        )
    state.model = selected
    if state.model_change_callback is not None:
        state.model_change_callback(state)
    return TuiCommandResult(
        True,
        messages=(
            TuiMessage(
                kind="status",
                title="/model",
                body=f"后续对话将使用模型：{_friendly_model_name(state.model)}",
            ),
        ),
    )


def _handle_behavior_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    runtime = state.behavior_runtime
    if runtime is None:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="warning",
                    title="/behavior",
                    body="Behavior learning is unavailable in this session.",
                ),
            ),
        )
    target = prompt[len("/behavior") :].strip().lower()
    if target in {"on", "enable", "enabled"}:
        runtime.set_interaction_learning(True)
        body = "Deepmate interaction learning is on. Real computer behavior learning is still separate."
    elif target in {"off", "disable", "disabled"}:
        runtime.set_interaction_learning(False)
        body = "Deepmate interaction learning is off. Existing rules are kept but not injected."
    elif target in {"forget", "clear"}:
        disabled = runtime.rule_store.disable_matching("all")
        body = f"Disabled {len(disabled)} learned behavior rule(s)."
    elif target in {"", "status"}:
        body = runtime.status_text()
    else:
        body = "Usage: /behavior, /behavior on, /behavior off, or /behavior forget"
    return TuiCommandResult(
        True,
        messages=(TuiMessage(kind="status", title="/behavior", body=body),),
    )


def _handle_computer_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    runtime = state.behavior_runtime
    if runtime is None:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="warning",
                    title="/computer",
                    body="Computer Use is unavailable in this session.",
                ),
            ),
        )
    target = prompt[len("/computer") :].strip()
    lowered = target.lower()
    if lowered in {"on", "enable", "enabled"} or lowered.startswith("on "):
        task = target[2:].strip() if lowered.startswith("on ") else ""
        runtime.set_computer_use(True, task=task)
        refresh_computer_tool_surface(state)
        body = (
            "Computer Use is on for this session. Deepmate can use browser tools "
            "and macOS screenshot/click/type/key/open actions for the current task. "
            "It will not save long-term Computer Use behavior unless that learning is explicitly enabled."
        )
    elif lowered in {"off", "disable", "disabled"}:
        runtime.set_computer_use(False)
        refresh_computer_tool_surface(state)
        body = "Computer Use is off. Learned interaction preferences are unchanged."
    elif lowered in {"learning on", "learn on"}:
        body = (
            "Long-term Computer Use learning is not enabled yet. Computer Use can "
            "still run current-task actions, but Deepmate will not save desktop "
            "behavior rules until review/confirm/rollback is available."
        )
    elif lowered in {"learning off", "learn off"}:
        runtime.set_computer_learning(False)
        body = "Long-term learning from Computer Use is off. Computer Use can still run current-task actions."
    elif lowered in {"", "status"}:
        body = runtime.status_text()
    else:
        body = "Usage: /computer on, /computer off, /computer status, /computer learning on|off"
    return TuiCommandResult(
        True,
        messages=(TuiMessage(kind="status", title="/computer", body=body),),
    )


def refresh_computer_tool_surface(state: TuiRuntimeState) -> None:
    if state.native_tools is None:
        return
    visible = {
        str(schema.get("name", "")).strip()
        for schema in state.tool_schemas
        if hasattr(schema, "get")
    }
    schemas = list(state.tool_schemas)
    if state.behavior_runtime is not None and state.behavior_runtime.computer_use_enabled:
        for name in COMPUTER_TOOL_NAMES:
            if name in visible:
                continue
            tool = state.native_tools.get(name)
            if tool is not None:
                schemas.append(tool.schema())
                visible.add(name)
    else:
        schemas = [
            schema
            for schema in schemas
            if str(schema.get("name", "")).strip()
            not in COMPUTER_TOOL_NAMES
        ]
    state.tool_schemas = tuple(schemas)
    if state.refresh_skill_surface_callback is not None:
        state.refresh_skill_surface_callback(state)


def _model_status(state: TuiRuntimeState) -> str:
    local_preset = local_model_by_id(state.model) or local_model_by_runtime_name(
        state.model
    )
    current_model = (
        f"{local_preset.short_label}（{local_preset.label}）"
        if local_preset is not None
        else state.model
    )
    lines = [
        f"当前模型：{current_model}",
        f"来源：{'本地' if state.provider_name == LOCAL_PROVIDER_NAME else '云端'}",
    ]
    if (
        state.provider_name == LOCAL_PROVIDER_NAME
        or state.upgrade_model.strip()
    ) and state.default_model.strip():
        lines.append(f"默认模型：{_friendly_model_name(state.default_model)}")
    if state.upgrade_model.strip():
        lines.append(f"强力模型：{_friendly_model_name(state.upgrade_model)}")
        lines.append("使用 /model upgrade 切到强力模型。")
    lines.append("使用 /model local 切到本地模型。")
    if state.remote_provider is not None and state.provider_name == LOCAL_PROVIDER_NAME:
        lines.append("使用 /model remote 切回云端模型。")
    return "\n".join(lines)


def _friendly_model_name(value: str) -> str:
    preset = local_model_by_id(value) or local_model_by_runtime_name(value)
    if preset is None:
        return value.strip()
    return f"{preset.short_label}（{preset.label}）"


def _switch_to_local_model(state: TuiRuntimeState, preset: LocalModelPreset) -> bool:
    if state.local_provider is None:
        state.local_provider = ChatCompletionsProvider(
            base_url=state.local_provider_base_url,
            api_key=state.local_provider_api_key,
        )
    context_prepared = False
    if state.local_context_prepare_callback is not None:
        try:
            context_prepared = state.local_context_prepare_callback(state, preset)
        except Exception:
            context_prepared = False
    if state.provider_name != LOCAL_PROVIDER_NAME:
        state.remote_provider = state.provider
        state.remote_provider_name = state.provider_name
        state.remote_model = state.model
        state.remote_default_model = state.default_model
        state.remote_upgrade_model = state.upgrade_model
        state.remote_provider_api_key_env = state.provider_api_key_env
        state.remote_provider_api_key_available = state.provider_api_key_available
        state.remote_options = dict(state.options)
        state.remote_model_capabilities = state.model_capabilities
        state.remote_conversation_budget_policy = state.conversation_budget_policy
    state.provider = state.local_provider
    state.provider_name = LOCAL_PROVIDER_NAME
    state.provider_api_key_env = "DEEPMATE_LOCAL_API_KEY"
    state.provider_api_key_available = True
    state.model = preset.runtime_name
    state.default_model = preset.runtime_name
    state.upgrade_model = (
        state.local_upgrade_model.strip()
        if state.local_upgrade_model.strip() != preset.runtime_name
        else ""
    )
    state.options = {**dict(state.options), "max_tokens": preset.max_tokens}
    state.model_capabilities = local_model_capabilities(preset.runtime_name)
    state.conversation_budget_policy = _local_budget_policy(state, preset)
    if state.tool_output_compactor is not None:
        state.tool_output_compactor = state.tool_output_compactor.with_policy(
            state.conversation_budget_policy
        )
    state.runtime_stats.reset_model_usage(preset.effective_context_tokens)
    if state.model_change_callback is not None:
        state.model_change_callback(state)
    if state.refresh_skill_surface_callback is not None:
        state.refresh_skill_surface_callback(state)
    return context_prepared


def prepare_and_switch_to_local_model(
    state: TuiRuntimeState,
    request: LocalModelPrepareRequest,
) -> TuiCommandResult:
    """Prepare a local model, then switch runtime state when it succeeds."""
    result = OllamaLocalRuntime(
        api_url=ollama_api_url_from_provider_base_url(state.local_provider_base_url)
    ).prepare_model(request.preset)
    return apply_local_model_prepare_result(state, request, result)


def apply_local_model_prepare_result(
    state: TuiRuntimeState,
    request: LocalModelPrepareRequest,
    result: LocalModelInstallResult,
) -> TuiCommandResult:
    """Apply one prepared local-model result on the UI/runtime state thread."""
    if not result.ok:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="warning",
                    title=request.source,
                    body=result.message,
                    preview=_local_status(state),
                ),
            ),
        )
    current_base_url = state.local_provider_base_url.strip().rstrip("/")
    result_base_url = (result.provider_base_url or state.local_provider_base_url).strip()
    if result_base_url:
        state.local_provider_base_url = result_base_url
    if state.local_provider is None or (
        result_base_url and result_base_url.rstrip("/") != current_base_url
    ):
        state.local_provider = ChatCompletionsProvider(
            base_url=state.local_provider_base_url,
            api_key=state.local_provider_api_key,
        )
    if request.defer_switch:
        state.pending_local_switch = request.preset
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="status",
                    title=request.source,
                    body=(
                        f"{request.preset.short_label}（{request.preset.label}）已就绪，"
                        "本轮完成后将自动切到本地模型。"
                    ),
                ),
            ),
        )
    context_prepared = _switch_to_local_model(state, request.preset)
    if state.provider_name != LOCAL_PROVIDER_NAME or state.model != request.preset.runtime_name:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="warning",
                    title=request.source,
                    body=(
                        f"{request.preset.short_label} 已就绪，但 Deepmate 没有成功切换到本地模型。"
                        f"当前模型仍是 {state.model}。"
                    ),
                    preview=_local_status(state),
                ),
            ),
        )
    body = (
        f"{request.preset.short_label}（{request.preset.label}）已就绪，"
        f"后续对话将使用本地模型：{state.model}。"
    )
    if context_prepared:
        body += "\n当前会话较长，已自动整理上下文。"
    return TuiCommandResult(
        True,
        messages=(
            TuiMessage(
                kind="status",
                title=request.source,
                body=body,
            ),
        ),
    )


def _switch_to_remote_model(state: TuiRuntimeState) -> None:
    state.provider = state.remote_provider or state.provider
    state.provider_name = state.remote_provider_name or state.provider_name
    state.model = state.remote_model or state.remote_default_model or state.model
    state.default_model = state.remote_default_model or state.default_model
    state.upgrade_model = state.remote_upgrade_model or state.upgrade_model
    state.provider_api_key_env = (
        state.remote_provider_api_key_env or state.provider_api_key_env
    )
    state.provider_api_key_available = state.remote_provider_api_key_available
    state.options = dict(state.remote_options)
    state.model_capabilities = state.remote_model_capabilities
    state.conversation_budget_policy = state.remote_conversation_budget_policy
    policy = state.conversation_budget_policy
    state.runtime_stats.reset_model_usage(policy.model_context_tokens if policy else None)
    if state.model_change_callback is not None:
        state.model_change_callback(state)
    if state.tool_output_compactor is not None:
        state.tool_output_compactor = state.tool_output_compactor.with_policy(
            state.conversation_budget_policy
        )
    if state.refresh_skill_surface_callback is not None:
        state.refresh_skill_surface_callback(state)


def _local_budget_policy(
    state: TuiRuntimeState,
    preset: LocalModelPreset,
) -> ConversationBudgetPolicy:
    protect_recent = 40
    history_mode = "warn"
    if state.conversation_budget_policy is not None:
        protect_recent = state.conversation_budget_policy.protect_recent_items
        history_mode = state.conversation_budget_policy.history_window_mode
    usable = max(
        1,
        preset.effective_context_tokens
        - preset.response_token_reserve
        - preset.safety_margin_tokens,
    )
    return ConversationBudgetPolicy(
        history_token_budget=max(1, int(usable * 0.75)),
        history_window_mode=history_mode,
        protect_recent_items=protect_recent,
        model_context_tokens=preset.effective_context_tokens,
        response_token_reserve=preset.response_token_reserve,
        safety_margin_tokens=preset.safety_margin_tokens,
    )


def _local_status(state: TuiRuntimeState | None = None) -> str:
    api_url = (
        ollama_api_url_from_provider_base_url(state.local_provider_base_url)
        if state is not None
        else None
    )
    runtime = OllamaLocalRuntime(api_url=api_url) if api_url is not None else OllamaLocalRuntime()
    status = runtime.status()
    lines = ["Deepmate Local"]
    if state is not None:
        prepare_state = LocalModelStateStore(state.data_dir).load()
        if prepare_state is not None:
            lines.append(f"- 上次准备：{prepare_state.user_message()}")
    names = set()
    if status.running:
        suffix = f" ({status.version})" if status.version else ""
        lines.append(f"- Ollama 已连接{suffix}")
        names = {name.lower() for name in runtime.installed_models()}
    elif status.installed:
        lines.append("- 已安装 Ollama。启动服务后输入 /local 继续。")
    else:
        lines.append("- 尚未连接 Ollama，可从 https://ollama.com/download 安装。")
    lines.extend(("", "可用预设"))
    recommended = recommended_local_model()
    for preset in local_model_presets():
        installed = preset.runtime_name.lower() in names or preset.ollama_ref.lower() in names
        markers = [f"约 {preset.size_label.removeprefix('about ')}"]
        if installed:
            markers.append("已安装")
        if preset.id == recommended.id:
            markers.append("推荐")
        lines.append(f"- {preset.label}（{' · '.join(markers)}）  /local {preset.short_label}")
    return "\n".join(lines)


def _handle_pet_command(prompt: str, state: TuiRuntimeState) -> TuiCommandResult:
    store = state.pet_state_store
    if store is None:
        return TuiCommandResult(
            True,
            messages=(
                TuiMessage(
                    kind="warning",
                    title="/pet",
                    body="Desktop pet state is unavailable for this TUI session.",
                ),
            ),
        )
    parts = prompt.split()
    action = parts[1].lower() if len(parts) >= 2 else "status"
    try:
        if action in {"status", "show"}:
            return TuiCommandResult(
                True,
                messages=(
                    TuiMessage(
                        kind="status",
                        title="desktop pet",
                        body="Desktop pet status opened in a content tab.",
                        preview=_format_pet_status(store),
                    ),
                ),
            )
        if action == "on":
            return TuiCommandResult(
                True,
                messages=(
                    TuiMessage(
                        kind="status",
                        title="desktop pet",
                        body=(
                            "Starting desktop pet window. Deepmate will report an "
                            "error here if the desktop window cannot be opened."
                        ),
                        refs=("pet_start_requested=true",),
                    ),
                ),
            )
        if action == "setup":
            status = pet_setup_status(state.data_dir)
            if status.ready:
                return TuiCommandResult(
                    True,
                    messages=(
                        TuiMessage(
                            kind="status",
                            title="desktop pet",
                            body="Desktop pet runtime is ready. Use /pet on to open it.",
                            preview=_pet_setup_preview(status),
                        ),
                    ),
                )
            return TuiCommandResult(
                True,
                messages=(
                    TuiMessage(
                        kind="status",
                        title="desktop pet",
                        body=(
                            "Desktop pet setup is ready to install Electron into "
                            "Deepmate's local data directory. Approve the setup "
                            "prompt to continue."
                        ),
                        refs=("pet_setup_requested=true",),
                        preview=_pet_setup_preview(status),
                    ),
                ),
            )
        if action == "select":
            if len(parts) < 3:
                return _pet_usage("Usage: /pet select dog|cat|squirrel|penguin")
            profile = store.select_pet(parts[2])
            return _pet_updated(profile)
        if action == "learning":
            if len(parts) < 3 or parts[2] not in {"off", "low", "standard"}:
                return _pet_usage("Usage: /pet learning off|low|standard")
            profile = store.load_profile()
            profile = PetProfile.from_record(
                {**profile.to_record(), "learning_mode": parts[2]}
            )
            store.save_profile(profile)
            return _pet_updated(
                profile,
                learning_sources=store.load_learning_state().get("sources"),
            )
        if action == "bubble":
            if len(parts) < 3 or parts[2] not in {"smart", "frugal"}:
                return _pet_usage("Usage: /pet bubble smart|frugal")
            profile = store.load_profile()
            profile = PetProfile.from_record(
                {**profile.to_record(), "bubble_generation": parts[2]}
            )
            store.save_profile(profile)
            return _pet_updated(profile)
    except (OSError, ValueError) as exc:
        return TuiCommandResult(
            True,
            messages=(TuiMessage(kind="error", title="/pet", body=str(exc)),),
        )
    return _pet_usage(
        "Usage: /pet, /pet on, /pet setup, /pet select <pet>, /pet learning <mode>, or /pet bubble <mode>"
    )


def _pet_usage(body: str) -> TuiCommandResult:
    return TuiCommandResult(
        True,
        messages=(TuiMessage(kind="status", title="/pet", body=body),),
    )


def _pet_updated(profile: PetProfile, learning_sources: object = None) -> TuiCommandResult:
    body = "Desktop pet settings updated."
    if profile.learning_mode != "off" and isinstance(learning_sources, list):
        sources = ", ".join(str(item) for item in learning_sources if str(item).strip())
        if sources:
            body += f"\nLearning mode may fetch: {sources}"
    elif profile.learning_mode != "off":
        body += "\nLearning mode is on, but no external learning sources are configured."
    return TuiCommandResult(
        True,
        messages=(
            TuiMessage(
                kind="status",
                title="desktop pet",
                body=body,
                preview=_format_pet_profile(profile),
            ),
        ),
    )


def _pet_setup_preview(status=None) -> str:
    ready = bool(getattr(status, "ready", False))
    ui_dir = str(getattr(status, "ui_dir", "") or "")
    message = str(getattr(status, "message", "") or "")
    return "\n".join(
        (
            "Desktop Pet Setup",
            "",
            f"Status: {'ready' if ready else 'not ready'}",
            f"Detail: {message or 'Electron runtime is required before the pet can open.'}",
            f"Runtime directory: {ui_dir or '(unavailable)'}",
            "",
            "Deepmate includes the pet UI assets in the Python package. /pet setup installs the optional Electron runtime into Deepmate's local data directory, not into your project.",
            "",
            "Options",
            "- If Electron is already installed, set DEEPMATE_PET_ELECTRON to that binary.",
            "- To use a faster Electron mirror, set DEEPMATE_PET_ELECTRON_MIRROR before running /pet setup.",
            "- From a source checkout, npm --prefix src/deepmate/pet_ui install also works.",
            "- After setup, run /pet on.",
            "",
            "The pet reads Deepmate local state only; it does not watch the screen, keyboard, camera, or microphone.",
        )
    )


def _format_pet_status(store) -> str:
    profile = store.load_profile()
    state = store.load_current_state() or store.offline_state()
    lines = [
        "Desktop pet",
        "",
        _format_pet_profile(profile),
        "",
        "Current work",
        f"- kind: {_text(state.get('kind')) or '(none)'}",
        f"- state: {_text(state.get('state')) or '(none)'}",
        f"- work: {_text(state.get('current_work_title')) or '(none)'}",
        f"- summary: {_single_line(_text(state.get('summary')), limit=180) or '(none)'}",
        "",
        "Quick actions",
        "- /pet on",
        "- /pet select dog|cat|squirrel|penguin",
        "- /pet learning off|low|standard",
        "- /pet bubble smart|frugal",
    ]
    return "\n".join(lines)


def _format_pet_profile(profile: PetProfile) -> str:
    choices = "|".join(PET_PRESETS)
    return "\n".join(
        (
            "Profile",
            f"- pet: {profile.pet_id}",
            f"- species: {profile.species}",
            f"- style: {profile.style}",
            f"- name: {profile.name.strip() or '(none)'}",
            f"- bubble: {profile.bubble_generation}",
            f"- learning: {profile.learning_mode}",
            f"- proactive care: {str(profile.proactive_care).lower()}",
            f"- available pets: {choices}",
        )
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _single_line(text: str, *, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _is_session_lineage_command(prompt: str) -> bool:
    clean = prompt.strip()
    return (
        clean == "/session"
        or clean.startswith("/session ")
        or clean == "/tree"
        or clean == "/clone"
        or clean.startswith("/clone ")
        or clean == "/fork"
        or clean.startswith("/fork ")
    )


def _handle_session_lineage_command(
    prompt: str,
    state: TuiRuntimeState,
) -> TuiCommandResult:
    turn_store = (
        TurnCheckpointStore.in_data_dir(
            state.data_dir,
            state.session.profile.name,
            state.session.session_id,
        )
        if state.data_dir is not None
        else None
    )
    try:
        result = handle_session_lineage_command(
            prompt,
            session_store=state.session_store,
            session=state.session,
            workspace=state.workspace,
            profile=state.profile,
            turn_store=turn_store,
        )
    except ValueError as exc:
        return TuiCommandResult(
            True,
            messages=(TuiMessage(kind="error", title=prompt, body=str(exc)),),
        )
    if result is None:
        return TuiCommandResult(False)
    if isinstance(result, SessionLineageCommandResult):
        state.session = result.session
        state.transcript = state.session_store.transcript_store(result.session)
        if state.checkpoint_controller_factory is not None:
            state.checkpoint_controller = state.checkpoint_controller_factory(
                result.session
            )
            if state.checkpoint_write_router is not None:
                state.checkpoint_write_router.set_controller(
                    state.checkpoint_controller
                )
        activation = start_runtime_activation(
            session_id=result.session.session_id,
            workspace=result.session.workspace,
            profile=result.session.profile,
            context_snapshot=(
                state.context_snapshot_factory(result.session.profile)
                if state.context_snapshot_factory is not None
                else None
            ),
        )
        state.runtime = start_session_runtime(
            activation,
            conversation=runtime_conversation_from_store(
                state.session_store,
                result.session,
                state.transcript,
                turn_checkpoint_store=(
                    state.checkpoint_controller.turn_store
                    if state.checkpoint_controller is not None
                    else None
                ),
            ),
        )
        return TuiCommandResult(
            True,
            messages=(TuiMessage(kind="status", title=prompt, body=result.body),),
        )
    return TuiCommandResult(
        True,
        messages=(TuiMessage(kind="status", title=prompt, body=result),),
    )


def _task_message(state: TuiRuntimeState) -> TuiMessage:
    if state.task_controller is None or state.task_controller.active_stage is None:
        return TuiMessage(kind="status", title="task", body="Task Mode is not active.")
    context = state.task_controller.context()
    preview_lines = ["Task Mode", f"- stage: {state.task_controller.active_stage.value}"]
    body_lines = [f"stage: {state.task_controller.active_stage.value}"]
    if context is not None:
        preview_lines.append(f"- context estimate: {context.estimated_tokens} tokens")
        body_lines.append(f"context: {context.estimated_tokens} estimated tokens")
        if context.plan:
            preview_lines.append("")
            preview_lines.append("Current plan")
            preview_lines.extend(_preview_lines(context.plan, limit=60))
        if context.rolling_summary:
            preview_lines.append("")
            preview_lines.append("Rolling summary")
            preview_lines.extend(_preview_lines(context.rolling_summary, limit=20))
        if context.recent_timeline:
            preview_lines.append("")
            preview_lines.append("Recent timeline")
            preview_lines.extend(_preview_lines(context.recent_timeline, limit=40))
        if context.recent_achievements:
            preview_lines.append("")
            preview_lines.append("Recent achievements")
            preview_lines.extend(_preview_lines(context.recent_achievements, limit=40))
    paths = _task_paths(state.workspace)
    if paths:
        preview_lines.append("")
        preview_lines.append("Files")
        preview_lines.extend(paths)
        body_lines.append(f"files: {len(paths)}")
    return TuiMessage(
        kind="task",
        title=state.task_stage_label(),
        body="Task Mode opened in a content tab.\n" + "\n".join(body_lines),
        preview="\n".join(preview_lines),
    )


def _task_paths(workspace: Path) -> tuple[str, ...]:
    task_dir = workspace / "task"
    paths = [
        task_dir / "plan.md",
        task_dir / "evolution.md",
    ]
    achievements = task_dir / "achievements"
    if achievements.exists():
        paths.extend(sorted(achievements.glob("*.md"))[-3:])
    return tuple(
        f"- {path.relative_to(workspace).as_posix()}"
        for path in paths
        if path.exists()
    )


def _handle_legacy_command(
    prompt: str,
    state: TuiRuntimeState,
) -> TuiCommandResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        command = legacy_interactive._handle_command(
            prompt=prompt,
            session_store=state.session_store,
            session=state.session,
            transcript=state.transcript,
            runtime=state.runtime,
            workspace=state.workspace,
            profile=state.profile,
            mcp_servers=state.mcp_servers,
            trace_recorder=state.trace_recorder,
            context_snapshot_factory=state.context_snapshot_factory,
            capability_state_store=state.capability_state_store,
            checkpoint_controller_factory=state.checkpoint_controller_factory,
            checkpoint_controller=state.checkpoint_controller,
            hook_context=state.hook_context,
            hook_load_options=state.hook_load_options,
            data_dir=state.data_dir,
        )
    if command.action == "":
        return TuiCommandResult(False)
    if command.session is not None:
        state.session = command.session
    if command.transcript is not None:
        state.transcript = command.transcript
    if command.runtime is not None:
        state.runtime = command.runtime
    if command.checkpoint_controller is not None:
        state.checkpoint_controller = command.checkpoint_controller
        if state.checkpoint_write_router is not None:
            state.checkpoint_write_router.set_controller(command.checkpoint_controller)
    output = _captured_output(stdout.getvalue(), stderr.getvalue())
    messages = (
        (TuiMessage(kind="status", title=prompt, body=output),)
        if output
        else ()
    )
    return TuiCommandResult(
        True,
        exit_requested=command.action == "exit",
        messages=messages,
    )


def _captured_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append(stdout.strip())
    if stderr.strip():
        parts.append(stderr.strip())
    return "\n".join(parts).strip()


def _commands_preview() -> str:
    suggestions = command_suggestions()
    groups = (
        ("常用", ("/commands", "/status", "/diff", "/open", "/find")),
        ("会话", ("/session", "/sessions", "/resume", "/title", "/new")),
        ("文件", ("/files", "/open", "/find", "/close-tab", "/workspace", "/detail")),
        ("队列", ("/followup", "/queue", "/resume-queue", "/clear-queue")),
        ("草稿", ("/compose", "/send", "/cancel-compose", "/restore-draft")),
        ("任务", ("/task",)),
        ("验收", ("/qa",)),
        ("远程和预览", ("/remote", "/deploy")),
        ("工具", ("/skills", "/show-skill", "/mcp", "/hooks")),
        ("模型", ("/model", "/local", "/setup-key")),
        ("桌宠", ("/pet",)),
        ("退出", ("/exit", "/quit")),
    )
    lines = ["命令列表"]
    seen: set[str] = set()
    for title, prefixes in groups:
        entries = [
            item
            for item in suggestions
            if item not in seen
            and item.split(maxsplit=1)[0].rstrip(",") in prefixes
        ]
        if not entries:
            continue
        lines.extend(("", title))
        lines.extend(f"  {entry}" for entry in entries)
        seen.update(entries)
    remaining = [item for item in suggestions if item not in seen]
    if remaining:
        lines.extend(("", "其他"))
        lines.extend(f"  {entry}" for entry in remaining)
    return "\n".join(lines)


def _preview_lines(text: str, *, limit: int) -> list[str]:
    lines = []
    for line in text.splitlines():
        clean = line.rstrip()
        if not clean.strip():
            continue
        lines.append(clean)
        if len(lines) >= limit:
            break
    return lines
