"""Runtime integration for behavior learning and Computer Use."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from deepmate.behavior.rules import (
    BehaviorRule,
    BehaviorRuleStore,
    BehaviorSettings,
    BehaviorSettingsStore,
    BehaviorTraceStore,
    extract_explicit_behavior_rules,
    extract_forget_query,
    match_behavior_rules,
    render_behavior_turn_tail,
    workspace_hash,
)
from deepmate.domain import Message, MessageRole, ProfileRef
from deepmate.providers import ModelConversationItem
from deepmate.runtime.agent_loop import UserTurnResult


@dataclass(frozen=True, slots=True)
class BehaviorRuntimeResult:
    """Turn-tail behavior context prepared before one model request."""

    messages: tuple[Message, ...] = ()
    matched_rules: tuple[BehaviorRule, ...] = ()
    disabled_rules: tuple[BehaviorRule, ...] = ()
    refs: tuple[str, ...] = ()


@dataclass(slots=True)
class BehaviorRuntime:
    """Session-scoped behavior and Computer Use runtime state."""

    data_dir: Path
    workspace: Path
    profile: ProfileRef
    session_id: str
    settings_store: BehaviorSettingsStore
    rule_store: BehaviorRuleStore
    trace_store: BehaviorTraceStore
    settings: BehaviorSettings = field(default_factory=BehaviorSettings)
    computer_use_enabled: bool = False
    computer_task: str = ""

    @classmethod
    def create(
        cls,
        *,
        data_dir: str | Path,
        workspace: str | Path,
        profile: ProfileRef,
        session_id: str,
        interaction_learning_enabled: bool | None = None,
        computer_learning_enabled: bool | None = None,
        computer_use_enabled: bool = False,
    ) -> "BehaviorRuntime":
        settings_store = BehaviorSettingsStore.in_data_dir(data_dir)
        settings = settings_store.load()
        if interaction_learning_enabled is not None:
            settings = replace(
                settings,
                interaction_learning_enabled=interaction_learning_enabled,
                evidence_enabled=interaction_learning_enabled,
            )
        if computer_learning_enabled is not None:
            settings = replace(
                settings,
                computer_learning_enabled=computer_learning_enabled,
            )
        if (
            interaction_learning_enabled is not None
            or computer_learning_enabled is not None
        ):
            settings_store.save(settings)
        return cls(
            data_dir=Path(data_dir),
            workspace=Path(workspace),
            profile=profile,
            session_id=session_id,
            settings_store=settings_store,
            rule_store=BehaviorRuleStore.in_data_dir(data_dir, profile.name),
            trace_store=BehaviorTraceStore.in_data_dir(data_dir),
            settings=settings,
            computer_use_enabled=computer_use_enabled,
        )

    def with_profile(
        self,
        *,
        workspace: str | Path,
        profile: ProfileRef,
        session_id: str,
    ) -> "BehaviorRuntime":
        return BehaviorRuntime.create(
            data_dir=self.data_dir,
            workspace=workspace,
            profile=profile,
            session_id=session_id,
            interaction_learning_enabled=self.settings.interaction_learning_enabled,
            computer_learning_enabled=self.settings.computer_learning_enabled,
            computer_use_enabled=False,
        )

    def set_interaction_learning(self, enabled: bool) -> BehaviorSettings:
        self.settings = self.settings_store.set_interaction_learning(enabled)
        return self.settings

    def set_computer_learning(self, enabled: bool) -> BehaviorSettings:
        self.settings = self.settings_store.set_computer_learning(enabled)
        return self.settings

    def set_computer_use(self, enabled: bool, *, task: str = "") -> None:
        self.computer_use_enabled = enabled
        self.computer_task = task.strip() if enabled else ""
        _write_computer_session(
            self.data_dir,
            session_id=self.session_id,
            payload={
                "session_id": self.session_id,
                "workspace_hash": workspace_hash(self.workspace),
                "profile": self.profile.name,
                "computer_use_enabled": self.computer_use_enabled,
                "computer_learning_enabled": self.settings.computer_learning_enabled,
                "task": self.computer_task,
                "updated_at": _timestamp(),
            },
        )

    def prepare_turn_tail(
        self,
        messages: Sequence[Message],
        *,
        tool_schema_names: Sequence[str] = (),
    ) -> BehaviorRuntimeResult:
        user_text = _messages_text(messages)
        disabled = self.apply_forget_request(user_text)
        rules = (
            match_behavior_rules(
                _safe_enabled_rules(self.rule_store),
                user_text,
                workspace_hash_value=workspace_hash(self.workspace),
            )
            if self.settings.interaction_learning_enabled
            else ()
        )
        tail_sections: list[str] = []
        behavior_section = render_behavior_turn_tail(rules)
        if behavior_section:
            tail_sections.append(behavior_section)
        computer_section = self.render_computer_use_turn_tail(
            user_text,
            tool_schema_names=tool_schema_names,
        )
        if computer_section:
            tail_sections.append(computer_section)
        if not tail_sections:
            return BehaviorRuntimeResult(disabled_rules=disabled)
        return BehaviorRuntimeResult(
            messages=(
                Message(
                    role=MessageRole.USER,
                    content="\n\n".join(tail_sections),
                ),
            ),
            matched_rules=rules,
            disabled_rules=disabled,
            refs=(
                f"behavior_rules={len(rules)}",
                f"computer_use={str(self.computer_use_enabled).lower()}",
                f"disabled_rules={len(disabled)}",
            ),
        )

    def render_computer_use_turn_tail(
        self,
        user_text: str,
        *,
        tool_schema_names: Sequence[str] = (),
    ) -> str:
        if not self.computer_use_enabled:
            return ""
        names = {name.strip() for name in tool_schema_names if name.strip()}
        screenshot_available = "computer_screenshot" in names or "browser_screenshot" in names
        snapshot_available = "computer_snapshot" in names or "browser_snapshot" in names
        lines = [
            "<deepmate_computer_use>",
            "Computer Use is enabled for this task only.",
            "- Use available browser/computer tools to complete the user's current task; do not save long-term behavior notes from this mode unless long-term computer learning is explicitly enabled.",
            "- Prefer browser DOM snapshots or desktop accessibility snapshots before screenshots; screenshots return a private image path and dimensions, not visual content for the model.",
            "- Before desktop clicks or typing, inspect the current screen state with a readable snapshot and use stable coordinates from that result.",
            "- Do not inspect cookies, localStorage, password stores, credentials, CAPTCHA, or private messages unrelated to the task.",
            "- Ask before irreversible external actions, purchases, account changes, or sending messages on the user's behalf.",
            "- Finish by stating what changed, what remains, and any handoff the user may need.",
        ]
        if self.computer_task:
            lines.append(f"- Current task: {self.computer_task}")
        elif user_text:
            lines.append(f"- Current task: {_preview(user_text, 220)}")
        lines.append(
            f"- Screenshot capability: {'available' if screenshot_available else 'load browser tools first if a screenshot is needed'}."
        )
        lines.append(
            f"- UI snapshot capability: {'available' if snapshot_available else 'load browser tools first if a DOM snapshot is needed'}."
        )
        lines.append("</deepmate_computer_use>")
        return "\n".join(lines)

    def apply_forget_request(self, user_text: str) -> tuple[BehaviorRule, ...]:
        query = extract_forget_query(user_text)
        if not query:
            return ()
        try:
            return self.rule_store.disable_matching(query, workspace=self.workspace)
        except OSError:
            return ()

    def learn_after_turn(
        self,
        messages: Sequence[Message],
        result: UserTurnResult,
    ) -> tuple[BehaviorRule, ...]:
        user_text = _messages_text(messages)
        tool_names = _tool_names(result)
        final_text = ""
        try:
            response = result.final_step().response
            final_text = response.content.strip() or response.reasoning.strip()
        except Exception:
            final_text = ""
        self._append_interaction_trace(
            user_text=user_text,
            final_text=final_text,
            tool_names=tool_names,
            result=result,
        )
        rules: list[BehaviorRule] = []
        if self.settings.interaction_learning_enabled:
            for rule in extract_explicit_behavior_rules(
                user_text,
                workspace=self.workspace,
                source="deepmate_interaction",
            ):
                try:
                    rules.append(self.rule_store.upsert(rule))
                except OSError:
                    pass
        if self.computer_use_enabled:
            try:
                _write_computer_session(
                    self.data_dir,
                    session_id=self.session_id,
                    payload={
                        "session_id": self.session_id,
                        "workspace_hash": workspace_hash(self.workspace),
                        "profile": self.profile.name,
                        "computer_use_enabled": True,
                        "computer_learning_enabled": self.settings.computer_learning_enabled,
                        "task": self.computer_task or _preview(user_text, 220),
                        "tool_names": list(tool_names),
                        "updated_at": _timestamp(),
                    },
                )
            except OSError:
                pass
        return tuple(rules)

    def _append_interaction_trace(
        self,
        *,
        user_text: str,
        final_text: str,
        tool_names: Sequence[str],
        result: UserTurnResult,
    ) -> None:
        payload: dict[str, object] = {
            "kind": "deepmate_interaction",
            "session_id": self.session_id,
            "profile": self.profile.name,
            "workspace_hash": workspace_hash(self.workspace),
            "tool_names": list(tool_names),
            "errors": len(result.errors()),
            "reached_max_steps": result.reached_max_steps,
            "computer_use_enabled": self.computer_use_enabled,
            "computer_learning_enabled": self.settings.computer_learning_enabled,
            "evidence_enabled": self.settings.evidence_enabled,
        }
        if self.settings.evidence_enabled:
            payload["prompt_preview"] = _preview(user_text, 500)
            payload["final_preview"] = _preview(final_text, 500)
        else:
            payload["prompt_chars"] = len(user_text)
            payload["final_chars"] = len(final_text)
        try:
            self.trace_store.append(payload)
        except OSError:
            pass

    def status_text(self) -> str:
        enabled_rules = self.rule_store.enabled_rules()
        lines = [
            "Behavior learning",
            f"- Deepmate interaction learning: {'on' if self.settings.interaction_learning_enabled else 'off'}",
            f"- Learning evidence preview: {'on' if self.settings.evidence_enabled else 'off'}",
            f"- Long-term learning from Computer Use: {'on' if self.settings.computer_learning_enabled else 'off'}",
            f"- Computer Use: {'on' if self.computer_use_enabled else 'off'}",
            f"- Enabled behavior rules: {len(enabled_rules)}",
        ]
        if enabled_rules:
            lines.append("- Recent rules:")
            for rule in enabled_rules[-5:]:
                scope = "project" if rule.scope == "workspace" else "global"
                lines.append(f"  - [{scope}] {rule.text}")
        return "\n".join(lines)


def behavior_runtime_for_session(
    *,
    data_dir: str | Path,
    workspace: str | Path,
    profile: ProfileRef,
    session_id: str,
    interaction_learning_enabled: bool | None = None,
    computer_learning_enabled: bool | None = None,
    computer_use_enabled: bool = False,
) -> BehaviorRuntime:
    return BehaviorRuntime.create(
        data_dir=data_dir,
        workspace=workspace,
        profile=profile,
        session_id=session_id,
        interaction_learning_enabled=interaction_learning_enabled,
        computer_learning_enabled=computer_learning_enabled,
        computer_use_enabled=computer_use_enabled,
    )


def _messages_text(messages: Sequence[Message]) -> str:
    return "\n".join(
        message.content.strip()
        for message in messages
        if message.role == MessageRole.USER and message.content.strip()
    )


def _tool_names(result: UserTurnResult) -> tuple[str, ...]:
    names: list[str] = []
    for exchange in result.tool_exchanges:
        for request in exchange.tool_requests:
            if request.name.strip():
                names.append(request.name.strip())
        for tool_result in exchange.tool_results:
            if tool_result.name.strip():
                names.append(tool_result.name.strip())
    return tuple(dict.fromkeys(names))


def _safe_enabled_rules(store: BehaviorRuleStore) -> tuple[BehaviorRule, ...]:
    try:
        return store.enabled_rules()
    except OSError:
        return ()


def _write_computer_session(
    data_dir: Path,
    *,
    session_id: str,
    payload: Mapping[str, object],
) -> None:
    clean_id = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in session_id)
    path = data_dir / "computer" / "sessions" / f"{clean_id or 'session'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            existing = dict(loaded)
    except (OSError, json.JSONDecodeError):
        existing = {}
    merged = {**existing, **dict(payload)}
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _timestamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _preview(text: str, limit: int) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."
