"""Structured behavior rule storage.

This module deliberately keeps learned behavior out of profile Markdown.  The
profile memory curator still owns user.md/memory.md; behavior rules are a small
machine-readable layer that can be matched per turn and injected through the
turn-tail path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_RULE_BUDGET_CHARS = 1_500
MIN_RULE_CONFIDENCE = 0.55


@dataclass(frozen=True, slots=True)
class BehaviorSettings:
    """User-controlled behavior learning switches."""

    interaction_learning_enabled: bool = True
    computer_learning_enabled: bool = False
    evidence_enabled: bool = True


class BehaviorSettingsStore:
    """Small JSON settings store under Deepmate's private data dir."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "BehaviorSettingsStore":
        return cls(Path(data_dir) / "behavior" / "settings.json")

    def load(self) -> BehaviorSettings:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return BehaviorSettings()
        if not isinstance(payload, Mapping):
            return BehaviorSettings()
        return BehaviorSettings(
            interaction_learning_enabled=_bool(
                payload.get("interaction_learning_enabled"),
                True,
            ),
            computer_learning_enabled=_bool(
                payload.get("computer_learning_enabled"),
                False,
            ),
            evidence_enabled=_bool(payload.get("evidence_enabled"), True),
        )

    def save(self, settings: BehaviorSettings) -> None:
        payload = {
            "interaction_learning_enabled": settings.interaction_learning_enabled,
            "computer_learning_enabled": settings.computer_learning_enabled,
            "evidence_enabled": settings.evidence_enabled,
        }
        _atomic_write_json(self.path, payload)

    def set_interaction_learning(self, enabled: bool) -> BehaviorSettings:
        settings = replace(
            self.load(),
            interaction_learning_enabled=enabled,
            evidence_enabled=enabled,
        )
        self.save(settings)
        return settings

    def set_computer_learning(self, enabled: bool) -> BehaviorSettings:
        settings = replace(self.load(), computer_learning_enabled=enabled)
        self.save(settings)
        return settings


@dataclass(frozen=True, slots=True)
class BehaviorRule:
    """One learned, turn-tail-injectable behavior rule."""

    rule_id: str
    text: str
    scope: str = "global"
    workspace_hash: str = ""
    surfaces: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source: str = "deepmate_interaction"
    confidence: float = 0.8
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""
    evidence: str = ""

    def normalized(self) -> "BehaviorRule":
        now = _timestamp()
        clean_text = _clean_rule_text(self.text)
        clean_scope = self.scope if self.scope in {"global", "workspace"} else "global"
        return replace(
            self,
            rule_id=self.rule_id.strip()
            or _rule_id(clean_text, clean_scope, self.workspace_hash),
            text=clean_text,
            scope=clean_scope,
            surfaces=tuple(_clean_items(self.surfaces)),
            tags=tuple(_clean_items(self.tags)),
            source=self.source.strip() or "deepmate_interaction",
            confidence=min(max(float(self.confidence), 0.0), 1.0),
            created_at=self.created_at.strip() or now,
            updated_at=self.updated_at.strip() or now,
            evidence=_preview(self.evidence, 500),
        )

    def to_json(self) -> dict[str, object]:
        rule = self.normalized()
        return {
            "rule_id": rule.rule_id,
            "text": rule.text,
            "scope": rule.scope,
            "workspace_hash": rule.workspace_hash,
            "surfaces": list(rule.surfaces),
            "tags": list(rule.tags),
            "source": rule.source,
            "confidence": rule.confidence,
            "enabled": rule.enabled,
            "created_at": rule.created_at,
            "updated_at": rule.updated_at,
            "evidence": rule.evidence,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "BehaviorRule | None":
        text = _text(payload.get("text"))
        if not text:
            return None
        return cls(
            rule_id=_text(payload.get("rule_id")),
            text=text,
            scope=_text(payload.get("scope")) or "global",
            workspace_hash=_text(payload.get("workspace_hash")),
            surfaces=tuple(_strings(payload.get("surfaces"))),
            tags=tuple(_strings(payload.get("tags"))),
            source=_text(payload.get("source")) or "deepmate_interaction",
            confidence=_float(payload.get("confidence"), 0.8),
            enabled=_bool(payload.get("enabled"), True),
            created_at=_text(payload.get("created_at")),
            updated_at=_text(payload.get("updated_at")),
            evidence=_text(payload.get("evidence")),
        ).normalized()

    def matches_workspace(self, current_workspace_hash: str) -> bool:
        if self.scope == "global":
            return True
        return bool(self.workspace_hash and self.workspace_hash == current_workspace_hash)


class BehaviorRuleStore:
    """Append-only JSONL store with latest-version reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_data_dir(
        cls,
        data_dir: str | Path,
        profile_name: str,
    ) -> "BehaviorRuleStore":
        clean_profile = _safe_name(profile_name or "default")
        return cls(Path(data_dir) / "behavior" / "rules" / f"{clean_profile}.jsonl")

    def list_rules(self) -> tuple[BehaviorRule, ...]:
        latest: dict[str, BehaviorRule] = {}
        for record in self._iter_records():
            rule = BehaviorRule.from_json(record)
            if rule is None:
                continue
            latest[rule.rule_id] = rule
        return tuple(
            sorted(
                latest.values(),
                key=lambda rule: (rule.updated_at, rule.created_at, rule.rule_id),
            )
        )

    def enabled_rules(self) -> tuple[BehaviorRule, ...]:
        return tuple(rule for rule in self.list_rules() if rule.enabled)

    def append(self, rule: BehaviorRule) -> BehaviorRule:
        normalized = rule.normalized()
        self._append_json(normalized.to_json())
        return normalized

    def upsert(self, rule: BehaviorRule) -> BehaviorRule:
        normalized = rule.normalized()
        duplicate = self.find_similar(
            normalized.text,
            normalized.scope,
            workspace_hash_value=normalized.workspace_hash,
        )
        if duplicate is not None:
            normalized = replace(
                normalized,
                rule_id=duplicate.rule_id,
                created_at=duplicate.created_at,
                updated_at=_timestamp(),
                confidence=max(duplicate.confidence, normalized.confidence),
                tags=tuple(dict.fromkeys((*duplicate.tags, *normalized.tags))),
                surfaces=tuple(
                    dict.fromkeys((*duplicate.surfaces, *normalized.surfaces))
                ),
            )
        return self.append(normalized)

    def find_similar(
        self,
        text: str,
        scope: str = "",
        *,
        workspace_hash_value: str = "",
    ) -> BehaviorRule | None:
        clean = _rule_key(text)
        for rule in reversed(self.list_rules()):
            if scope and rule.scope != scope:
                continue
            if (
                scope == "workspace"
                and workspace_hash_value
                and rule.workspace_hash != workspace_hash_value
            ):
                continue
            if _rule_key(rule.text) == clean:
                return rule
        return None

    def disable_matching(
        self,
        query: str,
        *,
        workspace: str | Path | None = None,
    ) -> tuple[BehaviorRule, ...]:
        rules = self.enabled_rules()
        if not rules:
            return ()
        clean_query = query.strip()
        current_hash = workspace_hash(workspace) if workspace is not None else ""
        if _looks_like_forget_all(clean_query):
            candidates = rules
        else:
            candidates = match_behavior_rules(
                rules,
                clean_query,
                workspace_hash_value=current_hash,
                max_rules=8,
                budget_chars=3_000,
                min_confidence=0.0,
            )
        disabled: list[BehaviorRule] = []
        for rule in candidates:
            updated = replace(rule, enabled=False, updated_at=_timestamp())
            self.append(updated)
            disabled.append(updated)
        return tuple(disabled)

    def _iter_records(self) -> Iterable[Mapping[str, object]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        records: list[Mapping[str, object]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                records.append(payload)
        return tuple(records)

    def _append_json(self, payload: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


class BehaviorTraceStore:
    """Daily JSONL behavior trace store.

    These records are private evidence and are intentionally separate from
    user.md/memory.md.  They are bounded previews, not raw tool output dumps.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def in_data_dir(cls, data_dir: str | Path) -> "BehaviorTraceStore":
        return cls(Path(data_dir) / "behavior" / "activity")

    def append(self, payload: Mapping[str, object], *, at: str | None = None) -> Path:
        recorded_at = at or _timestamp()
        date = recorded_at[:10]
        path = self.root / f"{date}.jsonl"
        record = {"recorded_at": recorded_at, **dict(payload)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return path


def extract_explicit_behavior_rules(
    text: str,
    *,
    workspace: str | Path,
    source: str = "deepmate_interaction",
) -> tuple[BehaviorRule, ...]:
    """Extract only explicit user-authored behavior preferences."""
    clean = " ".join(text.strip().split())
    if not clean:
        return ()
    candidates: list[str] = []
    lowered = clean.lower()
    for pattern in _RULE_PATTERNS:
        for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
            candidate = _clean_rule_text(match.group(1))
            if candidate:
                candidates.append(candidate)
    if not candidates and _starts_with_rule_marker(lowered):
        candidates.append(_clean_rule_text(clean))
    rules: list[BehaviorRule] = []
    scope = "workspace" if _looks_project_scoped(clean) else "global"
    whash = workspace_hash(workspace) if scope == "workspace" else ""
    for candidate in tuple(dict.fromkeys(candidates)):
        if not _useful_rule(candidate):
            continue
        rules.append(
            BehaviorRule(
                rule_id=_rule_id(candidate, scope, whash),
                text=candidate,
                scope=scope,
                workspace_hash=whash,
                tags=_infer_tags(candidate),
                source=source,
                confidence=0.86,
                evidence=_preview(clean, 500),
            ).normalized()
        )
    return tuple(rules)


def extract_forget_query(text: str) -> str:
    """Return the target of an explicit forget/do-not-learn request."""
    clean = " ".join(text.strip().split())
    lowered = clean.lower()
    if not clean:
        return ""
    markers = (
        "不要记住",
        "别记住",
        "不要学习",
        "别学习",
        "忘记",
        "forget",
        "do not learn",
        "don't learn",
    )
    if not any(marker in lowered for marker in markers) and not any(
        marker in clean for marker in markers
    ):
        return ""
    for marker in markers:
        index = lowered.find(marker.lower())
        if index >= 0:
            return clean[index + len(marker) :].strip(" ：:，,。.")
        index = clean.find(marker)
        if index >= 0:
            return clean[index + len(marker) :].strip(" ：:，,。.")
    return clean


def match_behavior_rules(
    rules: Sequence[BehaviorRule],
    query: str,
    *,
    workspace_hash_value: str,
    max_rules: int = 4,
    budget_chars: int = DEFAULT_RULE_BUDGET_CHARS,
    min_confidence: float = MIN_RULE_CONFIDENCE,
) -> tuple[BehaviorRule, ...]:
    clean_query = query.strip()
    query_tokens = set(_tokens(clean_query))
    scored: list[tuple[float, BehaviorRule]] = []
    for rule in rules:
        if not rule.enabled or rule.confidence < min_confidence:
            continue
        if not rule.matches_workspace(workspace_hash_value):
            continue
        rule_tokens = set(_tokens(" ".join((rule.text, " ".join(rule.tags)))))
        overlap = len(query_tokens & rule_tokens)
        score = rule.confidence + (0.08 * overlap)
        if rule.scope == "workspace":
            score += 0.08
        if overlap == 0 and rule.confidence < 0.8:
            continue
        scored.append((score, rule))
    selected: list[BehaviorRule] = []
    used_chars = 0
    for _score, rule in sorted(scored, key=lambda item: item[0], reverse=True):
        cost = len(rule.text) + 32
        if selected and used_chars + cost > budget_chars:
            continue
        selected.append(rule)
        used_chars += cost
        if len(selected) >= max_rules:
            break
    return tuple(selected)


def render_behavior_turn_tail(rules: Sequence[BehaviorRule]) -> str:
    if not rules:
        return ""
    lines = [
        "<deepmate_behavior_context>",
        "Use these learned preferences only when they are relevant to the current task.",
    ]
    for rule in rules:
        scope = "this project" if rule.scope == "workspace" else "all projects"
        lines.append(f"- ({scope}) {rule.text}")
    lines.append(
        "If the user corrects or contradicts these preferences, follow the newest user instruction."
    )
    lines.append("</deepmate_behavior_context>")
    return "\n".join(lines)


def workspace_hash(workspace: str | Path | None) -> str:
    if workspace is None:
        return ""
    try:
        text = str(Path(workspace).resolve())
    except OSError:
        text = str(workspace)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_RULE_PATTERNS = (
    r"(?:以后|今后|以后都|后续|之后)\s*([^。；;\n]{4,160})",
    r"(?:记住|请记住)\s*([^。；;\n]{4,160})",
    r"(?:我喜欢|我更喜欢|我希望|我不喜欢)\s*([^。；;\n]{4,160})",
    r"(?:always|never|prefer|remember that)\s+([^.;\n]{4,180})",
)


def _starts_with_rule_marker(lowered: str) -> bool:
    return lowered.startswith(("always ", "never ", "prefer ", "以后", "记住"))


def _looks_project_scoped(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "这个项目",
            "本项目",
            "当前项目",
            "this project",
            "this repo",
            "this repository",
            "workspace",
        )
    )


def _useful_rule(text: str) -> bool:
    if len(text) < 4 or len(text) > 220:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in ("api key", "password", "token=", "secret")):
        return False
    if lowered in {"ok", "好的", "可以", "不用"}:
        return False
    return True


def _infer_tags(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    tags: list[str] = []
    markers = {
        "review": ("review", "代码审查", "检查"),
        "reply_style": ("语气", "表达", "回复", "tone", "style"),
        "code": ("代码", "code", "test", "测试"),
        "computer": ("电脑", "浏览器", "browser", "screenshot", "computer"),
        "memory": ("记住", "记忆", "memory"),
    }
    for tag, values in markers.items():
        if any(value in lowered for value in values):
            tags.append(tag)
    return tuple(tags)


def _looks_like_forget_all(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "全部",
            "所有",
            "all",
            "everything",
            "偏好",
            "习惯",
            "规则",
        )
    )


def _tokens(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", lowered)
    return tuple(dict.fromkeys(words))


def _rule_key(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _rule_id(text: str, scope: str, workspace_hash_value: str = "") -> str:
    scope_key = (
        f"{scope}:{workspace_hash_value}" if scope == "workspace" else scope
    )
    digest = hashlib.sha256(
        f"{scope_key}\n{_rule_key(text)}".encode("utf-8")
    ).hexdigest()
    return f"br_{digest[:16]}"


def _clean_rule_text(text: str) -> str:
    clean = " ".join(text.strip().split())
    clean = clean.strip(" ：:，,。.;；")
    prefixes = ("就是", "是", "要", "需要", "请")
    for prefix in prefixes:
        if clean.startswith(prefix) and len(clean) > len(prefix) + 3:
            clean = clean[len(prefix) :].strip()
    return _preview(clean, 220)


def _clean_items(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _preview(text: str, limit: int) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _timestamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean or "default"


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
