"""Deterministic safety gates for higher-risk tool execution."""

from __future__ import annotations

import re
import unicodedata
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import Lock, local


class SafetyRiskLevel(StrEnum):
    """Small risk vocabulary for runtime safety decisions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DENIED = "denied"


class ApprovalDecision(StrEnum):
    """Approval result for actions that need user consent."""

    DENY = "deny"
    ALLOW_ONCE = "allow_once"
    ALLOW_FOR_SESSION = "allow_for_session"


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """Decision returned before executing a high-risk capability."""

    allowed: bool
    reason: str = ""
    requires_approval: bool = False
    requires_sandbox: bool = False
    risk_level: SafetyRiskLevel = SafetyRiskLevel.LOW
    approval_key: str = ""
    refs: tuple[str, ...] = field(default_factory=tuple)


ApprovalCallback = Callable[[SafetyDecision], ApprovalDecision]


@dataclass(slots=True)
class SessionApprovalCache:
    """Session-scoped approvals keyed by deterministic risk scope."""

    _allowed: set[str] = field(default_factory=set)
    _allow_once: set[str] = field(default_factory=set)
    approval_callback: ApprovalCallback | None = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _local: local = field(default_factory=local, init=False, repr=False)

    def allow_for_session(self, key: str) -> None:
        clean = key.strip()
        if clean:
            with self._lock:
                self._allowed.add(clean)

    def allow_once(self, key: str) -> None:
        """Allow a key for the rest of the current turn (cleared by reset_turn)."""
        clean = key.strip()
        if clean:
            local_allow_once = self._local_allow_once()
            if local_allow_once is not None:
                local_allow_once.add(clean)
                return
            with self._lock:
                self._allow_once.add(clean)

    def is_allowed(self, key: str) -> bool:
        clean = key.strip()
        if not clean:
            return False
        with self._lock:
            return clean in self._allowed

    def consume_once(self, key: str) -> bool:
        """Return whether a key is allowed for the current turn.

        Turn-scoped grants persist for the whole turn (so the same shell command
        prefix is not re-prompted on every call within one turn). They are cleared
        by reset_turn() at the next turn boundary, unlike session grants.
        """
        clean = key.strip()
        if not clean:
            return False
        local_allow_once = self._local_allow_once()
        if local_allow_once is not None and clean in local_allow_once:
            return True
        with self._lock:
            return clean in self._allow_once

    def reset_turn(self) -> None:
        """Drop turn-scoped (allow-once) grants at a turn boundary."""
        with self._lock:
            self._allow_once.clear()

    def request_approval(self, decision: SafetyDecision) -> SafetyDecision:
        """Ask the attached UI/channel for approval when available."""
        callback = self._current_approval_callback()
        if callback is None or not decision.requires_approval:
            return decision
        approval = callback(decision)
        return apply_session_approval(self, decision, approval)

    @contextmanager
    def scoped_approval_callback(self, callback: ApprovalCallback):
        """Install an approval callback only for the current worker thread.

        Remote channels can run multiple user turns concurrently. The legacy
        approval_callback field remains for single-session UIs, while this scope
        prevents one remote turn from overwriting another turn's callback or
        turn-scoped allow-once grants.
        """
        sentinel = object()
        previous_callback = getattr(self._local, "approval_callback", sentinel)
        previous_allow_once = getattr(self._local, "allow_once", sentinel)
        self._local.approval_callback = callback
        self._local.allow_once = set()
        try:
            yield
        finally:
            if previous_callback is sentinel:
                try:
                    delattr(self._local, "approval_callback")
                except AttributeError:
                    pass
            else:
                self._local.approval_callback = previous_callback
            if previous_allow_once is sentinel:
                try:
                    delattr(self._local, "allow_once")
                except AttributeError:
                    pass
            else:
                self._local.allow_once = previous_allow_once

    def _current_approval_callback(self) -> ApprovalCallback | None:
        callback = getattr(self._local, "approval_callback", None)
        return callback or self.approval_callback

    def _local_allow_once(self) -> set[str] | None:
        value = getattr(self._local, "allow_once", None)
        return value if isinstance(value, set) else None


@dataclass(frozen=True, slots=True)
class ToolSafetyPolicy:
    """Deterministic safety policy for shell/setup/future write surfaces."""

    workspace: Path
    shell_enabled: bool = False
    network_enabled: bool = False
    env_change_enabled: bool = False
    approval_cache: SessionApprovalCache | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())

    def check_shell_command(
        self,
        command: str,
        *,
        cwd: str | Path = ".",
        network: str = "off",
    ) -> SafetyDecision:
        """Return whether one shell command may run."""
        clean_command = command.strip()
        if not clean_command:
            return _denied("Shell command cannot be empty.", refs=("command=<empty>",))
        if not self.shell_enabled:
            if (
                self.approval_cache is not None
                and (
                    self.approval_cache.is_allowed("capability:shell")
                    or self.approval_cache.consume_once("capability:shell")
                )
            ):
                pass
            else:
                decision = SafetyDecision(
                    allowed=False,
                    requires_approval=True,
                    requires_sandbox=True,
                    risk_level=SafetyRiskLevel.HIGH,
                    approval_key="capability:shell",
                    reason="Shell execution requires approval for this session.",
                    refs=(f"command={_preview(clean_command)}",),
                )
                if self.approval_cache is not None:
                    approved = self.approval_cache.request_approval(decision)
                    if not approved.allowed:
                        return approved
                else:
                    return decision
        cwd_decision = self.check_workspace_cwd(cwd)
        if not cwd_decision.allowed:
            return cwd_decision
        hard_deny_reason = _hard_deny_command_reason(clean_command)
        if hard_deny_reason:
            return _denied(
                hard_deny_reason,
                refs=(f"command={_preview(clean_command)}",),
                requires_sandbox=True,
            )
        network_mode = network.strip().lower() if isinstance(network, str) else ""
        network_mode = network_mode or "off"
        if network_mode not in {"off", "on"}:
            return _denied(
                "Shell network mode must be off or on.",
                refs=(f"network={network}",),
                requires_sandbox=True,
            )
        if network_mode == "on" and not self.network_enabled:
            if (
                self.approval_cache is not None
                and (
                    self.approval_cache.is_allowed("capability:shell-network")
                    or self.approval_cache.consume_once("capability:shell-network")
                )
            ):
                pass
            else:
                reason = (
                    "Shell network access requires approval for this session."
                    if self.approval_cache is not None
                    else (
                        "Shell network access is disabled. In non-interactive CLI runs, "
                        "rerun with --allow-network if this is required."
                    )
                )
                decision = SafetyDecision(
                    allowed=False,
                    requires_approval=True,
                    requires_sandbox=True,
                    risk_level=SafetyRiskLevel.HIGH,
                    approval_key="capability:shell-network",
                    reason=reason,
                    refs=(f"command={_preview(clean_command)}", "network=on"),
                )
                if self.approval_cache is not None:
                    approved = self.approval_cache.request_approval(decision)
                    if not approved.allowed:
                        return approved
                else:
                    return decision

        risk = _command_risk(clean_command, network_mode)
        approval_key = _approval_key(clean_command, risk, network_mode)
        if risk == SafetyRiskLevel.LOW:
            return SafetyDecision(
                allowed=True,
                requires_sandbox=True,
                risk_level=risk,
                approval_key=approval_key,
                reason=f"{risk.value.title()}-risk workspace shell command is allowed.",
                refs=(f"command={_preview(clean_command)}", f"network={network_mode}"),
            )
        if _is_env_change_command(clean_command) and not self.env_change_enabled:
            if (
                self.approval_cache is not None
                and (
                    self.approval_cache.is_allowed(approval_key)
                    or self.approval_cache.consume_once(approval_key)
                )
            ):
                return SafetyDecision(
                    allowed=True,
                    requires_sandbox=True,
                    risk_level=risk,
                    approval_key=approval_key,
                    reason="Shell command allowed by session approval cache.",
                    refs=(
                        f"command={_preview(clean_command)}",
                        f"approval_key={approval_key}",
                        f"network={network_mode}",
                    ),
                )
            decision = SafetyDecision(
                allowed=False,
                requires_approval=True,
                requires_sandbox=True,
                risk_level=risk,
                approval_key=approval_key,
                reason=(
                    "Shell command may modify the package/environment state and requires "
                    "approval. In non-interactive CLI runs, rerun with --allow-env-change "
                    "if this is required."
                ),
                refs=(
                    f"command={_preview(clean_command)}",
                    f"approval_key={approval_key}",
                    f"network={network_mode}",
                ),
            )
            if self.approval_cache is not None:
                return self.approval_cache.request_approval(decision)
            return decision
        if (
            self.approval_cache is not None
            and (
                self.approval_cache.is_allowed(approval_key)
                or self.approval_cache.consume_once(approval_key)
            )
        ):
            return SafetyDecision(
                allowed=True,
                requires_sandbox=True,
                risk_level=risk,
                approval_key=approval_key,
                reason="Shell command allowed by session approval cache.",
                refs=(
                    f"command={_preview(clean_command)}",
                    f"approval_key={approval_key}",
                    f"network={network_mode}",
                ),
            )
        decision = SafetyDecision(
            allowed=False,
            requires_approval=True,
            requires_sandbox=True,
            risk_level=risk,
            approval_key=approval_key,
            reason=_approval_reason(risk, network_mode),
            refs=(
                f"command={_preview(clean_command)}",
                f"approval_key={approval_key}",
                f"network={network_mode}",
            ),
        )
        if self.approval_cache is not None:
            return self.approval_cache.request_approval(decision)
        return decision

    def check_workspace_cwd(self, cwd: str | Path) -> SafetyDecision:
        """Return whether cwd stays inside the workspace."""
        candidate = Path(cwd)
        path = candidate if candidate.is_absolute() else self.workspace / candidate
        resolved = path.resolve()
        if resolved != self.workspace and self.workspace not in resolved.parents:
            return _denied(
                "Shell cwd must stay inside the workspace.",
                refs=(f"cwd={cwd}", f"workspace={self.workspace}"),
                requires_sandbox=True,
            )
        if _sensitive_path(resolved, self.workspace):
            return _denied(
                "Shell cwd cannot be a sensitive workspace path.",
                refs=(f"cwd={cwd}",),
                requires_sandbox=True,
            )
        return SafetyDecision(
            allowed=True,
            reason="Shell cwd is inside the workspace.",
            refs=(f"cwd={resolved}",),
        )


def safe_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment with obvious secrets removed."""
    import os

    source = dict(os.environ if env is None else env)
    return {
        key: value
        for key, value in source.items()
        if not _secret_env_name(key)
    }


def apply_session_approval(
    cache: SessionApprovalCache,
    decision: SafetyDecision,
    approval: ApprovalDecision,
) -> SafetyDecision:
    """Apply one approval decision and return the post-approval decision."""
    if approval == ApprovalDecision.DENY:
        return decision
    if decision.approval_key:
        if approval == ApprovalDecision.ALLOW_FOR_SESSION:
            cache.allow_for_session(decision.approval_key)
        elif approval == ApprovalDecision.ALLOW_ONCE:
            # Turn-scoped: don't re-prompt for the same scope again this turn.
            cache.allow_once(decision.approval_key)
    return SafetyDecision(
        allowed=True,
        requires_approval=False,
        requires_sandbox=decision.requires_sandbox,
        risk_level=decision.risk_level,
        approval_key=decision.approval_key,
        reason="Action allowed by approval.",
        refs=decision.refs,
    )


def _denied(
    reason: str,
    *,
    refs: tuple[str, ...] = (),
    requires_sandbox: bool = False,
) -> SafetyDecision:
    return SafetyDecision(
        allowed=False,
        reason=reason,
        requires_sandbox=requires_sandbox,
        risk_level=SafetyRiskLevel.DENIED,
        refs=refs,
    )


def _hard_deny_command_reason(command: str) -> str:
    normalized = _normalized_command(command)
    shell_text = _shell_match_command(command)
    if _references_sensitive_workspace_path(shell_text):
        return "Shell command references a sensitive workspace path, which Deepmate will not run."
    if re.search(r"(^|[;&|`]\s*|\(\s*)sudo(\s|$)", shell_text):
        return "Shell command uses sudo, which Deepmate will not run."
    if re.search(r"(^|[;&|`]\s*|\(\s*)su(\s|$)", shell_text):
        return "Shell command changes user identity, which Deepmate will not run."
    if _executes_forbidden_identity_command(shell_text, "sudo"):
        return "Shell command uses sudo, which Deepmate will not run."
    if _executes_forbidden_identity_command(shell_text, "su"):
        return "Shell command changes user identity, which Deepmate will not run."
    alias_reason = _hard_deny_alias_reason(shell_text)
    if alias_reason:
        return alias_reason
    if _uses_dynamic_command_position(shell_text):
        return "Shell command invokes a command through dynamic shell expansion, which Deepmate will not run."
    remote_script_interpreters = (
        r"(?:ba)?sh|zsh|dash|fish|ksh|python3?|perl|ruby|node|php"
    )
    if (
        re.search(r"\b(curl|wget)\b", normalized)
        and "|"
        in normalized
        and re.search(rf"\|\s*(?:env\s+)?{remote_script_interpreters}(\s|$)", normalized)
    ):
        return "Remote script piped directly to shell is not allowed."
    if re.search(r"rm\s+(-[a-z]*[rf][a-z]*\s+)+/(?:\s|$)", normalized):
        return "Recursive removal of the filesystem root is not allowed."
    if ":(){:|:&};:" in normalized.replace(" ", ""):
        return "Fork bomb command is not allowed."
    if re.search(r"\bdd\b.*\bof=/dev/(disk|sd|nvme|rdisk)", normalized):
        return "Direct writes to block devices are not allowed."
    if re.search(r"\b(git\s+push\s+--force|git\s+push\s+-f)\b", normalized):
        return "Force-pushing from Agent shell is not allowed."
    if re.search(r"(~|\$home)/\.(zshrc|bashrc|profile|bash_profile)", normalized):
        return "Modifying shell profile files is not allowed."
    if ".git/hooks" in normalized:
        return "Modifying Git hooks is not allowed."
    return ""


def _command_risk(command: str, network_mode: str) -> SafetyRiskLevel:
    normalized = _normalized_command(command)
    if network_mode == "on":
        return SafetyRiskLevel.HIGH
    if re.match(
        r"^(python3? -m unittest|python3? -m pytest|pytest|npm test|npm run test|"
        r"pnpm test|yarn test|git status|git diff|git log|git show|ls|pwd)\b",
        normalized,
    ):
        if normalized.startswith("ls") and _ls_can_show_hidden_paths(normalized):
            return SafetyRiskLevel.MEDIUM
        return SafetyRiskLevel.LOW
    if _is_env_change_command(normalized):
        return SafetyRiskLevel.HIGH
    if _uses_dynamic_shell_execution(command):
        return SafetyRiskLevel.HIGH
    return SafetyRiskLevel.MEDIUM


def _is_env_change_command(command: str) -> bool:
    normalized = _normalized_command(command)
    return bool(
        re.match(
            r"^((pip3?|python3? -m pip)\s+install|npm install|pnpm install|"
            r"yarn install|brew install)\b",
            normalized,
        )
    )


def _approval_key(command: str, risk: SafetyRiskLevel, network_mode: str) -> str:
    if risk == SafetyRiskLevel.LOW:
        return ""
    if _is_env_change_command(command):
        return "capability:env_change"
    if network_mode == "on":
        return "capability:shell-network"
    if risk == SafetyRiskLevel.HIGH:
        return "capability:shell-high"
    return "capability:shell-medium"


def _approval_reason(risk: SafetyRiskLevel, network_mode: str) -> str:
    if network_mode == "on":
        return "Shell command needs network access and requires approval."
    if risk == SafetyRiskLevel.HIGH:
        return "Shell command may modify the environment and requires approval."
    return "Shell command is outside the low-risk validation set and requires approval."


def _ls_can_show_hidden_paths(command: str) -> bool:
    return bool(
        re.search(r"(^|\s)-[a-z]*a[a-z]*($|\s)", command)
        or re.search(r"(^|\s)--all($|\s)", command)
        or re.search(r"(^|\s)--almost-all($|\s)", command)
    )


def _command_prefix(command: str) -> str:
    parts = _normalized_command(command).split()
    return " ".join(parts[:3]) if parts else ""


def _normalized_command(command: str) -> str:
    return " ".join(_normalize_shell_text(command).strip().lower().split())


def _shell_match_command(command: str) -> str:
    """Normalize for shell-deny matching while preserving command separators."""
    text = _normalize_shell_text(command).strip().lower().replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s*\n+\s*", "; ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\s*([;&|()])\s*", r"\1", text)
    return text


def _uses_dynamic_shell_execution(command: str) -> bool:
    text = _shell_match_command(command)
    if re.search(r"(^|[;&|])\s*[a-z_][a-z0-9_]*=", text):
        return True
    if re.search(r"(^|[;&|])\s*(\$[{(]?|`)", text):
        return True
    if "$(" in text or "`" in text:
        return True
    return False


def _uses_dynamic_command_position(shell_text: str) -> bool:
    return bool(
        re.search(
            r"(^|[;&|()`])\s*(?:\$\(|\$\{?[a-z_][a-z0-9_]*\}?)",
            shell_text,
        )
        or re.search(
            r"(^|[;&|()`])\s*(?:[a-z_][a-z0-9_]*=[^;&|()`\s]*\s+)+"
            r"(?:\$\(|\$\{?[a-z_][a-z0-9_]*\}?)",
            shell_text,
        )
    )


def _references_sensitive_workspace_path(shell_text: str) -> bool:
    sensitive_segments = (
        ".aws",
        ".env",
        ".git",
        ".hg",
        ".ssh",
        ".svn",
        ".npmrc",
        ".pypirc",
        "var",
    )
    suffixes = (".pem", ".key", ".p12", ".pfx")
    if re.search(r"(^|[/'\"\s(])\.env(?:\.|[/\"'\s)]|$)", shell_text):
        return True
    if re.search(r"(^|[/'\"\s(])\.(?:aws|git|hg|ssh|svn)(?:[/\"'\s)]|$)", shell_text):
        return True
    if re.search(r"(^|[/'\"\s(])\.(?:npmrc|pypirc)(?:[/\"'\s)]|$)", shell_text):
        return True
    if re.search(r"(^|[/'\"\s(])var(?:[/\"'\s)]|$)", shell_text):
        return True
    if re.search(r"\.(?:pem|key|p12|pfx)(?:[/\"'\s)]|$)", shell_text):
        return True
    tokens = re.findall(r"""(?:"[^"]*"|'[^']*'|[^\s;&|()<>]+)""", shell_text)
    quoted_values = re.findall(r"""["']([^"']+)["']""", shell_text)
    candidates = (*tokens, *quoted_values)
    for token in candidates:
        clean = token.strip("\"'")
        if not clean or clean.startswith("-"):
            continue
        parts = tuple(part for part in re.split(r"/+", clean) if part and part != ".")
        if any(part in sensitive_segments or part.startswith(".env.") for part in parts):
            return True
        if clean.endswith(suffixes):
            return True
    return False


def _hard_deny_alias_reason(shell_text: str) -> str:
    aliases: dict[str, str] = {}
    for match in re.finditer(r"(^|[;&|])\s*([a-z_][a-z0-9_]*)=(sudo|su)(?=$|[;&|\s])", shell_text):
        aliases[match.group(2)] = match.group(3)
    if not aliases:
        return ""
    for name, target in aliases.items():
        if re.search(rf"(^|[;&|])\s*(\$\{{{re.escape(name)}\}}|\${re.escape(name)})(\s|$)", shell_text):
            if target == "sudo":
                return "Shell command uses sudo, which Deepmate will not run."
            return "Shell command changes user identity, which Deepmate will not run."
    return ""


def _normalize_shell_text(command: str) -> str:
    return unicodedata.normalize("NFKC", command)


def _executes_forbidden_identity_command(shell_text: str, command_name: str) -> bool:
    escaped = re.escape(command_name)
    wrappers = r"(command|exec|env|xargs|nice|nohup|time|timeout)"
    if re.search(rf"(^|[;&|()`]\s*){wrappers}\b[^;&|()`]*\b{escaped}(\s|$)", shell_text):
        return True
    if re.search(rf"(^|[;&|()`]\s*)find\b[^;&|()`]*\s-exec\s+{escaped}(\s|$)", shell_text):
        return True
    return False


def _preview(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _sensitive_path(path: Path, workspace: Path) -> bool:
    denied_names = {
        ".aws",
        ".env",
        ".git",
        ".hg",
        ".ssh",
        ".svn",
        ".npmrc",
        ".pypirc",
        "var",
    }
    denied_suffixes = (".key", ".p12", ".pem", ".pfx")
    parts = path.relative_to(workspace).parts if path != workspace else ()
    return (
        any(part in denied_names or part.startswith(".env.") for part in parts)
        or path.suffix.lower() in denied_suffixes
    )


def _secret_env_name(name: str) -> bool:
    upper = name.upper()
    if upper == "SSH_AUTH_SOCK":
        return False
    exact_names = {
        "DATABASE_URL",
        "REDIS_URL",
        "MONGO_URL",
        "MONGODB_URI",
        "GH_PAT",
        "GITHUB_PAT",
        "GITLAB_PAT",
    }
    suffixes = (
        "_DSN",
        "_DATABASE_URL",
        "_CONNECTION_STRING",
        "_CONN_STRING",
        "_PRIVATE_KEY",
        "_CLIENT_SECRET",
        "_ACCESS_TOKEN",
        "_REFRESH_TOKEN",
        "_SESSION",
        "_COOKIE",
        "_PAT",
    )
    if upper in exact_names or upper.endswith(suffixes):
        return True
    markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    return any(
        re.search(rf"(^|_){re.escape(marker)}($|_)", upper) is not None
        for marker in markers
    )
