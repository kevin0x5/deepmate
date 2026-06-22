"""Lightweight sandbox runner for external process execution."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from deepmate.runtime.process_env import subprocess_environment
from deepmate.runtime.safety import safe_environment

SENSITIVE_WORKSPACE_NAMES = (
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
SENSITIVE_WORKSPACE_GLOBS = (".env.*", "*.pem", "*.key", "*.p12", "*.pfx")


class SandboxMode(StrEnum):
    """Runtime sandbox availability policy."""

    AUTO = "auto"
    REQUIRE = "require"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Execution policy for one external command."""

    workspace: Path
    cwd: Path
    network_enabled: bool = False
    mode: SandboxMode = SandboxMode.AUTO

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        object.__setattr__(self, "cwd", Path(self.cwd).resolve())


@dataclass(frozen=True, slots=True)
class SandboxRunResult:
    """Result of one external command execution."""

    stdout: str
    stderr: str
    exit_code: int
    backend: str
    sandboxed: bool
    refs: tuple[str, ...] = ()

    def output_text(self) -> str:
        """Return model-facing shell output with stderr included when present."""
        parts = []
        stdout = self.stdout.rstrip()
        stderr = self.stderr.rstrip()
        if stdout.strip():
            parts.append(stdout)
        if stderr.strip():
            parts.append("[stderr]\n" + stderr)
        if not parts:
            parts.append(f"[process exited with code {self.exit_code}]")
        return "\n\n".join(parts)


@dataclass(frozen=True, slots=True)
class SandboxStatus:
    """Local sandbox diagnostic status."""

    platform: str
    mode: SandboxMode
    backend: str
    available: bool
    sandboxed: bool
    workspace: Path
    network_default: str
    warning: str = ""

    def refs(self) -> tuple[str, ...]:
        return (
            f"platform={self.platform}",
            f"sandbox_mode={self.mode.value}",
            f"sandbox_backend={self.backend}",
            f"sandbox_available={str(self.available).lower()}",
            f"sandboxed={str(self.sandboxed).lower()}",
            f"network_default={self.network_default}",
        )


class SandboxRunner:
    """Run commands through the best available lightweight sandbox backend."""

    def run(
        self,
        command: str,
        policy: SandboxPolicy,
        *,
        timeout_seconds: int = 120,
    ) -> SandboxRunResult:
        """Execute command according to policy."""
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        backend = self.backend(policy)
        if backend == "unavailable":
            raise RuntimeError("sandbox backend is required but unavailable")
        if backend == "permission-only" and not policy.network_enabled:
            raise RuntimeError(
                "OS sandbox backend is unavailable; refusing to run a network-disabled "
                "shell command without enforceable network isolation. Enable network for "
                "this command or install a supported sandbox backend."
            )
        launch = _backend_launch(command, policy, backend)
        try:
            completed = subprocess.run(
                launch.argv,
                cwd=policy.cwd,
                env=subprocess_environment(safe_environment()),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return SandboxRunResult(
                stdout=stdout,
                stderr=(stderr + f"\nCommand timed out after {timeout_seconds}s").strip(),
                exit_code=124,
                backend=backend,
                sandboxed=backend not in {"permission-only", "off"},
                refs=(f"sandbox_backend={backend}", "shell_timeout=true"),
            )
        finally:
            launch.cleanup()
        return SandboxRunResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            backend=backend,
            sandboxed=backend not in {"permission-only", "off"},
            refs=(
                f"sandbox_backend={backend}",
                f"sandboxed={str(backend not in {'permission-only', 'off'}).lower()}",
                f"network={str(policy.network_enabled).lower()}",
            ),
        )

    def backend(self, policy: SandboxPolicy) -> str:
        """Return the backend selected for policy without executing it."""
        if policy.mode == SandboxMode.OFF:
            return "off"
        system = platform.system().lower()
        if system == "darwin" and shutil.which("sandbox-exec"):
            return "sandbox-exec"
        if system == "linux" and shutil.which("bwrap"):
            return "bwrap"
        if policy.mode == SandboxMode.REQUIRE:
            return "unavailable"
        return "permission-only"

    def status(self, policy: SandboxPolicy) -> SandboxStatus:
        """Return local sandbox availability without executing a command."""
        backend = self.backend(policy)
        warning = ""
        if backend == "permission-only":
            warning = (
                "No supported OS sandbox backend was found; external commands "
                "will use permission-only fallback."
            )
        elif backend == "unavailable":
            warning = "Sandbox backend is required but unavailable."
        elif backend == "off":
            warning = "OS sandbox is disabled by --sandbox off."
        return SandboxStatus(
            platform=platform.system() or "unknown",
            mode=policy.mode,
            backend=backend,
            available=backend not in {"permission-only", "unavailable", "off"},
            sandboxed=backend not in {"permission-only", "unavailable", "off"},
            workspace=policy.workspace,
            network_default="on" if policy.network_enabled else "off",
            warning=warning,
        )


@dataclass(frozen=True, slots=True)
class _SandboxLaunch:
    argv: list[str]
    cleanup_path: Path | None = None
    cleanup_paths: tuple[Path, ...] = ()

    def cleanup(self) -> None:
        paths = self.cleanup_paths
        if self.cleanup_path is not None:
            paths = (*paths, self.cleanup_path)
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass


def _backend_launch(
    command: str,
    policy: SandboxPolicy,
    backend: str,
) -> _SandboxLaunch:
    if backend == "sandbox-exec":
        return _sandbox_exec_launch(command, policy)
    if backend == "bwrap":
        return _bwrap_launch(command, policy)
    if backend not in {"permission-only", "off"}:
        raise RuntimeError(f"unknown sandbox backend: {backend}")
    return _SandboxLaunch(["/bin/sh", "-lc", command])


def _sandbox_exec_launch(command: str, policy: SandboxPolicy) -> _SandboxLaunch:
    profile = _seatbelt_profile(policy)
    temp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    try:
        temp.write(profile)
        temp.flush()
    finally:
        temp.close()
    return _SandboxLaunch(
        ["sandbox-exec", "-f", temp.name, "/bin/sh", "-lc", command],
        cleanup_path=Path(temp.name),
    )


def _seatbelt_profile(policy: SandboxPolicy) -> str:
    workspace = _sbpl_string(policy.workspace)
    cwd = _sbpl_string(policy.cwd)
    network = "(allow network*)\n" if policy.network_enabled else "(deny network*)\n"
    readable_paths = tuple(
        f"(allow file-read* (subpath {_sbpl_string(path)}))"
        for path in _seatbelt_readable_paths(policy)
    )
    deny_lines = tuple(
        f"(deny file-read* file-write* {_seatbelt_path_filter(path)})"
        for path in _sensitive_workspace_paths(policy.workspace)
    )
    git_path = policy.workspace / ".git"
    home_deny_lines = tuple(
        f"(deny file-read* file-write* {_seatbelt_path_filter(path)})"
        for path in _sensitive_home_paths()
    )
    return "\n".join(
        (
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow file-read-data)",
            "(allow file-read-metadata)",
            *readable_paths,
            f"(allow file-write* (subpath {workspace}))",
            f"(allow file-write* (subpath {cwd}))",
            f"(deny file-write* {_seatbelt_path_filter(git_path)})",
            *deny_lines,
            *home_deny_lines,
            network.rstrip(),
            "",
        )
    )


def _bwrap_launch(command: str, policy: SandboxPolicy) -> _SandboxLaunch:
    mask_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    try:
        mask_file.write("")
        mask_file.flush()
    finally:
        mask_file.close()
    return _SandboxLaunch(
        _bwrap_argv(command, policy, file_mask=Path(mask_file.name)),
        cleanup_paths=(Path(mask_file.name),),
    )


def _bwrap_argv(
    command: str,
    policy: SandboxPolicy,
    *,
    file_mask: Path | None = None,
) -> list[str]:
    argv = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
    ]
    if not policy.network_enabled:
        argv.append("--unshare-net")
    argv.extend(
        [
            "--dev",
            "/dev",
            "--proc",
            "/proc",
        ]
    )
    for path in (Path("/bin"), Path("/usr"), Path("/lib"), Path("/lib64")):
        if path.exists():
            argv.extend(("--ro-bind", str(path), str(path)))
    argv.extend(("--bind", str(policy.workspace), str(policy.workspace)))
    git_path = policy.workspace / ".git"
    if git_path.exists():
        argv.extend(("--ro-bind", str(git_path), str(git_path)))
    for path in _sensitive_workspace_paths(policy.workspace):
        if path.is_dir():
            argv.extend(("--tmpfs", str(path)))
        elif path.is_file() and file_mask is not None:
            argv.extend(("--ro-bind", str(file_mask), str(path)))
    argv.extend(
        [
            "--tmpfs",
            "/tmp",
            "--chdir",
            str(policy.cwd),
            "/bin/sh",
            "-lc",
            command,
        ]
    )
    return argv


def _sbpl_string(path: Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _seatbelt_path_filter(path: Path) -> str:
    if path.is_dir():
        return f"(subpath {_sbpl_string(path)})"
    return f"(regex #{_sbpl_regex(_resolved_path_regex(path))})"


def _resolved_path_regex(path: Path) -> str:
    resolved = _safe_resolve(path)
    variants = {str(resolved)}
    if str(resolved).startswith("/private/"):
        variants.add(str(resolved)[len("/private") :])
    else:
        private_path = Path("/private") / str(resolved).lstrip("/")
        if private_path.exists() or str(resolved).startswith("/var/"):
            variants.add(str(private_path))
    escaped = sorted(re.escape(variant) for variant in variants)
    return f"^({'|'.join(escaped)})(/.*)?$"


def _sbpl_regex(pattern: str) -> str:
    escaped = pattern.replace('"', '\\"')
    return f'"{escaped}"'


def _seatbelt_readable_paths(policy: SandboxPolicy) -> tuple[Path, ...]:
    paths = [
        policy.workspace,
        policy.cwd,
        Path("/bin"),
        Path("/sbin"),
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path("/System"),
        Path("/Library"),
        Path("/opt/homebrew"),
        Path("/private/tmp"),
        Path("/tmp"),
        Path("/dev"),
    ]
    return tuple(dict.fromkeys(path.resolve() for path in paths if path.exists()))


def _sensitive_workspace_paths(workspace: Path) -> tuple[Path, ...]:
    paths = [workspace / name for name in SENSITIVE_WORKSPACE_NAMES if name != ".git"]
    for pattern in SENSITIVE_WORKSPACE_GLOBS:
        paths.extend(_safe_glob(workspace, pattern))
    return tuple(dict.fromkeys(_safe_resolve(path) for path in paths))


def _sensitive_home_paths() -> tuple[Path, ...]:
    try:
        home = Path.home().resolve()
    except RuntimeError:
        return ()
    names = (".aws", ".ssh", ".gnupg", ".config/gcloud", ".docker", ".kube")
    paths = [home / name for name in names]
    paths.extend(_safe_glob(home, ".env*"))
    suffix_paths = []
    for suffix in ("*.pem", "*.key", "*.p12", "*.pfx"):
        suffix_paths.extend(_safe_glob(home, suffix))
    paths.extend(suffix_paths)
    return tuple(dict.fromkeys(_safe_resolve(path) for path in paths if path.exists()))


def _safe_glob(root: Path, pattern: str) -> tuple[Path, ...]:
    try:
        return tuple(root.glob(pattern))
    except OSError:
        return ()


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.resolve(strict=False)
