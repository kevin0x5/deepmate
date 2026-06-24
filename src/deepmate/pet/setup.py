"""One-time Electron runtime setup for the desktop pet."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from deepmate.pet.electron_host import electron_pet_command, pet_runtime_ui_dir
from deepmate.runtime.safety import safe_environment

PET_SETUP_TIMEOUT_SECONDS = 10 * 60


@dataclass(frozen=True, slots=True)
class PetSetupStatus:
    """Current desktop pet runtime setup status."""

    ready: bool
    ui_dir: Path
    electron_binary: Path | None
    message: str


@dataclass(frozen=True, slots=True)
class PetSetupResult:
    """Result of preparing the desktop pet Electron runtime."""

    ok: bool
    ui_dir: Path
    message: str
    stdout: str = ""
    stderr: str = ""


ProgressCallback = Callable[[str], None]


def pet_setup_status(data_dir: str | Path | None) -> PetSetupStatus:
    """Return whether the Electron pet frontend can be started."""
    ui_dir = pet_runtime_ui_dir(data_dir)
    electron = _electron_binary_for_ui(ui_dir)
    command = electron_pet_command(data_dir)
    if command is not None:
        return PetSetupStatus(
            ready=True,
            ui_dir=ui_dir,
            electron_binary=electron or Path(command[0]),
            message="Desktop pet runtime is ready.",
        )
    if electron is None:
        message = "Electron runtime is not installed for the desktop pet."
    else:
        message = "Desktop pet UI assets are unavailable."
    return PetSetupStatus(
        ready=False,
        ui_dir=ui_dir,
        electron_binary=electron,
        message=message,
    )


def prepare_pet_runtime(
    data_dir: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> PetSetupResult:
    """Install the desktop pet Electron runtime into Deepmate's local data dir."""
    root = Path(data_dir).expanduser()
    ui_dir = pet_runtime_ui_dir(root)
    try:
        _emit(progress, "Preparing desktop pet runtime directory...")
        ui_dir.mkdir(parents=True, exist_ok=True)
        _copy_ui_manifest(ui_dir)
        _copy_ui_assets(ui_dir)

        status = pet_setup_status(root)
        if status.ready:
            return PetSetupResult(
                ok=True,
                ui_dir=ui_dir,
                message="Desktop pet runtime is already ready.",
            )

        npm = shutil.which("npm")
        if npm is None:
            return PetSetupResult(
                ok=False,
                ui_dir=ui_dir,
                message=(
                    "npm is required to install the optional Electron runtime. "
                    "Install Node.js/npm, then run /pet setup again."
                ),
            )

        _emit(progress, "Installing Electron runtime for the desktop pet...")
        env = safe_environment()
        mirror = os.environ.get("DEEPMATE_PET_ELECTRON_MIRROR", "").strip()
        if mirror and not env.get("ELECTRON_MIRROR"):
            env["ELECTRON_MIRROR"] = mirror
        result = subprocess.run(
            [npm, "install", "--omit=optional"],
            cwd=str(ui_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=PET_SETUP_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode != 0:
            return PetSetupResult(
                ok=False,
                ui_dir=ui_dir,
                message=f"Electron runtime install failed with exit code {result.returncode}.",
                stdout=_tail(result.stdout),
                stderr=_tail(result.stderr),
            )
        status = pet_setup_status(root)
        if not status.ready:
            return PetSetupResult(
                ok=False,
                ui_dir=ui_dir,
                message="Electron install completed, but the runtime binary was not found.",
                stdout=_tail(result.stdout),
                stderr=_tail(result.stderr),
            )
        _emit(progress, "Desktop pet runtime is ready.")
        return PetSetupResult(
            ok=True,
            ui_dir=ui_dir,
            message="Desktop pet runtime is ready. You can now run /pet on.",
            stdout=_tail(result.stdout),
            stderr=_tail(result.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        return PetSetupResult(
            ok=False,
            ui_dir=ui_dir,
            message="Electron runtime install timed out.",
            stdout=_tail(exc.stdout if isinstance(exc.stdout, str) else ""),
            stderr=_tail(exc.stderr if isinstance(exc.stderr, str) else ""),
        )
    except OSError as exc:
        return PetSetupResult(
            ok=False,
            ui_dir=ui_dir,
            message=f"Could not prepare desktop pet runtime: {exc}",
        )


def _copy_ui_manifest(ui_dir: Path) -> None:
    source = _packaged_ui_dir()
    for filename in ("package.json", "package-lock.json"):
        src = source / filename
        if src.exists():
            shutil.copy2(src, ui_dir / filename)


def _copy_ui_assets(ui_dir: Path) -> None:
    source = _packaged_ui_dir()
    for dirname in ("electron", "renderer"):
        src_dir = source / dirname
        dst_dir = ui_dir / dirname
        if not src_dir.exists():
            continue
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)


def _packaged_ui_dir() -> Path:
    from deepmate.pet.electron_host import packaged_pet_ui_dir

    return packaged_pet_ui_dir()


def _electron_binary_for_ui(ui_dir: Path) -> Path | None:
    override = os.environ.get("DEEPMATE_PET_ELECTRON", "").strip()
    if override:
        binary = Path(override).expanduser()
        if binary.exists():
            return binary
    bin_name = "electron.cmd" if os.name == "nt" else "electron"
    binary = ui_dir / "node_modules" / ".bin" / bin_name
    return binary if binary.exists() else None


def _tail(value: str, *, max_chars: int = 4000) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
