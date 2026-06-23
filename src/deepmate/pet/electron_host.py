"""Electron frontend host for the Deepmate desktop pet."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from deepmate.pet.service import PetBackendService
from deepmate.pet.state import PetStateStore
from deepmate.providers import ModelProvider

SOURCE_PET_UI_DIR = Path(__file__).resolve().parents[3] / "pet_ui"
PACKAGED_PET_UI_DIR = Path(__file__).resolve().parents[1] / "pet_ui"
PET_RUNTIME_RELATIVE_DIR = Path("pet") / "ui_runtime"


def run_pet_host(
    data_dir: str | Path,
    *,
    provider: ModelProvider | None = None,
    model: str = "",
    poll_seconds: float = 1.2,
) -> int:
    """Run the Python pet service and the Electron pet frontend."""
    command = electron_pet_command(data_dir)
    if command is None:
        print(electron_pet_missing_message(), file=sys.stderr)
        return 2

    store = PetStateStore.in_data_dir(data_dir)
    service = PetBackendService(
        store,
        provider=provider,
        model=model,
        poll_seconds=poll_seconds,
    )
    service.tick()

    stderr_file = tempfile.TemporaryFile("w+", encoding="utf-8")
    try:
        try:
            pet_ui_dir = _pet_ui_dir(data_dir)
            process = subprocess.Popen(
                command,
                cwd=str(pet_ui_dir),
                env=_electron_env(),
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            print(f"error: cannot start desktop pet frontend: {exc}", file=sys.stderr)
            return 2

        interrupted = False
        try:
            while process.poll() is None:
                service.tick()
                service.stop_event.wait(max(0.2, float(poll_seconds)))
        except KeyboardInterrupt:
            interrupted = True
            _terminate_process(process)
        finally:
            service.stop()

        stderr = ""
        try:
            stderr_file.seek(0)
            stderr = stderr_file.read(900)
        except OSError:
            stderr = ""
        if interrupted:
            if stderr.strip():
                print(stderr.strip(), file=sys.stderr)
            return 130
        if process.returncode not in (0, None):
            print(
                f"error: desktop pet frontend exited with code {process.returncode}."
                + (f"\n{stderr.strip()}" if stderr.strip() else ""),
                file=sys.stderr,
            )
            return int(process.returncode or 1)
        return 0
    finally:
        stderr_file.close()


def electron_pet_command(data_dir: str | Path | None) -> list[str] | None:
    """Return the Electron command for the pet frontend, if installed."""
    if data_dir is None:
        return None
    resolved = _electron_and_ui_dir(data_dir)
    if resolved is None:
        return None
    electron, pet_ui_dir = resolved
    main_js = pet_ui_dir / "electron" / "main.js"
    if not main_js.exists():
        return None
    return [
        str(electron),
        str(main_js),
        "--data-dir",
        str(Path(data_dir)),
    ]


def electron_pet_missing_message() -> str:
    """Return a user-facing message for missing Electron pet dependencies."""
    return (
        "Desktop pet is optional and needs an Electron runtime before it can open.\n"
        "Normal Deepmate CLI/TUI work continues without it.\n"
        "Recommended: run `/pet setup` in the TUI, or set DEEPMATE_PET_ELECTRON "
        "to an existing Electron binary.\n"
        "From a source checkout you can also run `npm --prefix pet_ui install`, "
        "then retry `deepmate --pet` or `/pet on`."
    )


def source_pet_ui_dir() -> Path:
    """Return the source checkout pet UI directory."""
    return SOURCE_PET_UI_DIR


def packaged_pet_ui_dir() -> Path:
    """Return the packaged pet UI asset directory."""
    return PACKAGED_PET_UI_DIR


def pet_runtime_ui_dir(data_dir: str | Path | None) -> Path:
    """Return the writable local pet runtime directory."""
    root = Path(data_dir).expanduser() if data_dir is not None else Path.home()
    return root / PET_RUNTIME_RELATIVE_DIR


def _pet_ui_dir(data_dir: str | Path | None = None) -> Path:
    runtime = pet_runtime_ui_dir(data_dir)
    if _local_electron_binary(runtime) is not None and (runtime / "electron" / "main.js").exists():
        return runtime
    if (SOURCE_PET_UI_DIR / "electron" / "main.js").exists():
        return SOURCE_PET_UI_DIR
    if (runtime / "electron" / "main.js").exists():
        return runtime
    return PACKAGED_PET_UI_DIR


def _electron_binary(data_dir: str | Path | None = None) -> Path | None:
    resolved = _electron_and_ui_dir(data_dir)
    return resolved[0] if resolved is not None else None


def _electron_and_ui_dir(data_dir: str | Path | None = None) -> tuple[Path, Path] | None:
    override = os.environ.get("DEEPMATE_PET_ELECTRON", "").strip()
    if override:
        binary = Path(override).expanduser()
        if binary.exists():
            return binary, _pet_ui_dir(data_dir)
    for ui_dir in (
        SOURCE_PET_UI_DIR,
        pet_runtime_ui_dir(data_dir),
        PACKAGED_PET_UI_DIR,
    ):
        if not (ui_dir / "electron" / "main.js").exists():
            continue
        local = _local_electron_binary(ui_dir)
        if local is not None:
            return local, ui_dir
    return None


def _local_electron_binary(ui_dir: Path) -> Path | None:
    bin_name = "electron.cmd" if os.name == "nt" else "electron"
    local = ui_dir / "node_modules" / ".bin" / bin_name
    return local if local.exists() else None


def _electron_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("ELECTRON_DISABLE_SECURITY_WARNINGS", "true")
    return env


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=2)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()
