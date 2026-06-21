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

PET_UI_DIR = Path(__file__).resolve().parents[3] / "pet_ui"
PET_ELECTRON_MAIN = PET_UI_DIR / "electron" / "main.js"


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
            process = subprocess.Popen(
                command,
                cwd=str(PET_UI_DIR),
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
    electron = _electron_binary()
    if electron is None or not PET_ELECTRON_MAIN.exists():
        return None
    return [
        str(electron),
        str(PET_ELECTRON_MAIN),
        "--data-dir",
        str(Path(data_dir)),
    ]


def electron_pet_missing_message() -> str:
    """Return a user-facing message for missing Electron pet dependencies."""
    return (
        "Desktop pet is optional, and its Electron frontend is not installed.\n"
        "Normal Deepmate CLI/TUI runs continue without it.\n"
        "To enable the pet from a source checkout, run:\n"
        "  npm --prefix pet_ui install\n"
        "This creates the ignored local directory pet_ui/node_modules/.\n"
        "Or set DEEPMATE_PET_ELECTRON to an existing Electron binary.\n"
        "Then retry `deepmate --pet` or `/pet on`."
    )


def _electron_binary() -> Path | None:
    override = os.environ.get("DEEPMATE_PET_ELECTRON", "").strip()
    if override:
        binary = Path(override).expanduser()
        if binary.exists():
            return binary
    bin_name = "electron.cmd" if os.name == "nt" else "electron"
    local = PET_UI_DIR / "node_modules" / ".bin" / bin_name
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
