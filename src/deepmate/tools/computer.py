"""Computer Use helper tools.

These tools are intentionally small.  Browser automation remains owned by
tools/browser.py; this module adds macOS desktop observation and action
primitives for explicit Computer Use sessions.
"""

from __future__ import annotations

import platform
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from deepmate.runtime.process_env import subprocess_environment
from deepmate.tools.registry import NativeTool, NativeToolResult

COMPUTER_STATUS_TOOL_NAME = "computer_status"
COMPUTER_SNAPSHOT_TOOL_NAME = "computer_snapshot"
COMPUTER_SCREENSHOT_TOOL_NAME = "computer_screenshot"
COMPUTER_CLICK_TOOL_NAME = "computer_click"
COMPUTER_TYPE_TOOL_NAME = "computer_type"
COMPUTER_KEY_TOOL_NAME = "computer_key"
COMPUTER_OPEN_TOOL_NAME = "computer_open"
COMPUTER_WAIT_TOOL_NAME = "computer_wait"

COMPUTER_TOOL_NAMES = (
    COMPUTER_STATUS_TOOL_NAME,
    COMPUTER_SNAPSHOT_TOOL_NAME,
    COMPUTER_SCREENSHOT_TOOL_NAME,
    COMPUTER_CLICK_TOOL_NAME,
    COMPUTER_TYPE_TOOL_NAME,
    COMPUTER_KEY_TOOL_NAME,
    COMPUTER_OPEN_TOOL_NAME,
    COMPUTER_WAIT_TOOL_NAME,
)

DEFAULT_COMPUTER_TIMEOUT_SECONDS = 30
MAX_TEXT_CHARS = 8_000
OBSERVATION_TTL_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class ComputerUseState:
    """Minimal state protocol shared with BehaviorRuntime."""

    enabled: Callable[[], bool]
    computer_learning_enabled: Callable[[], bool]


@dataclass(slots=True)
class _ComputerObservation:
    observed_at: float = 0.0
    readable_observed_at: float = 0.0
    screenshot_observed_at: float = 0.0
    kind: str = ""
    pixel_width: int = 0
    pixel_height: int = 0
    logical_width: int = 0
    logical_height: int = 0
    scale: float = 1.0
    path: str = ""

    def is_fresh(self) -> bool:
        return self.observed_at > 0 and (
            time.monotonic() - self.observed_at
        ) <= OBSERVATION_TTL_SECONDS

    def has_fresh_readable_state(self) -> bool:
        return self.readable_observed_at > 0 and (
            time.monotonic() - self.readable_observed_at
        ) <= OBSERVATION_TTL_SECONDS

    def has_fresh_screenshot_geometry(self) -> bool:
        return self.screenshot_observed_at > 0 and (
            time.monotonic() - self.screenshot_observed_at
        ) <= OBSERVATION_TTL_SECONDS


def computer_tools(
    *,
    data_dir: str | Path,
    workspace: str | Path,
    session_id: str,
    state: ComputerUseState,
    exposed_by_default: bool = False,
) -> tuple[NativeTool, ...]:
    root = Path(data_dir)
    workspace_path = Path(workspace).resolve()
    observation = _ComputerObservation()
    return (
        NativeTool(
            name=COMPUTER_STATUS_TOOL_NAME,
            description=(
                "Show the current explicit Computer Use session state and privacy mode."
            ),
            input_schema=_empty_schema(),
            handler=lambda _arguments: _computer_status(
                session_id=session_id,
                workspace=workspace_path,
                state=state,
            ),
            exposed_by_default=exposed_by_default,
        ),
        NativeTool(
            name=COMPUTER_SCREENSHOT_TOOL_NAME,
            description=(
                "Capture a screenshot of the current macOS desktop for an explicitly "
                "enabled Computer Use task. The image is saved under Deepmate's "
                "private data dir and attached for vision-capable model requests."
            ),
            input_schema=_computer_screenshot_schema(),
            handler=lambda arguments: _computer_screenshot(
                data_dir=root,
                session_id=session_id,
                state=state,
                observation=observation,
                arguments=arguments,
            ),
            exposed_by_default=exposed_by_default,
        ),
        NativeTool(
            name=COMPUTER_SNAPSHOT_TOOL_NAME,
            description=(
                "Read a bounded macOS Accessibility snapshot of the frontmost "
                "application, including visible UI element names, roles, values, "
                "and coordinates. Use this before desktop clicks or typing when "
                "a browser DOM snapshot is not available."
            ),
            input_schema=_computer_snapshot_schema(),
            handler=lambda arguments: _computer_snapshot(
                state=state,
                observation=observation,
                arguments=arguments,
            ),
            exposed_by_default=exposed_by_default,
        ),
        NativeTool(
            name=COMPUTER_CLICK_TOOL_NAME,
            description=(
                "Click a visible point on the macOS desktop by screen coordinates "
                "during an explicitly enabled Computer Use task. Take a screenshot "
                "first when coordinates are uncertain."
            ),
            input_schema=_computer_click_schema(),
            handler=lambda arguments: _computer_click(
                state=state,
                observation=observation,
                arguments=arguments,
            ),
            read_only=False,
            exposed_by_default=exposed_by_default,
        ),
        NativeTool(
            name=COMPUTER_TYPE_TOOL_NAME,
            description=(
                "Type or paste text into the currently focused macOS UI control "
                "during an explicitly enabled Computer Use task. This uses a "
                "clipboard-preserving paste by default so non-ASCII text works."
            ),
            input_schema=_computer_type_schema(),
            handler=lambda arguments: _computer_type(
                state=state,
                observation=observation,
                arguments=arguments,
            ),
            read_only=False,
            exposed_by_default=exposed_by_default,
        ),
        NativeTool(
            name=COMPUTER_KEY_TOOL_NAME,
            description=(
                "Press a key or keyboard shortcut in the currently focused macOS "
                "application during an explicitly enabled Computer Use task."
            ),
            input_schema=_computer_key_schema(),
            handler=lambda arguments: _computer_key(
                state=state,
                observation=observation,
                arguments=arguments,
            ),
            read_only=False,
            exposed_by_default=exposed_by_default,
        ),
        NativeTool(
            name=COMPUTER_OPEN_TOOL_NAME,
            description=(
                "Open a URL, application, or local path on macOS for the current "
                "Computer Use task. Ask the user before opening unrelated private "
                "files or making irreversible external changes."
            ),
            input_schema=_computer_open_schema(),
            handler=lambda arguments: _computer_open(
                workspace=workspace_path,
                state=state,
                observation=observation,
                arguments=arguments,
            ),
            read_only=False,
            exposed_by_default=exposed_by_default,
        ),
        NativeTool(
            name=COMPUTER_WAIT_TOOL_NAME,
            description=(
                "Wait briefly for desktop UI changes during an explicitly enabled "
                "Computer Use task."
            ),
            input_schema=_computer_wait_schema(),
            handler=lambda arguments: _computer_wait(state=state, arguments=arguments),
            exposed_by_default=exposed_by_default,
        ),
    )


def _computer_status(
    *,
    session_id: str,
    workspace: Path,
    state: ComputerUseState,
) -> NativeToolResult:
    enabled = state.enabled()
    learning = state.computer_learning_enabled()
    bounds = _screen_bounds()
    scale = _display_scale_factor(bounds)
    content = "\n".join(
        (
            "Computer Use status",
            f"- enabled: {str(enabled).lower()}",
            f"- long_term_learning_from_computer_use: {str(learning).lower()}",
            f"- session: {session_id}",
            f"- workspace: {workspace}",
            "- browser automation: use browser tools when loaded",
            "- desktop snapshot: macOS Accessibility snapshot when permission is granted",
            "- desktop screenshot: macOS screencapture when Screen Recording permission is granted",
            "- desktop actions: macOS System Events/open when Accessibility permission is granted",
            (
                f"- screen: {bounds.get('width', 0)}x{bounds.get('height', 0)} logical, scale {scale:g}"
                if bounds
                else "- screen: unavailable"
            ),
            "- privacy: credentials, cookies, password stores, CAPTCHA, and unrelated private messages are out of scope",
        )
    )
    return NativeToolResult(
        content=content,
        data={
            "enabled": enabled,
            "computer_learning": learning,
            "session_id": session_id,
            "workspace": str(workspace),
            "desktop_screenshot_supported": platform.system() == "Darwin",
            "desktop_snapshot_supported": platform.system() == "Darwin",
            "desktop_actions_supported": platform.system() == "Darwin",
            "screen_bounds": bounds,
            "scale": scale,
            "coordinate_space": "logical",
        },
        refs=(
            "computer_use_enabled=" + str(enabled).lower(),
            "computer_learning_enabled=" + str(learning).lower(),
            f"computer_session={session_id}",
        ),
    )


def _computer_screenshot(
    *,
    data_dir: Path,
    session_id: str,
    state: ComputerUseState,
    observation: _ComputerObservation,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    _require_enabled(state)
    _require_macos("Desktop screenshots")
    output_path = _screenshot_path(data_dir, session_id, arguments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ("screencapture", "-x", str(output_path))
    _run_process(command, cwd=data_dir, timeout_seconds=_timeout_seconds(arguments))
    info = _image_info(output_path)
    bounds = _screen_bounds()
    logical_width = _int_value(bounds.get("width"), 0)
    logical_height = _int_value(bounds.get("height"), 0)
    pixel_width = _int_value(info.get("width"), 0)
    pixel_height = _int_value(info.get("height"), 0)
    scale = _scale_factor(pixel_width, pixel_height, logical_width, logical_height)
    _record_observation(
        observation,
        kind="screenshot",
        readable=False,
        screenshot_geometry=True,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        logical_width=logical_width,
        logical_height=logical_height,
        scale=scale,
        path=str(output_path),
    )
    return NativeToolResult(
        content="\n".join(
            part
            for part in (
                "Desktop screenshot saved.",
                f"- path: {output_path}",
                f"- bytes: {info['bytes']}",
                (
                    f"- dimensions: {info['width']}x{info['height']}"
                    if info["width"] and info["height"]
                    else ""
                ),
                (
                    f"- logical_screen: {logical_width}x{logical_height}, scale: {scale:g}"
                    if logical_width and logical_height
                    else ""
                ),
                "- observation: screenshot geometry recorded for coordinate conversion.",
                "- note: vision-capable model requests receive this screenshot as an image attachment; text-only models receive path and dimensions only.",
                "- note: this screenshot is stored under Deepmate's private data dir, not the project source.",
            )
            if part
        ),
        data={
            "path": str(output_path),
            "bytes": info["bytes"],
            "image_format": info["image_format"],
            "width": info["width"],
            "height": info["height"],
            "logical_width": logical_width,
            "logical_height": logical_height,
            "scale": scale,
            "coordinate_space": "screenshot_pixel",
            "session_id": session_id,
            "computer_learning": state.computer_learning_enabled(),
        },
        refs=(
            f"computer_screenshot={output_path}",
            f"computer_session={session_id}",
            f"computer_image_attachment={output_path}",
        ),
        attachments=(
            {
                "type": "image",
                "path": str(output_path),
                "mime_type": "image/png",
                "width": info["width"],
                "height": info["height"],
                "source": "computer_screenshot",
            },
        ),
    )


def _computer_snapshot(
    *,
    state: ComputerUseState,
    observation: _ComputerObservation,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    _require_enabled(state)
    _require_macos("Desktop accessibility snapshot")
    max_depth = _int_argument(arguments, "max_depth", 1, 1, 6)
    max_items = _int_argument(arguments, "max_items", 10, 10, 200)
    completed = _run_osascript(
        _ACCESSIBILITY_SNAPSHOT_SCRIPT,
        str(max_depth),
        str(max_items),
        timeout_seconds=_timeout_seconds(arguments),
    )
    content = completed.stdout.strip()
    if not content:
        content = "Computer accessibility snapshot returned no visible UI elements."
    protected_reason = _protected_snapshot_reason(content)
    if protected_reason:
        return NativeToolResult(
            content=(
                "Computer accessibility snapshot is protected. "
                f"{protected_reason} Ask the user before inspecting this app."
            ),
            data={
                "max_depth": max_depth,
                "max_items": max_items,
                "protected": True,
                "reason": protected_reason,
            },
            refs=(
                "computer_snapshot=protected",
                "risk=local_private_access",
            ),
        )
    content = _redact_snapshot_text(content)
    bounds = _screen_bounds()
    _record_observation(
        observation,
        kind="snapshot",
        readable=True,
        logical_width=_int_value(bounds.get("width"), 0),
        logical_height=_int_value(bounds.get("height"), 0),
    )
    return NativeToolResult(
        content=content,
        data={
            "max_depth": max_depth,
            "max_items": max_items,
            "screen_bounds": bounds,
            "coordinate_space": "logical",
        },
        refs=(
            "computer_snapshot=frontmost_app",
            f"snapshot_max_depth={max_depth}",
            f"snapshot_max_items={max_items}",
        ),
    )


def _computer_click(
    *,
    state: ComputerUseState,
    observation: _ComputerObservation,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    _require_enabled(state)
    _require_macos("Desktop click")
    _require_recent_readable_observation(observation)
    raw_x = _int_argument(arguments, "x", 0, -100_000, 100_000, required=True)
    raw_y = _int_argument(arguments, "y", 0, -100_000, 100_000, required=True)
    coordinate_space = _text_argument(arguments, "coordinate_space") or "logical"
    x, y = _logical_click_point(raw_x, raw_y, coordinate_space, observation)
    _require_point_in_bounds(x, y)
    count = _int_argument(arguments, "count", 1, 1, 3)
    script = """
on run argv
  set px to item 1 of argv as integer
  set py to item 2 of argv as integer
  set clickCount to item 3 of argv as integer
  tell application "System Events"
    repeat clickCount times
      click at {px, py}
      delay 0.05
    end repeat
  end tell
end run
""".strip()
    _run_osascript(
        script,
        str(x),
        str(y),
        str(count),
        timeout_seconds=_timeout_seconds(arguments),
    )
    return NativeToolResult(
        content=(
            f"Clicked desktop point ({x}, {y}) {count} time(s). "
            "Observe the screen again before the next desktop action."
        ),
        data={
            "x": x,
            "y": y,
            "input_x": raw_x,
            "input_y": raw_y,
            "coordinate_space": coordinate_space,
            "count": count,
            "observation_kind": observation.kind,
        },
        refs=(
            f"computer_click={x},{y}",
            f"click_count={count}",
            f"coordinate_space={coordinate_space}",
        ),
    )


def _computer_type(
    *,
    state: ComputerUseState,
    observation: _ComputerObservation,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    _require_enabled(state)
    _require_macos("Desktop typing")
    _require_recent_readable_observation(observation)
    text = _text_argument(arguments, "text")
    if not text:
        raise ValueError("computer_type requires non-empty text.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"computer_type text is limited to {MAX_TEXT_CHARS} characters.")
    method = _text_argument(arguments, "method") or "paste"
    if method not in {"paste", "keystroke"}:
        raise ValueError("computer_type method must be paste or keystroke.")
    if method == "keystroke":
        script = """
on run argv
  tell application "System Events" to keystroke (item 1 of argv)
end run
""".strip()
    else:
        script = """
on run argv
  set textToType to item 1 of argv
  set oldClipboard to the clipboard
  set the clipboard to textToType
  delay 0.1
  tell application "System Events" to keystroke "v" using {command down}
  delay 0.5
  set the clipboard to oldClipboard
end run
""".strip()
    _run_osascript(script, text, timeout_seconds=_timeout_seconds(arguments))
    return NativeToolResult(
        content=f"Typed text into the focused UI control ({len(text)} chars).",
        data={"chars": len(text), "method": method},
        refs=(f"computer_type_chars={len(text)}", f"computer_type_method={method}"),
    )


def _computer_key(
    *,
    state: ComputerUseState,
    observation: _ComputerObservation,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    _require_enabled(state)
    _require_macos("Desktop keyboard")
    _require_recent_readable_observation(observation)
    key = _text_argument(arguments, "key").lower()
    if not key:
        raise ValueError("computer_key requires a key.")
    modifiers = _modifiers(arguments.get("modifiers"))
    using = _applescript_modifiers(modifiers)
    code = _KEY_CODES.get(key)
    if code is not None:
        script = (
            f'tell application "System Events" to key code {code}'
            + (f" using {using}" if using else "")
        )
        args: tuple[str, ...] = ()
    else:
        key_text = _key_text(key)
        script = (
            'on run argv\n'
            '  tell application "System Events" to keystroke (item 1 of argv)'
            + (f" using {using}" if using else "")
            + "\nend run"
        )
        args = (key_text,)
    _run_osascript(script, *args, timeout_seconds=_timeout_seconds(arguments))
    combo = "+".join((*modifiers, key)) if modifiers else key
    return NativeToolResult(
        content=f"Pressed key: {combo}.",
        data={"key": key, "modifiers": list(modifiers)},
        refs=(f"computer_key={combo}",),
    )


def _computer_open(
    *,
    workspace: Path,
    state: ComputerUseState,
    observation: _ComputerObservation,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    _require_enabled(state)
    _require_macos("Desktop open")
    target = _text_argument(arguments, "target")
    if not target:
        raise ValueError("computer_open requires a target.")
    kind = _text_argument(arguments, "kind") or "auto"
    if kind not in {"auto", "url", "app", "path"}:
        raise ValueError("computer_open kind must be auto, url, app, or path.")
    resolved_kind = _open_kind(target, kind)
    if resolved_kind != "path" or _path_is_private(target, workspace):
        _require_recent_readable_observation(observation)
    if resolved_kind == "url":
        if not target.startswith(("http://", "https://")):
            raise ValueError("computer_open URL targets must start with http:// or https://.")
        command = ("open", target)
    elif resolved_kind == "app":
        command = ("open", "-a", target)
    else:
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = workspace / path
        command = ("open", str(path))
    _run_process(command, cwd=Path.home(), timeout_seconds=_timeout_seconds(arguments))
    return NativeToolResult(
        content=f"Opened {resolved_kind}: {target}",
        data={"target": target, "kind": resolved_kind},
        refs=(f"computer_open_kind={resolved_kind}",),
    )


def _computer_wait(
    *,
    state: ComputerUseState,
    arguments: Mapping[str, object],
) -> NativeToolResult:
    _require_enabled(state)
    seconds = _float_argument(arguments, "seconds", 1.0, 0.1, 20.0)
    time.sleep(seconds)
    return NativeToolResult(
        content=f"Waited {seconds:g} seconds.",
        data={"seconds": seconds},
        refs=(f"computer_wait_seconds={seconds:g}",),
    )


def _screenshot_path(
    data_dir: Path,
    session_id: str,
    arguments: Mapping[str, object],
) -> Path:
    raw_name = str(arguments.get("name", "")).strip()
    if raw_name:
        safe_name = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in raw_name)
        filename = safe_name if safe_name.endswith(".png") else f"{safe_name}.png"
    else:
        stamp = datetime.now().astimezone().strftime("%H%M%S")
        filename = f"{_safe_id(session_id)}-{stamp}.png"
    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    return data_dir / "computer" / "screenshots" / "tmp" / date / filename


def _require_enabled(state: ComputerUseState) -> None:
    if not state.enabled():
        raise ValueError(
            "Computer Use is not enabled for this session. Use /computer on or "
            "start Deepmate with --computer-use for a current-task-only session."
        )


def _require_recent_observation(observation: _ComputerObservation) -> None:
    if observation.is_fresh():
        return
    raise ValueError(
        "Computer action requires a recent screen observation. Run "
        "computer_snapshot or computer_screenshot first, then act from that result."
    )


def _require_recent_readable_observation(observation: _ComputerObservation) -> None:
    if observation.has_fresh_readable_state():
        return
    raise ValueError(
        "Computer action requires a recent readable screen observation. Run "
        "computer_snapshot first. computer_screenshot can provide visual input "
        "only for vision-capable model requests and is used here for coordinate conversion."
    )


def _record_observation(
    observation: _ComputerObservation,
    *,
    kind: str,
    readable: bool = False,
    screenshot_geometry: bool = False,
    pixel_width: int = 0,
    pixel_height: int = 0,
    logical_width: int = 0,
    logical_height: int = 0,
    scale: float = 1.0,
    path: str = "",
) -> None:
    now = time.monotonic()
    observation.observed_at = now
    if readable:
        observation.readable_observed_at = now
    if screenshot_geometry:
        observation.screenshot_observed_at = now
    observation.kind = kind
    if screenshot_geometry or pixel_width > 0:
        observation.pixel_width = pixel_width
    if screenshot_geometry or pixel_height > 0:
        observation.pixel_height = pixel_height
    if logical_width > 0:
        observation.logical_width = logical_width
    if logical_height > 0:
        observation.logical_height = logical_height
    if screenshot_geometry or scale != 1.0:
        observation.scale = scale if scale > 0 else 1.0
    if path:
        observation.path = path


def _logical_click_point(
    x: int,
    y: int,
    coordinate_space: str,
    observation: _ComputerObservation,
) -> tuple[int, int]:
    if coordinate_space == "logical":
        return x, y
    if coordinate_space != "screenshot_pixel":
        raise ValueError("coordinate_space must be logical or screenshot_pixel.")
    if not observation.has_fresh_screenshot_geometry():
        raise ValueError(
            "screenshot_pixel coordinates require a recent computer_screenshot."
        )
    if observation.pixel_width <= 0 or observation.pixel_height <= 0:
        raise ValueError(
            "screenshot_pixel coordinates require a recent computer_screenshot."
        )
    scale_x = _axis_scale(observation.pixel_width, observation.logical_width, observation.scale)
    scale_y = _axis_scale(observation.pixel_height, observation.logical_height, observation.scale)
    return round(x / scale_x), round(y / scale_y)


def _axis_scale(pixel_size: int, logical_size: int, fallback: float) -> float:
    if pixel_size > 0 and logical_size > 0:
        return max(pixel_size / logical_size, 0.1)
    return max(fallback, 0.1)


def _require_point_in_bounds(x: int, y: int) -> None:
    bounds = _screen_bounds()
    if not bounds:
        return
    left = _int_value(bounds.get("left"), 0)
    top = _int_value(bounds.get("top"), 0)
    right = _int_value(bounds.get("right"), 0)
    bottom = _int_value(bounds.get("bottom"), 0)
    if x < left or x > right or y < top or y > bottom:
        raise ValueError(
            f"click point ({x}, {y}) is outside the current screen bounds "
            f"({left}, {top})-({right}, {bottom})."
        )


def _require_macos(capability: str) -> None:
    if platform.system() != "Darwin":
        raise ValueError(f"{capability} is currently supported on macOS only.")


def _run_osascript(
    script: str,
    *args: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return _run_process(
        ("osascript", "-e", script, *args),
        cwd=Path.home(),
        timeout_seconds=timeout_seconds,
        permission_hint=(
            "macOS blocked the action. Grant Accessibility permission to the "
            "terminal app running Deepmate, then try again."
        ),
    )


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    permission_hint: str = "",
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        tuple(command),
        cwd=str(cwd),
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "command failed").strip()
        if permission_hint:
            message = f"{message}\n{permission_hint}" if message else permission_hint
        raise ValueError(message)
    return completed


def _timeout_seconds(arguments: Mapping[str, object]) -> int:
    value = arguments.get("timeout_seconds", DEFAULT_COMPUTER_TIMEOUT_SECONDS)
    if isinstance(value, bool):
        return DEFAULT_COMPUTER_TIMEOUT_SECONDS
    if isinstance(value, int):
        return min(max(value, 5), 120)
    if isinstance(value, str) and value.strip().isdigit():
        return min(max(int(value.strip()), 5), 120)
    return DEFAULT_COMPUTER_TIMEOUT_SECONDS


def _int_argument(
    arguments: Mapping[str, object],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
    *,
    required: bool = False,
) -> int:
    value = arguments.get(name)
    if value is None:
        if required:
            raise ValueError(f"{name} is required.")
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(f"{name} must be an integer.")
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _float_argument(
    arguments: Mapping[str, object],
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = arguments.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a number.") from exc
    else:
        raise ValueError(f"{name} must be a number.")
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
    return parsed


def _text_argument(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name, "")
    return value.strip() if isinstance(value, str) else ""


def _modifiers(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.split("+")]
    elif isinstance(value, (list, tuple)):
        raw = [str(part).strip() for part in value]
    else:
        raw = []
    aliases = {
        "cmd": "command",
        "command": "command",
        "ctrl": "control",
        "control": "control",
        "option": "option",
        "alt": "option",
        "shift": "shift",
    }
    clean: list[str] = []
    for item in raw:
        normalized = aliases.get(item.lower())
        if normalized and normalized not in clean:
            clean.append(normalized)
    return tuple(clean)


def _applescript_modifiers(modifiers: Sequence[str]) -> str:
    if not modifiers:
        return ""
    return "{" + ", ".join(f"{modifier} down" for modifier in modifiers) + "}"


def _key_text(key: str) -> str:
    aliases = {"space": " ", "plus": "+", "minus": "-", "comma": ",", "period": "."}
    if key in aliases:
        return aliases[key]
    if len(key) == 1:
        return key
    raise ValueError(f"Unsupported computer key: {key}")


_KEY_CODES = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "escape": 53,
    "esc": 53,
    "backspace": 51,
    "delete": 51,
    "forward_delete": 117,
    "home": 115,
    "end": 119,
    "page_up": 116,
    "page_down": 121,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f8": 100,
    "f9": 101,
    "f10": 109,
    "f11": 103,
    "f12": 111,
    "capslock": 57,
    "caps_lock": 57,
    "numlock": 71,
    "num_lock": 71,
}


def _open_kind(target: str, kind: str) -> str:
    if kind != "auto":
        return kind
    if target.startswith(("http://", "https://")):
        return "url"
    if "/" in target or target.startswith(("~", ".")):
        return "path"
    return "app"


def _path_is_private(target: str, workspace: Path) -> bool:
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = workspace / path
    try:
        path.resolve().relative_to(workspace.resolve())
        return False
    except ValueError:
        return True


_PROTECTED_FRONT_APPS = (
    "1password",
    "bitwarden",
    "dashlane",
    "keychain access",
    "lastpass",
    "mail",
    "messages",
    "passwords",
    "system settings",
    "wechat",
    "wecom",
)

_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)\b(password|passcode|passwd|secret|token|api[_ -]?key|access[_ -]?key|"
    r"验证码|verification code|captcha|card number|credit card|ssn|otp)\b"
)
_QUOTED_VALUE_RE = re.compile(r'(value|name|description)="([^"]*)"')


def _protected_snapshot_reason(content: str) -> str:
    front_app = _snapshot_field(content, "front_app").lower()
    bundle_id = _snapshot_field(content, "bundle_id").lower()
    combined = f"{front_app} {bundle_id}".strip()
    if not combined:
        return ""
    for app in _PROTECTED_FRONT_APPS:
        if app in combined:
            return f"The frontmost app appears sensitive ({front_app or bundle_id})."
    return ""


def _snapshot_field(content: str, name: str) -> str:
    prefix = f"- {name}:"
    for line in content.splitlines():
        if line.strip().lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _redact_snapshot_text(content: str) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        if _SENSITIVE_VALUE_RE.search(line):
            lines.append(_redact_snapshot_line(line))
        else:
            lines.append(line)
    return "\n".join(lines)


def _redact_snapshot_line(line: str) -> str:
    return _QUOTED_VALUE_RE.sub(lambda match: f'{match.group(1)}="[redacted]"', line)


def _screen_bounds() -> Mapping[str, int]:
    if platform.system() != "Darwin":
        return {}
    script = """
use framework "AppKit"
set minX to 0
set minY to 0
set maxX to 0
set maxY to 0
set didSet to false
repeat with screen in current application's NSScreen's screens()
  set frame to screen's frame()
  set origin to frame's origin()
  set size to frame's size()
  set leftEdge to origin's x as integer
  set bottomEdge to origin's y as integer
  set rightEdge to (origin's x + size's width) as integer
  set topEdge to (origin's y + size's height) as integer
  if didSet is false then
    set minX to leftEdge
    set minY to bottomEdge
    set maxX to rightEdge
    set maxY to topEdge
    set didSet to true
  else
    if leftEdge < minX then set minX to leftEdge
    if bottomEdge < minY then set minY to bottomEdge
    if rightEdge > maxX then set maxX to rightEdge
    if topEdge > maxY then set maxY to topEdge
  end if
end repeat
return (minX as text) & "," & (minY as text) & "," & (maxX as text) & "," & (maxY as text)
""".strip()
    try:
        completed = _run_process(
            ("osascript", "-e", script),
            cwd=Path.home(),
            timeout_seconds=5,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return _finder_desktop_bounds()
    return _bounds_from_text(completed.stdout)


def _finder_desktop_bounds() -> Mapping[str, int]:
    script = 'tell application "Finder" to get bounds of window of desktop'
    try:
        completed = _run_process(
            ("osascript", "-e", script),
            cwd=Path.home(),
            timeout_seconds=5,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    return _bounds_from_text(completed.stdout)


def _bounds_from_text(value: str) -> Mapping[str, int]:
    values: list[int] = []
    for part in value.replace(",", " ").split():
        try:
            values.append(int(part))
        except ValueError:
            continue
    if len(values) < 4:
        return {}
    left, top, right, bottom = values[:4]
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": max(0, right - left),
        "height": max(0, bottom - top),
    }


def _display_scale_factor(bounds: Mapping[str, object]) -> float:
    if platform.system() != "Darwin":
        return 1.0
    script = """
use framework "AppKit"
set maxScale to 1.0
repeat with screen in current application's NSScreen's screens()
  set currentScale to screen's backingScaleFactor()
  if currentScale > maxScale then set maxScale to currentScale
end repeat
return maxScale as text
""".strip()
    try:
        completed = _run_process(
            ("osascript", "-e", script),
            cwd=Path.home(),
            timeout_seconds=5,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 1.0
    try:
        return max(1.0, float(completed.stdout.strip()))
    except ValueError:
        return 1.0


def _scale_factor(
    pixel_width: int,
    pixel_height: int,
    logical_width: int,
    logical_height: int,
) -> float:
    scale_x = pixel_width / logical_width if pixel_width and logical_width else 0.0
    scale_y = pixel_height / logical_height if pixel_height and logical_height else 0.0
    candidates = [value for value in (scale_x, scale_y) if value > 0]
    return max(candidates) if candidates else 1.0


def _int_value(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _image_info(path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "bytes": 0,
        "image_format": "",
        "width": 0,
        "height": 0,
    }
    try:
        info["bytes"] = path.stat().st_size
        data = path.read_bytes()[:1_048_576]
    except OSError:
        return info
    image_format, width, height = _image_metadata(data)
    info["image_format"] = image_format
    info["width"] = width
    info["height"] = height
    return info


def _image_metadata(data: bytes) -> tuple[str, int, int]:
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ("png", int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))
    if len(data) >= 10 and data.startswith(b"GIF"):
        return ("gif", int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little"))
    jpeg_size = _jpeg_size(data)
    if jpeg_size is not None:
        return ("jpeg", jpeg_size[0], jpeg_size[1])
    return ("", 0, 0)


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            return None
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            return None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length >= 7:
                height = int.from_bytes(data[index + 3 : index + 5], "big")
                width = int.from_bytes(data[index + 5 : index + 7], "big")
                return (width, height)
        index += segment_length
    return None


_ACCESSIBILITY_SNAPSHOT_SCRIPT = r'''
on joinList(theList, delimiter)
  set oldDelims to AppleScript's text item delimiters
  set AppleScript's text item delimiters to delimiter
  set joinedText to theList as text
  set AppleScript's text item delimiters to oldDelims
  return joinedText
end joinList

on cleanText(rawText)
  set valueText to rawText as text
  set valueText to my joinList(paragraphs of valueText, " ")
  if (length of valueText) > 120 then
    return (text 1 thru 117 of valueText) & "..."
  end if
  return valueText
end cleanText

on describeElement(theElement)
  set parts to {}
  try
    set roleText to role of theElement as text
    if roleText is not "" then set end of parts to "role=" & my cleanText(roleText)
  end try
  try
    set subroleText to subrole of theElement as text
    if subroleText is not "" then set end of parts to "subrole=" & my cleanText(subroleText)
  end try
  try
    set nameText to name of theElement as text
    if nameText is not "" then set end of parts to "name=\"" & my cleanText(nameText) & "\""
  end try
  try
    set descText to description of theElement as text
    if descText is not "" then set end of parts to "description=\"" & my cleanText(descText) & "\""
  end try
  try
    set valueText to value of theElement as text
    if valueText is not "" then set end of parts to "value=\"" & my cleanText(valueText) & "\""
  end try
  try
    set p to position of theElement
    set s to size of theElement
    set end of parts to "bounds=" & (item 1 of p as integer) & "," & (item 2 of p as integer) & "," & (item 1 of s as integer) & "," & (item 2 of s as integer)
  end try
  if (count of parts) is 0 then return "role=unknown"
  return my joinList(parts, " ")
end describeElement

on dumpElement(theElement, depth, maxDepth, maxItems, counter)
  if counter >= maxItems then return {"", counter}
  set counter to counter + 1
  set indentText to ""
  repeat depth times
    set indentText to indentText & "  "
  end repeat
  set outputText to indentText & "- " & my describeElement(theElement) & linefeed
  if depth < maxDepth then
    try
      set children to UI elements of theElement
      repeat with childElement in children
        if counter >= maxItems then exit repeat
        set childResult to my dumpElement(childElement, depth + 1, maxDepth, maxItems, counter)
        set outputText to outputText & item 1 of childResult
        set counter to item 2 of childResult
      end repeat
    end try
  end if
  return {outputText, counter}
end dumpElement

on run argv
  set maxDepth to item 1 of argv as integer
  set maxItems to item 2 of argv as integer
  tell application "System Events"
    set frontApps to every application process whose frontmost is true
    if (count of frontApps) is 0 then return "Computer accessibility snapshot" & linefeed & "- front_app: (none)"
    set frontApp to item 1 of frontApps
    set outputText to "Computer accessibility snapshot" & linefeed & "- front_app: " & (name of frontApp as text) & linefeed
    try
      set outputText to outputText & "- bundle_id: " & (bundle identifier of frontApp as text) & linefeed
    end try
    set outputText to outputText & "Windows:" & linefeed
    set counter to 0
    try
      set frontWindows to windows of frontApp
      repeat with windowElement in frontWindows
        if counter >= maxItems then exit repeat
        set windowResult to my dumpElement(windowElement, 0, maxDepth, maxItems, counter)
        set outputText to outputText & item 1 of windowResult
        set counter to item 2 of windowResult
      end repeat
    on error errText
      set outputText to outputText & "- accessibility_error: " & errText & linefeed
    end try
    if counter >= maxItems then set outputText to outputText & "- truncated: max_items reached" & linefeed
    return outputText
  end tell
end run
'''.strip()


def _safe_id(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)
    return clean or "session"


def _empty_schema() -> Mapping[str, object]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _computer_screenshot_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Optional short file name for the screenshot PNG.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Screenshot timeout in seconds, between 5 and 120.",
            },
        },
        "additionalProperties": False,
    }


def _computer_snapshot_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "max_depth": {
                "type": "integer",
                "description": "Maximum UI tree depth, between 1 and 6. Defaults to 1.",
            },
            "max_items": {
                "type": "integer",
                "description": "Maximum UI elements to return, between 10 and 200. Defaults to 10.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Snapshot timeout in seconds, between 5 and 120.",
            },
        },
        "additionalProperties": False,
    }


def _computer_click_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Absolute screen x coordinate."},
            "y": {"type": "integer", "description": "Absolute screen y coordinate."},
            "count": {
                "type": "integer",
                "description": "Number of left clicks, 1 to 3. Defaults to 1.",
            },
            "coordinate_space": {
                "type": "string",
                "enum": ["logical", "screenshot_pixel"],
                "description": (
                    "Coordinate space for x/y. Use screenshot_pixel when points "
                    "come from the last computer_screenshot; defaults to logical."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Action timeout in seconds, between 5 and 120.",
            },
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    }


def _computer_type_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to enter into the currently focused UI control.",
            },
            "method": {
                "type": "string",
                "enum": ["paste", "keystroke"],
                "description": "Input method. Defaults to clipboard-preserving paste.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Action timeout in seconds, between 5 and 120.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }


def _computer_key_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Key name such as enter, escape, tab, left, right, up, down, "
                    "delete, space, or a single printable character."
                ),
            },
            "modifiers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["command", "control", "option", "shift"],
                },
                "description": "Optional modifier keys.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Action timeout in seconds, between 5 and 120.",
            },
        },
        "required": ["key"],
        "additionalProperties": False,
    }


def _computer_open_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "URL, application name, or local path to open.",
            },
            "kind": {
                "type": "string",
                "enum": ["auto", "url", "app", "path"],
                "description": "How to interpret target. Defaults to auto.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Action timeout in seconds, between 5 and 120.",
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    }


def _computer_wait_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "Seconds to wait, between 0.1 and 20. Defaults to 1.",
            },
        },
        "additionalProperties": False,
    }
