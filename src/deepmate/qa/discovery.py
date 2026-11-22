"""Project discovery for project-adaptive QA Audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from deepmate.qa.model import ProjectProfile


def discover_project(workspace: str | Path) -> ProjectProfile:
    """Infer project shape from common manifests, docs, and source layout."""
    root = Path(workspace).resolve()
    package_json = root / "package.json"
    package = _read_json_object(package_json) if package_json.is_file() else {}
    package_name = package.get("name")
    name = package_name.strip() if isinstance(package_name, str) else ""
    evidence: list[str] = []
    kinds: list[str] = []
    surfaces: list[str] = []
    package_managers: list[str] = []
    test_commands: list[str] = []
    run_commands: list[str] = []

    def add_kind(kind: str, ref: str) -> None:
        _append_unique(kinds, kind)
        _append_unique(evidence, ref)

    def add_surface(surface: str, ref: str) -> None:
        _append_unique(surfaces, surface)
        _append_unique(evidence, ref)

    if (root / "pyproject.toml").exists():
        add_kind("python", "pyproject.toml")
        add_surface("install_surface", "pyproject.toml")
        _append_unique(package_managers, "python")
        _append_unique(test_commands, "python3 -m unittest discover -s tests -v")
        if (root / "tests").is_dir():
            add_surface("test_surface", "tests/")
    if (root / "setup.py").exists() or (root / "setup.cfg").exists():
        add_kind("python", "setup.py/setup.cfg")
        add_surface("install_surface", "setup.py/setup.cfg")
    if (root / "requirements.txt").exists():
        add_kind("python", "requirements.txt")
        _append_unique(package_managers, "pip")
    if package_json.exists():
        add_kind("node", "package.json")
        add_surface("install_surface", "package.json")
        _append_unique(package_managers, "npm")
        scripts = package.get("scripts")
        if isinstance(scripts, Mapping):
            if "test" in scripts:
                _append_unique(test_commands, "npm test")
                add_surface("test_surface", "package.json scripts.test")
            if "build" in scripts:
                _append_unique(test_commands, "npm run build")
            for key in ("dev", "start", "serve"):
                if key in scripts:
                    _append_unique(run_commands, f"npm run {key}" if key != "start" else "npm start")
                    add_surface("service_surface", f"package.json scripts.{key}")
                    break
    if (root / "pnpm-lock.yaml").exists():
        _append_unique(package_managers, "pnpm")
    if (root / "yarn.lock").exists():
        _append_unique(package_managers, "yarn")
    if (root / "Cargo.toml").exists():
        add_kind("rust", "Cargo.toml")
        add_surface("install_surface", "Cargo.toml")
        _append_unique(test_commands, "cargo test")
    if (root / "go.mod").exists():
        add_kind("go", "go.mod")
        add_surface("install_surface", "go.mod")
        _append_unique(test_commands, "go test ./...")
    if (root / "pom.xml").exists():
        add_kind("java", "pom.xml")
        add_surface("install_surface", "pom.xml")
        _append_unique(test_commands, "mvn test")
    if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
        add_kind("java", "gradle build file")
        add_surface("install_surface", "gradle build file")
        _append_unique(test_commands, "./gradlew test")

    if (root / "README.md").exists() or (root / "README").exists():
        add_surface("docs_surface", "README")
    if any((root / name).exists() for name in ("Dockerfile", "docker-compose.yml", "compose.yml")):
        add_surface("ops_surface", "Dockerfile/docker-compose")
    if any((root / name).exists() for name in ("openapi.yaml", "openapi.yml", "openapi.json")):
        add_surface("api_surface", "OpenAPI spec")
    if any((root / name).exists() for name in ("playwright.config.ts", "playwright.config.js")):
        add_surface("ui_surface", "playwright config")
        _append_unique(test_commands, "npx playwright test")
    if any((root / name).exists() for name in ("vite.config.ts", "vite.config.js", "next.config.js", "next.config.mjs")):
        add_kind("web", "frontend config")
        add_surface("ui_surface", "frontend config")
    if any((root / name).exists() for name in ("electron", "pet_ui")) or _package_dep(package, "electron"):
        add_kind("desktop", "electron marker")
        add_surface("desktop_surface", "electron marker")
    markers = _source_markers(root)
    if "cli" in markers:
        add_surface("command_surface", "CLI markers")
    if "api" in markers:
        add_surface("api_surface", "API route/server markers")
    if "data" in markers:
        add_kind("data", "data file/notebook markers")
        add_surface("data_surface", "data file/notebook markers")
    if "agent" in markers:
        add_kind("agent", "agent/tool markers")
        add_surface("integration_surface", "agent/tool markers")

    if not kinds:
        _append_unique(kinds, "unknown")
    if not surfaces:
        _append_unique(surfaces, "artifact_surface")
        _append_unique(evidence, "workspace files")
    if test_commands:
        add_surface("test_surface", "test command detected")
    if run_commands:
        add_surface("service_surface", "run command detected")

    return ProjectProfile(
        project_name=name or root.name or "workspace",
        project_kinds=tuple(kinds),
        surfaces=tuple(surfaces),
        package_managers=tuple(package_managers),
        test_commands=tuple(test_commands),
        run_commands=tuple(run_commands),
        evidence=tuple(evidence[:16]),
    )


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _package_dep(package: Mapping[str, object], name: str) -> bool:
    for section in ("dependencies", "devDependencies", "optionalDependencies"):
        value = package.get(section)
        if isinstance(value, Mapping) and name in value:
            return True
    return False


_IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".deepmate", "var",
}
_DATA_SUFFIXES = {".ipynb", ".csv", ".parquet"}
_SOURCE_CHECKS = {
    ".py": ("cli", "api", "agent"),
    ".js": ("cli", "api", "agent"),
    ".ts": ("cli", "api", "agent"),
    ".go": ("cli", "api"),
    ".rs": ("cli",),
}
_CLI_MARKERS = ("argparse", "click.command", "commander(")
_API_MARKERS = ("fastapi", "flask", "express(", "app.get(", "@app.route", "http.server")
_AGENT_MARKERS = ("tool_call", "mcp", "agent", "subagent", "openai", "anthropic")


def _source_markers(root: Path) -> set[str]:
    """Inspect each candidate once, using only workspace-relative evidence."""
    found: set[str] = set()
    if any((root / name).exists() for name in ("bin", "cli", "cmd")):
        found.add("cli")
    if any((root / name).is_dir() for name in ("notebooks", "data", "datasets")):
        found.add("data")
    remaining = {"cli": 400, "api": 500, "agent": 500}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_DIRS)
        parent = Path(directory)
        parts = {part.lower() for part in parent.relative_to(root).parts}
        for name in sorted(files):
            path = parent / name
            suffix = path.suffix.lower()
            checks = {key for key in _SOURCE_CHECKS.get(suffix, ()) if key not in found and remaining[key]}
            needs_data_check = suffix in _DATA_SUFFIXES and "data" not in found
            if not (checks or needs_data_check) or path.is_symlink() or not path.is_file():
                continue
            if needs_data_check:
                found.add("data")
            else:
                for key in checks:
                    remaining[key] -= 1
                found.update(_file_markers(path, parts, checks))
            if "data" in found and all(key in found or not budget for key, budget in remaining.items()):
                return found
    return found


def _file_markers(path: Path, parts: set[str], checks: set[str]) -> set[str]:
    name = path.name.lower()
    found: set[str] = set()
    if "cli" in checks and (parts & {"cli", "cmd", "bin"} or name in {"cli.py", "main.py", "main.go", "main.rs"}):
        found.add("cli")
    if "api" in checks and (parts & {"api", "routes"} or name.startswith(("server.", "app."))):
        found.add("api")
    if "agent" in checks and parts & {"agents", "agent", "mcp", "tools"}:
        found.add("agent")
    missing = checks - found
    if not missing:
        return found
    try:
        with path.open(encoding="utf-8", errors="ignore") as file:
            text = file.read(8000)
    except OSError:
        return found
    if "cli" in missing and any(marker in text[:6000] for marker in _CLI_MARKERS):
        found.add("cli")
    lowered = text.lower()
    if "api" in missing and any(marker in lowered for marker in _API_MARKERS):
        found.add("api")
    if "agent" in missing and any(marker in lowered for marker in _AGENT_MARKERS):
        found.add("agent")
    return found


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
