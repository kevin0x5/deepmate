"""Project discovery for project-adaptive QA Audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from deepmate.qa.model import ProjectProfile


def discover_project(workspace: str | Path) -> ProjectProfile:
    """Infer project shape from common manifests, docs, and source layout."""
    root = Path(workspace).resolve()
    name = _project_name(root)
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
    package_json = root / "package.json"
    if package_json.exists():
        add_kind("node", "package.json")
        add_surface("install_surface", "package.json")
        _append_unique(package_managers, "npm")
        package = _read_json_object(package_json)
        scripts = package.get("scripts") if isinstance(package, Mapping) else None
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
    if any((root / name).exists() for name in ("electron", "pet_ui")) or _package_dep(package_json, "electron"):
        add_kind("desktop", "electron marker")
        add_surface("desktop_surface", "electron marker")
    if _has_cli_markers(root):
        add_surface("command_surface", "CLI markers")
    if _has_api_markers(root):
        add_surface("api_surface", "API route/server markers")
    if _has_data_markers(root):
        add_kind("data", "data file/notebook markers")
        add_surface("data_surface", "data file/notebook markers")
    if _has_agent_markers(root):
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
        project_name=name,
        project_kinds=tuple(kinds),
        surfaces=tuple(surfaces),
        package_managers=tuple(package_managers),
        test_commands=tuple(test_commands),
        run_commands=tuple(run_commands),
        evidence=tuple(evidence[:16]),
    )


def _project_name(root: Path) -> str:
    package_json = root / "package.json"
    if package_json.exists():
        package = _read_json_object(package_json)
        name = package.get("name") if isinstance(package, Mapping) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    return root.name or "workspace"


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _package_dep(package_json: Path, name: str) -> bool:
    if not package_json.exists():
        return False
    package = _read_json_object(package_json)
    for section in ("dependencies", "devDependencies", "optionalDependencies"):
        value = package.get(section)
        if isinstance(value, Mapping) and name in value:
            return True
    return False


def _has_cli_markers(root: Path) -> bool:
    if any((root / name).exists() for name in ("bin", "cli", "cmd")):
        return True
    for path in _iter_limited(root, ("*.py", "*.js", "*.ts", "*.go", "*.rs"), limit=400):
        name = path.name.lower()
        if name in {"cli.py", "main.py", "main.go", "main.rs"} or any(
            part.lower() in {"cli", "cmd", "bin"} for part in path.parts
        ):
            return True
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:6000]
        except OSError:
            continue
        if "argparse" in text or "click.command" in text or "commander(" in text:
            return True
    return False


def _has_api_markers(root: Path) -> bool:
    markers = ("fastapi", "flask", "express(", "app.get(", "@app.route", "http.server")
    for path in _iter_limited(root, ("*.py", "*.js", "*.ts", "*.go"), limit=500):
        lowered_path = "/".join(part.lower() for part in path.parts)
        if any(part in lowered_path for part in ("/api/", "/routes/", "/server.", "/app.")):
            return True
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:8000].lower()
        except OSError:
            continue
        if any(marker in text for marker in markers):
            return True
    return False


def _has_data_markers(root: Path) -> bool:
    for name in ("notebooks", "data", "datasets"):
        if (root / name).is_dir():
            return True
    return any(_iter_limited(root, ("*.ipynb", "*.csv", "*.parquet"), limit=20))


def _has_agent_markers(root: Path) -> bool:
    markers = ("tool_call", "mcp", "agent", "subagent", "openai", "anthropic")
    for path in _iter_limited(root, ("*.py", "*.js", "*.ts"), limit=500):
        lowered_path = "/".join(part.lower() for part in path.parts)
        if any(part in lowered_path for part in ("/agents/", "/agent/", "/mcp/", "/tools/")):
            return True
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:8000].lower()
        except OSError:
            continue
        if any(marker in text for marker in markers):
            return True
    return False


def _iter_limited(root: Path, patterns: tuple[str, ...], *, limit: int):
    count = 0
    ignored = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if any(part in ignored for part in path.parts):
                continue
            if not path.is_file():
                continue
            yield path
            count += 1
            if count >= limit:
                return


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
