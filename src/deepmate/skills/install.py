"""Inspect, install, and verify community-style skill bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from deepmate.capabilities.state import (
    CapabilityAssetState,
    CapabilitySource,
    CapabilityStateStore,
    CapabilityTemperature,
    skill_capability_id,
)
from deepmate.domain import CapabilityKind
from deepmate.foundation import display_path, normalize_name, utc_isoformat
from deepmate.skills.catalog import SkillCard, SkillCatalog, load_skill_card
from deepmate.skills.loader import SkillDocument, load_skill_document
from deepmate.skills.manifest import (
    InstalledSkillManifestStore,
    InstalledSkillRecord,
)
from deepmate.skills.skill_file import SKILL_FILE_NAME

DEFAULT_SKILL_TARGET = "workspace"
USER_SKILL_TARGETS = frozenset({"user", "global", "personal", "deepmate"})
HTTP_TIMEOUT_SECONDS = 20
MAX_SKILL_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_SKILL_TEXT_BYTES = 2 * 1024 * 1024
MAX_SKILL_ARCHIVE_MEMBER_BYTES = 30 * 1024 * 1024
MAX_SKILL_ARCHIVE_TOTAL_BYTES = 100 * 1024 * 1024
STANDARD_METADATA_KEYS = {
    "name",
    "description",
    "when_to_use",
    "disable-model-invocation",
    "allowed-tools",
    "allowed_tools",
    "metadata",
}
RUNTIME_SPECIFIC_KEYS = {
    "hooks",
    "commands",
    "slash_commands",
    "slash-commands",
    "mcp",
    "subagents",
}
RESOURCE_DIRS = ("references", "scripts", "assets", "agents", "examples")
SETUP_FILE_NAMES = {
    "install.sh",
    "setup.sh",
    "requirements.txt",
    "package.json",
    "pyproject.toml",
    "uv.lock",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
}


@dataclass(frozen=True, slots=True)
class SkillResourceSummary:
    """Summary of supporting files inside one skill bundle."""

    references: int = 0
    scripts: int = 0
    assets: int = 0
    agents: int = 0
    examples: int = 0
    other_files: int = 0
    setup_files: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "references": self.references,
            "scripts": self.scripts,
            "assets": self.assets,
            "agents": self.agents,
            "examples": self.examples,
            "other_files": self.other_files,
            "setup_files": list(self.setup_files),
        }


@dataclass(frozen=True, slots=True)
class SkillInstallCandidate:
    """One installable SKILL.md bundle candidate."""

    name: str
    description: str
    skill_path: Path
    bundle_path: Path
    resources: SkillResourceSummary
    metadata_keys: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "name": self.name,
            "description": self.description,
            "skill_path": str(self.skill_path),
            "bundle_path": str(self.bundle_path),
            "resources": self.resources.to_record(),
            "metadata_keys": list(self.metadata_keys),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SkillInspectionResult:
    """Result of inspecting a skill source."""

    source_kind: str
    source_ref: str
    compatibility: str
    candidates: tuple[SkillInstallCandidate, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    approval_required: tuple[str, ...] = field(default_factory=tuple)
    fatal_errors: tuple[str, ...] = field(default_factory=tuple)

    def is_installable(self) -> bool:
        """Return whether at least one standard bundle can be installed."""
        return bool(self.candidates) and not self.fatal_errors

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "compatibility": self.compatibility,
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "warnings": list(self.warnings),
            "approval_required": list(self.approval_required),
            "fatal_errors": list(self.fatal_errors),
        }


@dataclass(frozen=True, slots=True)
class SkillInstallResult:
    """Result of installing one skill bundle."""

    status: str
    skill: SkillDocument
    target_path: Path
    manifest_record: InstalledSkillRecord
    state_temperature: str
    state_source: str
    content_sha256: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "status": self.status,
            "skill": self.skill.name,
            "description": self.skill.description,
            "target_path": str(self.target_path),
            "manifest": self.manifest_record.to_record(),
            "state_temperature": self.state_temperature,
            "state_source": self.state_source,
            "content_sha256": self.content_sha256,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SkillBundleInstallResult:
    """End-to-end result for installing and preparing one skill bundle."""

    install: SkillInstallResult
    verify: SkillVerifyResult
    setup_status: str
    setup_command: str = ""
    setup_message: str = ""

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "install": self.install.to_record(),
            "verify": self.verify.to_record(),
            "setup_status": self.setup_status,
            "setup_command": self.setup_command,
            "setup_message": self.setup_message,
        }


@dataclass(frozen=True, slots=True)
class SkillVerifyResult:
    """Result of verifying one installed or local skill."""

    status: str
    skill: SkillDocument
    skill_path: Path
    manifest_record: InstalledSkillRecord | None = None
    state_temperature: str = ""
    state_source: str = ""
    resources: SkillResourceSummary = field(default_factory=SkillResourceSummary)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "status": self.status,
            "skill": self.skill.name,
            "description": self.skill.description,
            "skill_path": str(self.skill_path),
            "manifest": (
                self.manifest_record.to_record()
                if self.manifest_record is not None
                else None
            ),
            "state_temperature": self.state_temperature,
            "state_source": self.state_source,
            "resources": self.resources.to_record(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SkillUninstallResult:
    """Result of uninstalling or hiding one skill."""

    status: str
    name: str
    target_path: Path | None = None
    removed_manifest: bool = False
    deleted_files: bool = False
    archived_state: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record."""
        return {
            "status": self.status,
            "name": self.name,
            "target_path": str(self.target_path) if self.target_path is not None else "",
            "removed_manifest": self.removed_manifest,
            "deleted_files": self.deleted_files,
            "archived_state": self.archived_state,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    kind: str
    ref: str
    root: Path | None
    warnings: tuple[str, ...] = ()
    approval_required: tuple[str, ...] = ()
    fatal_errors: tuple[str, ...] = ()


def inspect_skill_source(
    source: str | Path,
    workspace: str | Path,
    *,
    skill_name: str = "",
) -> SkillInspectionResult:
    """Inspect a local or remote skill source without installing it."""
    with tempfile.TemporaryDirectory() as tmp:
        prepared = _prepare_source(source, Path(workspace), Path(tmp))
        return _inspect_prepared_source(prepared, skill_name=skill_name)


def install_skill_source(
    source: str | Path,
    workspace: str | Path,
    data_dir: str | Path,
    state_store: CapabilityStateStore,
    *,
    target: str = DEFAULT_SKILL_TARGET,
    skill_name: str = "",
    force: bool = False,
    update: bool = False,
    manifest_store: InstalledSkillManifestStore | None = None,
) -> SkillInstallResult:
    """Install one full skill bundle and connect it to capability state."""
    root = Path(workspace)
    manifest_store = manifest_store or InstalledSkillManifestStore.in_data_dir(data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = _prepare_source(source, root, Path(tmp))
        inspection = _inspect_prepared_source(prepared, skill_name=skill_name)
        if prepared.approval_required and not inspection.candidates:
            raise ValueError(
                "skill source requires environment approval before install: "
                + "; ".join(prepared.approval_required)
            )
        if not inspection.is_installable():
            raise ValueError(_inspection_error_message(inspection))
        candidate = _select_candidate(inspection.candidates, skill_name)
        data_root = Path(data_dir)
        destination = _target_skill_dir(root, data_root, target, skill_name or candidate.name)
        target_scope = _target_scope(target)
        status = "installed"
        backup_path: Path | None = None
        copied = False
        source_root = candidate.bundle_path.resolve()
        destination_resolved = destination.resolve() if destination.exists() else destination
        if destination.exists() and source_root == destination_resolved:
            status = "already_present"
        elif destination.exists():
            if not force and not update:
                raise ValueError(
                    f"skill destination already exists: {_display_path(destination, root)}"
                )
            backup_path = _backup_existing_skill(destination, Path(data_dir))
            _copy_bundle(candidate.bundle_path, destination)
            copied = True
            status = "updated" if update else "reinstalled"
        else:
            _copy_bundle(candidate.bundle_path, destination)
            copied = True

        try:
            card = load_skill_card(destination / SKILL_FILE_NAME)
            document = load_skill_document(card)
        except Exception:
            if copied:
                _restore_backup(destination, backup_path)
            raise

        if copied and backup_path is not None and backup_path.exists():
            shutil.rmtree(backup_path)

        content_hash = _bundle_sha256(destination)
        warning_tuple = tuple(dict.fromkeys((*inspection.warnings, *candidate.warnings)))
        timestamp = _isoformat(datetime.now(UTC))
        record = InstalledSkillRecord(
            name=document.name,
            source_kind=prepared.kind,
            source_ref=prepared.ref,
            target_path=_display_path_for_scope(destination, root, data_root, target_scope),
            target_scope=target_scope,
            installed_at=_existing_installed_at(manifest_store, document.name) or timestamp,
            updated_at=timestamp,
            content_sha256=content_hash,
            compatibility=inspection.compatibility,
            setup_status=_setup_status(candidate, prepared),
            warnings=warning_tuple,
        )
        try:
            manifest_store.upsert(record)
            if target_scope == "workspace":
                state = state_store.record_skill_installed(
                    card,
                    root,
                    source=CapabilitySource.IMPORTED,
                )
                state_temperature = state.temperature.value
                state_source = state.source.value
            else:
                state_temperature = "hot"
                state_source = "imported"
        except Exception:
            if copied:
                _restore_backup(destination, backup_path)
            raise
        return SkillInstallResult(
            status=status,
            skill=document,
            target_path=destination,
            manifest_record=record,
            state_temperature=state_temperature,
            state_source=state_source,
            content_sha256=content_hash,
            warnings=warning_tuple,
        )


def update_skill_source(
    source_or_name: str | Path,
    workspace: str | Path,
    data_dir: str | Path,
    state_store: CapabilityStateStore,
    *,
    target: str = DEFAULT_SKILL_TARGET,
    skill_name: str = "",
    manifest_store: InstalledSkillManifestStore | None = None,
) -> SkillInstallResult:
    """Update an installed skill from its original source or an explicit source."""
    manifest_store = manifest_store or InstalledSkillManifestStore.in_data_dir(data_dir)
    source_text = str(source_or_name)
    record = manifest_store.get(source_text)
    source = record.source_ref if record is not None else source_or_name
    name = skill_name or (record.name if record is not None else "")
    return install_skill_source(
        source,
        workspace,
        data_dir,
        state_store,
        target=target,
        skill_name=name,
        force=True,
        update=True,
        manifest_store=manifest_store,
    )


def install_skill_bundle(
    source: str | Path,
    workspace: str | Path,
    data_dir: str | Path,
    state_store: CapabilityStateStore,
    *,
    target: str = DEFAULT_SKILL_TARGET,
    skill_name: str = "",
    force: bool = False,
    manifest_store: InstalledSkillManifestStore | None = None,
) -> SkillBundleInstallResult:
    """Install, verify, and prepare setup metadata for one community skill bundle."""
    manifest_store = manifest_store or InstalledSkillManifestStore.in_data_dir(data_dir)
    install = install_skill_source(
        source,
        workspace,
        data_dir,
        state_store,
        target=target,
        skill_name=skill_name,
        force=force,
        manifest_store=manifest_store,
    )
    verify = verify_skill_install(
        install.skill.name,
        workspace,
        data_dir,
        state_store,
        manifest_store=manifest_store,
    )
    setup_command = _setup_command(install.target_path)
    setup_status = "not_required"
    setup_message = "No dependency setup was detected."
    if setup_command:
        setup_status = "approval_required"
        setup_message = (
            "Dependency setup is available but was not run automatically. "
            "Approve running setup to install dependencies."
        )
    updated_record = manifest_store.update_setup_status(
        install.skill.name,
        status=setup_status,
        command=setup_command,
        updated_at=_isoformat(datetime.now(UTC)),
    )
    if updated_record.setup_status == "not_required":
        setup_status = "not_required"
    return SkillBundleInstallResult(
        install=install,
        verify=verify,
        setup_status=setup_status,
        setup_command=setup_command,
        setup_message=setup_message,
    )


def verify_skill_install(
    name: str,
    workspace: str | Path,
    data_dir: str | Path,
    state_store: CapabilityStateStore,
    *,
    manifest_store: InstalledSkillManifestStore | None = None,
) -> SkillVerifyResult:
    """Verify that a skill can be discovered and its full body can be loaded."""
    root = Path(workspace)
    data_root = Path(data_dir)
    manifest_store = manifest_store or InstalledSkillManifestStore.in_data_dir(data_dir)
    cards = tuple(_discover_cards(root, data_root))
    if not cards:
        raise ValueError("no skills found")
    catalog = SkillCatalog(cards)
    card = catalog.get(name)
    if card is None:
        raise ValueError(f"skill not found: {name}")
    workspace_cards = tuple(card for card in cards if _is_workspace_skill_path(card.path, root))
    state_store.sync_workspace_skills(workspace_cards, root)
    states = state_store.skill_states_by_name()
    state = states.get(_normalize_name(card.name))
    document = load_skill_document(card)
    record = manifest_store.get(document.name)
    resources = _resource_summary(card.path.parent)
    warnings: list[str] = []
    if record is not None:
        manifest_target = _manifest_target_path(root, data_root, record).resolve()
        if manifest_target != card.path.parent.resolve():
            warnings.append(
                "manifest target differs from discovered skill path: "
                f"{record.target_path}"
            )
    return SkillVerifyResult(
        status="ok",
        skill=document,
        skill_path=card.path,
        manifest_record=record,
        state_temperature=state.temperature.value if state is not None else "",
        state_source=state.source.value if state is not None else "",
        resources=resources,
        warnings=tuple(warnings),
    )


def uninstall_skill(
    name: str,
    workspace: str | Path,
    data_dir: str | Path,
    state_store: CapabilityStateStore,
    *,
    force: bool = False,
    manifest_store: InstalledSkillManifestStore | None = None,
) -> SkillUninstallResult:
    """Uninstall an imported skill, or hide an untracked local skill."""
    root = Path(workspace)
    data_root = Path(data_dir)
    manifest_store = manifest_store or InstalledSkillManifestStore.in_data_dir(data_dir)
    record = manifest_store.remove(name)
    warnings: list[str] = []
    deleted_files = False
    target_path: Path | None = None
    if record is not None:
        target_path = _manifest_target_path(root, data_root, record).resolve()
        if _is_safe_skill_target(target_path, root, data_root) and target_path.exists():
            shutil.rmtree(target_path)
            deleted_files = True
        elif target_path.exists():
            warnings.append(f"manifest target is outside skill roots: {target_path}")
    elif force:
        card = SkillCatalog(_discover_cards(root, data_root)).get(name)
        if card is not None and _is_safe_skill_target(card.path.parent.resolve(), root, data_root):
            target_path = card.path.parent.resolve()
            shutil.rmtree(target_path)
            deleted_files = True
        else:
            warnings.append("skill was not installed by Deepmate and was not deleted")
    else:
        warnings.append(
            "skill has no Deepmate install manifest; files were left in place"
        )
    archived = _archive_skill_state(state_store, name)
    status = "uninstalled" if deleted_files else "hidden"
    return SkillUninstallResult(
        status=status,
        name=name,
        target_path=target_path,
        removed_manifest=record is not None,
        deleted_files=deleted_files,
        archived_state=archived,
        warnings=tuple(warnings),
    )


def format_skill_inspection(result: SkillInspectionResult, workspace: str | Path) -> str:
    """Render inspection output for CLI users."""
    root = Path(workspace)
    lines = [
        f"skill source: {result.source_kind}",
        f"- source: {result.source_ref}",
        f"- compatibility: {result.compatibility}",
        f"- candidates: {len(result.candidates)}",
    ]
    if result.approval_required:
        lines.append("- approval required:")
        lines.extend(f"  - {item}" for item in result.approval_required)
    if result.fatal_errors:
        lines.append("- errors:")
        lines.extend(f"  - {item}" for item in result.fatal_errors)
    if result.warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {item}" for item in result.warnings)
    for candidate in result.candidates:
        lines.extend(
            (
                "",
                f"candidate: {candidate.name}",
                f"- description: {candidate.description}",
                f"- skill: {_display_path(candidate.skill_path, root)}",
                f"- bundle: {_display_path(candidate.bundle_path, root)}",
                "- resources: "
                f"references={candidate.resources.references}, "
                f"scripts={candidate.resources.scripts}, "
                f"assets={candidate.resources.assets}, "
                f"agents={candidate.resources.agents}, "
                f"examples={candidate.resources.examples}, "
                f"other_files={candidate.resources.other_files}",
            )
        )
        if candidate.resources.setup_files:
            lines.append("- setup files:")
            lines.extend(f"  - {item}" for item in candidate.resources.setup_files)
        if candidate.warnings:
            lines.append("- candidate warnings:")
            lines.extend(f"  - {item}" for item in candidate.warnings)
    return "\n".join(lines)


def format_skill_install_result(result: SkillInstallResult, workspace: str | Path) -> str:
    """Render install output for CLI users."""
    root = Path(workspace)
    resources = _resource_summary(result.target_path)
    lines = [
        f"skill {result.status}: {result.skill.name}",
        f"- description: {result.skill.description}",
        f"- target: {_display_path(result.target_path, root)}",
        f"- scope: {result.manifest_record.target_scope or 'workspace'}",
        f"- source: {result.manifest_record.source_kind} {result.manifest_record.source_ref}",
        f"- sha256: {result.content_sha256}",
        f"- capability: source={result.state_source}, temperature={result.state_temperature}",
        "- resources: "
        f"references={resources.references}, "
        f"scripts={resources.scripts}, "
        f"assets={resources.assets}, "
        f"agents={resources.agents}, "
        f"examples={resources.examples}, "
        f"other_files={resources.other_files}",
    ]
    if resources.setup_files:
        lines.extend(
            (
                f"- setup_status: {result.manifest_record.setup_status or 'pending'}",
                "- setup_files: " + ", ".join(resources.setup_files[:6]),
                "- readiness: installed; dependency setup still needs review",
                "- next_step: run plan_skill_setup, then approve run_skill_setup if dependencies are needed",
            )
        )
    else:
        lines.append("- setup_status: not_required")
        lines.append("- readiness: ready to use")
    if result.warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {item}" for item in result.warnings)
    return "\n".join(lines)


def format_skill_bundle_install_result(
    result: SkillBundleInstallResult,
    workspace: str | Path,
) -> str:
    """Render a compact end-to-end install summary for agent-facing flows."""
    root = Path(workspace)
    install = result.install
    verify = result.verify
    lines = [
        f"Skill installed: {install.skill.name}",
        "- installed: yes",
        f"- description: {install.skill.description}",
        f"- target: {_display_path(install.target_path, root)}",
        f"- scope: {install.manifest_record.target_scope or 'workspace'}",
        f"- source: {install.manifest_record.source_kind} {install.manifest_record.source_ref}",
        f"- verified: {verify.status}",
        "- resources: "
        f"references={verify.resources.references}, "
        f"scripts={verify.resources.scripts}, "
        f"assets={verify.resources.assets}, "
        f"agents={verify.resources.agents}, "
        f"examples={verify.resources.examples}, "
        f"other_files={verify.resources.other_files}",
        f"- setup_status: {result.setup_status}",
    ]
    if result.setup_command:
        lines.extend(
            (
                f"- setup_command: {result.setup_command}",
                "- readiness: installed and verified; dependency setup is pending approval",
                f"- next_step: Setup command detected. Call run_skill_setup(name='{install.skill.name}') to execute it after user approval.",
            )
        )
    else:
        lines.append("- readiness: ready to use")
        lines.append("- next_step: load or use the skill when relevant")
    if result.setup_message:
        lines.append(f"- note: {result.setup_message}")
    warnings = tuple(dict.fromkeys((*install.warnings, *verify.warnings)))
    if warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {item}" for item in warnings)
    return "\n".join(lines)


def format_skill_verify_result(result: SkillVerifyResult, workspace: str | Path) -> str:
    """Render verify output for CLI users."""
    root = Path(workspace)
    lines = [
        f"skill verify: {result.status}",
        f"- skill: {result.skill.name}",
        f"- description: {result.skill.description}",
        f"- path: {_display_path(result.skill_path, root)}",
        "- resources: "
        f"references={result.resources.references}, "
        f"scripts={result.resources.scripts}, "
        f"assets={result.resources.assets}, "
        f"agents={result.resources.agents}, "
        f"examples={result.resources.examples}, "
        f"other_files={result.resources.other_files}",
    ]
    if result.state_source or result.state_temperature:
        lines.append(
            f"- capability: source={result.state_source}, "
            f"temperature={result.state_temperature}"
        )
    if result.manifest_record is not None:
        lines.append(f"- manifest source: {result.manifest_record.source_ref}")
        lines.append(f"- manifest scope: {result.manifest_record.target_scope or 'workspace'}")
    if result.warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {item}" for item in result.warnings)
    return "\n".join(lines)


def format_skill_uninstall_result(result: SkillUninstallResult, workspace: str | Path) -> str:
    """Render uninstall output for CLI users."""
    root = Path(workspace)
    lines = [
        f"skill {result.status}: {result.name}",
        f"- removed_manifest: {str(result.removed_manifest).lower()}",
        f"- deleted_files: {str(result.deleted_files).lower()}",
        f"- archived_state: {str(result.archived_state).lower()}",
    ]
    if result.target_path is not None:
        lines.append(f"- target: {_display_path(result.target_path, root)}")
    if result.warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {item}" for item in result.warnings)
    return "\n".join(lines)


def format_installed_skill_list(
    manifest_store: InstalledSkillManifestStore,
    workspace: str | Path,
) -> str:
    """Render installed skill manifest records."""
    records = tuple(manifest_store.load().values())
    if not records:
        return "No installed skills found."
    lines = [f"{'SKILL':<28}  {'SCOPE':<10}  {'SOURCE':<12}  TARGET"]
    for record in sorted(records, key=lambda item: _normalize_name(item.name)):
        lines.append(
            f"{record.name:<28}  "
            f"{record.target_scope or 'workspace':<10}  "
            f"{record.source_kind:<12}  "
            f"{record.target_path}"
        )
    return "\n".join(lines)


def _prepare_source(
    source: str | Path,
    workspace: Path,
    tmp_dir: Path,
    *,
    visited: frozenset[str] = frozenset(),
) -> _PreparedSource:
    clean_source = str(source).strip()
    if not clean_source:
        return _PreparedSource(
            kind="unknown",
            ref=source,
            root=None,
            fatal_errors=("skill source cannot be empty",),
        )
    if clean_source in visited:
        return _PreparedSource(
            kind="unknown",
            ref=clean_source,
            root=None,
            fatal_errors=("recursive skill source resolution detected",),
        )
    path = Path(clean_source).expanduser()
    if not path.is_absolute():
        path = workspace / path
    if path.exists():
        if path.is_dir():
            return _PreparedSource(kind="local_dir", ref=clean_source, root=path)
        if _is_archive(path.name):
            root = tmp_dir / "archive"
            root.mkdir(parents=True, exist_ok=True)
            _extract_archive(path, root)
            return _PreparedSource(kind="local_archive", ref=clean_source, root=root)
        return _PreparedSource(
            kind="local_file",
            ref=clean_source,
            root=None,
            fatal_errors=(f"local source is not a skill directory or archive: {clean_source}",),
        )
    if _looks_like_skill_install_command(clean_source):
        return _PreparedSource(
            kind="install_instruction",
            ref=clean_source,
            root=None,
            approval_required=(
                "Skill installation commands require explicit environment-change "
                "approval; Deepmate did not run this instruction automatically.",
            ),
        )
    parsed = urllib.parse.urlparse(clean_source)
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc.lower()
        if "github.com" in host:
            return _prepare_github_source(clean_source, tmp_dir)
        if _looks_like_remote_skill_page(host):
            return _prepare_page_source(
                clean_source,
                tmp_dir,
                kind="remote_skill_page",
                visited=visited,
                workspace=workspace,
            )
        if _is_archive(parsed.path):
            archive_path = _download_to(tmp_dir, clean_source, "skill-archive")
            root = tmp_dir / "remote-archive"
            root.mkdir(parents=True, exist_ok=True)
            _extract_archive(archive_path, root)
            return _PreparedSource(kind="remote_archive", ref=clean_source, root=root)
        return _prepare_page_source(
            clean_source,
            tmp_dir,
            kind="remote_page",
            visited=visited,
            workspace=workspace,
        )
    return _PreparedSource(
        kind="unknown",
        ref=clean_source,
        root=None,
        fatal_errors=(f"skill source not found or unsupported: {clean_source}",),
    )


def _prepare_page_source(
    url: str,
    tmp_dir: Path,
    *,
    kind: str,
    visited: frozenset[str],
    workspace: Path,
) -> _PreparedSource:
    try:
        html = _fetch_text(url)
    except OSError as exc:
        return _PreparedSource(
            kind=kind,
            ref=url,
            root=None,
            fatal_errors=(f"failed to fetch skill page: {exc}",),
        )
    linked_source = _first_installable_link(html)
    if linked_source:
        prepared = _prepare_source(
            linked_source,
            workspace,
            tmp_dir,
            visited=frozenset((*visited, url)),
        )
        warnings = (
            f"{kind} page resolved to {linked_source}",
            *prepared.warnings,
        )
        return _PreparedSource(
            kind=kind,
            ref=url,
            root=prepared.root,
            warnings=warnings,
            approval_required=prepared.approval_required,
            fatal_errors=prepared.fatal_errors,
        )
    embedded = _embedded_skill_page_source(html, tmp_dir, url)
    if embedded is not None:
        return embedded
    approval_required = ()
    if "curl" in html and "bash" in html:
        approval_required = (
            "page contains curl/bash install instructions; Deepmate did not run them automatically",
        )
    if _contains_skill_install_instruction(html):
        approval_required = (
            *approval_required,
            "page contains a skill installation command; running it requires approval",
        )
    return _PreparedSource(
        kind=kind,
        ref=url,
        root=None,
        approval_required=approval_required,
        fatal_errors=(
            ()
            if approval_required
            else ("no downloadable or GitHub skill bundle link found on page",)
        ),
    )


def _embedded_skill_page_source(html: str, tmp_dir: Path, url: str) -> _PreparedSource | None:
    text = _html_to_text(html)
    skill_name = _embedded_skill_name(text, url)
    if not skill_name:
        return None
    body = _embedded_skill_body(text, skill_name)
    if not body:
        return None
    root = tmp_dir / "remote-skill-page" / _safe_dir_name(skill_name)
    root.mkdir(parents=True, exist_ok=True)
    description = _embedded_skill_description(body, skill_name)
    skill_file = root / SKILL_FILE_NAME
    skill_file.write_text(
        "\n".join(
            (
                "---",
                f"name: {skill_name}",
                f"description: {description}",
                "---",
                "",
                body,
                "",
            )
        ),
        encoding="utf-8",
    )
    return _PreparedSource(
        kind="remote_skill_page",
        ref=url,
        root=root,
        warnings=("remote skill page converted to SKILL.md",),
    )


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "\n", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|main|h[1-6]|li|tr)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    replacements = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _embedded_skill_name(text: str, url: str) -> str:
    for pattern in (
        r"(?im)^#\s+([A-Za-z0-9][\w .-]{1,80})\s*$",
        r"(?im)^([A-Za-z0-9][\w .-]{1,80})\s+SKILL\.md\b",
    ):
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if _embedded_name_is_plausible(candidate):
                return _safe_dir_name(candidate)
    parsed = urllib.parse.urlparse(url)
    tail = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return _safe_dir_name(tail) if tail else ""


def _embedded_name_is_plausible(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered not in {"skill card", "files", "versions", "download"}


def _embedded_skill_body(text: str, skill_name: str) -> str:
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if re.match(r"^#\s+", line):
            start = index
            break
        if _safe_dir_name(line) == _safe_dir_name(skill_name):
            start = index
            break
    body_lines: list[str] = []
    skip_markers = {
        "SKILL.md",
        "Skill Card",
        "Files",
        "Versions",
        "Download",
    }
    for line in lines[start:]:
        clean = line.strip()
        if not clean or clean in skip_markers:
            continue
        if clean.lower().startswith(("downloads ", "version ")):
            continue
        body_lines.append(clean)
    return "\n".join(body_lines).strip()


def _embedded_skill_description(body: str, skill_name: str) -> str:
    for line in body.splitlines():
        clean = line.strip().lstrip("#").strip()
        if clean and _safe_dir_name(clean) != _safe_dir_name(skill_name):
            return _yaml_scalar(clean[:180])
    return _yaml_scalar(f"{skill_name} skill imported from a remote skill page.")


def _yaml_scalar(value: str) -> str:
    clean = " ".join(value.strip().split()).replace('"', '\\"')
    return f'"{clean or "Imported skill."}"'


def _prepare_github_source(url: str, tmp_dir: Path) -> _PreparedSource:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return _PreparedSource(
            kind="github",
            ref=url,
            root=None,
            fatal_errors=("GitHub source requires owner and repo",),
        )
    owner, repo = parts[0], parts[1].removesuffix(".git")
    branch = ""
    subpath = ""
    if len(parts) >= 5 and parts[2] == "tree":
        branch = parts[3]
        subpath = "/".join(parts[4:])
    if len(parts) >= 5 and parts[2] == "blob":
        branch = parts[3]
        subpath = "/".join(parts[4:-1])
    branches = (branch,) if branch else (_github_default_branch(owner, repo), "main", "master")
    errors: list[str] = []
    for candidate_branch in tuple(dict.fromkeys(item for item in branches if item)):
        archive_url = (
            f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/"
            f"{urllib.parse.quote(candidate_branch)}"
        )
        try:
            archive_path = _download_to(tmp_dir, archive_url, "github.zip")
            root = tmp_dir / "github"
            root.mkdir(parents=True, exist_ok=True)
            _extract_archive(archive_path, root)
            extracted_root = _single_child_dir(root) or root
            search_root = extracted_root / subpath if subpath else extracted_root
            if not search_root.exists():
                errors.append(f"GitHub path not found: {subpath or '.'}")
                continue
            return _PreparedSource(
                kind="github",
                ref=url,
                root=search_root,
                warnings=(f"GitHub archive branch: {candidate_branch}",),
            )
        except OSError as exc:
            errors.append(str(exc))
    return _PreparedSource(
        kind="github",
        ref=url,
        root=None,
        fatal_errors=tuple(errors or ("failed to download GitHub archive",)),
    )


def _inspect_prepared_source(
    prepared: _PreparedSource,
    *,
    skill_name: str = "",
) -> SkillInspectionResult:
    candidates: tuple[SkillInstallCandidate, ...] = ()
    fatal_errors = list(prepared.fatal_errors)
    warnings = list(prepared.warnings)
    if prepared.root is not None:
        candidates, discovery_warnings = _candidate_bundles(prepared.root)
        warnings.extend(discovery_warnings)
        if skill_name.strip():
            candidates = tuple(
                candidate
                for candidate in candidates
                if _matches_candidate(candidate, skill_name)
            )
            if not candidates:
                fatal_errors.append(f"skill candidate not found: {skill_name}")
        if not candidates and not fatal_errors:
            fatal_errors.append(f"no {SKILL_FILE_NAME} bundle found")
    compatibility = _compatibility(
        candidates=candidates,
        warnings=warnings,
        fatal_errors=fatal_errors,
        approval_required=prepared.approval_required,
    )
    return SkillInspectionResult(
        source_kind=prepared.kind,
        source_ref=prepared.ref,
        compatibility=compatibility,
        candidates=candidates,
        warnings=tuple(dict.fromkeys(warnings)),
        approval_required=prepared.approval_required,
        fatal_errors=tuple(dict.fromkeys(fatal_errors)),
    )


def _candidate_bundles(root: Path) -> tuple[tuple[SkillInstallCandidate, ...], tuple[str, ...]]:
    candidates: list[SkillInstallCandidate] = []
    warnings: list[str] = []
    for skill_path in _iter_skill_files(root):
        try:
            card = load_skill_card(skill_path)
        except (OSError, ValueError) as exc:
            warnings.append(f"skill skipped: {exc}")
            continue
        resources = _resource_summary(skill_path.parent)
        candidate_warnings = _candidate_warnings(card, resources)
        candidates.append(
            SkillInstallCandidate(
                name=card.name,
                description=card.description,
                skill_path=skill_path,
                bundle_path=skill_path.parent,
                resources=resources,
                metadata_keys=tuple(sorted(str(key) for key in card.metadata.keys())),
                warnings=candidate_warnings,
            )
        )
    return tuple(candidates), tuple(warnings)


def _setup_status(candidate: SkillInstallCandidate, prepared: _PreparedSource) -> str:
    if prepared.approval_required:
        return "approval_required"
    if candidate.resources.scripts:
        return "pending"
    return "not_required"


def _setup_command(target: Path) -> str:
    scripts = target / "scripts"
    if not scripts.is_dir():
        return ""
    setup_py = scripts / "setup.py"
    if setup_py.is_file():
        return f"python3 {setup_py.relative_to(target).as_posix()}"
    setup_sh = scripts / "setup.sh"
    if setup_sh.is_file():
        return f"sh {setup_sh.relative_to(target).as_posix()}"
    first_script = next(
        (path for path in sorted(scripts.iterdir()) if path.is_file()),
        None,
    )
    if first_script is None:
        return ""
    return f"# inspect before running: {first_script.relative_to(target).as_posix()}"


def _iter_skill_files(root: Path) -> tuple[Path, ...]:
    if (root / SKILL_FILE_NAME).is_file():
        return (root / SKILL_FILE_NAME,)
    return tuple(sorted(root.rglob(SKILL_FILE_NAME)))


def _candidate_warnings(
    card: SkillCard,
    resources: SkillResourceSummary,
) -> tuple[str, ...]:
    warnings: list[str] = []
    metadata_keys = {str(key) for key in card.metadata.keys()}
    runtime_keys = tuple(sorted(metadata_keys & RUNTIME_SPECIFIC_KEYS))
    unknown_keys = tuple(sorted(metadata_keys - STANDARD_METADATA_KEYS - RUNTIME_SPECIFIC_KEYS))
    if runtime_keys:
        warnings.append(
            "runtime-specific metadata preserved but not enforced by Deepmate: "
            + ", ".join(runtime_keys)
        )
    if unknown_keys:
        warnings.append(
            "unknown metadata preserved but not interpreted: " + ", ".join(unknown_keys)
        )
    if "allowed-tools" in metadata_keys or "allowed_tools" in metadata_keys:
        warnings.append("allowed-tools metadata is preserved but not enforced yet")
    if resources.setup_files:
        warnings.append(
            "setup or dependency files found; Deepmate installed the bundle but did not "
            "run setup commands automatically"
        )
    return tuple(warnings)


def _resource_summary(bundle_path: Path) -> SkillResourceSummary:
    counts = {name: _file_count(bundle_path / name) for name in RESOURCE_DIRS}
    known_dirs = set(RESOURCE_DIRS)
    other_files = 0
    setup_files: list[str] = []
    for path in bundle_path.rglob("*"):
        if not path.is_file() or path.name == SKILL_FILE_NAME:
            continue
        try:
            relative = path.relative_to(bundle_path)
        except ValueError:
            continue
        first_part = relative.parts[0] if relative.parts else ""
        if first_part not in known_dirs:
            other_files += 1
        if path.name in SETUP_FILE_NAMES or path.name.startswith(("install.", "setup.")):
            setup_files.append(str(relative))
    return SkillResourceSummary(
        references=counts["references"],
        scripts=counts["scripts"],
        assets=counts["assets"],
        agents=counts["agents"],
        examples=counts["examples"],
        other_files=other_files,
        setup_files=tuple(sorted(set(setup_files))),
    )


def _file_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _select_candidate(
    candidates: tuple[SkillInstallCandidate, ...],
    skill_name: str,
) -> SkillInstallCandidate:
    if skill_name.strip():
        for candidate in candidates:
            if _matches_candidate(candidate, skill_name):
                return candidate
        raise ValueError(f"skill candidate not found: {skill_name}")
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(
        "skill source contains multiple candidates; pass --skill-name. "
        f"Candidates: {names}"
    )


def _matches_candidate(candidate: SkillInstallCandidate, skill_name: str) -> bool:
    clean = _normalize_name(skill_name)
    return clean in {
        _normalize_name(candidate.name),
        _normalize_name(candidate.bundle_path.name),
    }


def user_skill_library_root(data_dir: str | Path) -> Path:
    """Return the user-level skill library root."""
    return Path(data_dir) / "skills" / "library"


def _target_skill_dir(workspace: Path, data_dir: Path, target: str, name: str) -> Path:
    clean_target = target.strip().lower() or DEFAULT_SKILL_TARGET
    if clean_target in USER_SKILL_TARGETS:
        root = user_skill_library_root(data_dir)
    elif clean_target in {"workspace", "skills"}:
        root = workspace / "skills"
    else:
        candidate = Path(target)
        root = candidate if candidate.is_absolute() else workspace / candidate
    return root / _safe_dir_name(name)


def _target_scope(target: str) -> str:
    clean_target = target.strip().lower() or DEFAULT_SKILL_TARGET
    return "user" if clean_target in USER_SKILL_TARGETS else "workspace"


def _display_path_for_scope(
    path: Path,
    workspace: Path,
    data_dir: Path,
    scope: str,
) -> str:
    root = data_dir if scope == "user" else workspace
    return _display_path(path, root)


def _manifest_target_path(
    workspace: Path,
    data_dir: Path,
    record: InstalledSkillRecord,
) -> Path:
    base = data_dir if (record.target_scope or "workspace") == "user" else workspace
    target = Path(record.target_path)
    return target if target.is_absolute() else base / target


def _safe_dir_name(name: str) -> str:
    clean = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", name.strip(), flags=re.UNICODE)
    clean = clean.strip("-._").lower()
    return clean[:80] or "skill"


def _copy_bundle(source: Path, destination: Path) -> None:
    _reject_links_in_local_bundle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _reject_links_in_local_bundle(source: Path) -> None:
    for path in (source, *source.rglob("*")):
        if path.is_symlink():
            try:
                relative = path.relative_to(source)
            except ValueError:
                relative = path
            raise ValueError(f"local skill bundle uses unsupported link: {relative}")


def _backup_existing_skill(destination: Path, data_dir: Path) -> Path:
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace(":", "-")
    backup_root = data_dir / "skills" / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / f"{destination.name}-{timestamp}"
    shutil.move(str(destination), str(backup_path))
    return backup_path


def _restore_backup(destination: Path, backup_path: Path | None) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    if backup_path is not None and backup_path.exists():
        shutil.move(str(backup_path), str(destination))


def _discover_cards(workspace: Path, data_dir: Path | None = None) -> tuple[SkillCard, ...]:
    roots = (
        workspace / "skills",
        workspace / ".claude" / "skills",
        *((user_skill_library_root(data_dir),) if data_dir is not None else ()),
    )
    cards: list[SkillCard] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for skill_path in sorted(root.rglob(SKILL_FILE_NAME)):
            card = load_skill_card(skill_path)
            key = _normalize_name(card.name)
            directory_key = _normalize_name(card.path.parent.name)
            if key in seen or directory_key in seen:
                continue
            seen.add(key)
            seen.add(directory_key)
            cards.append(card)
    return tuple(cards)


def _archive_skill_state(state_store: CapabilityStateStore, name: str) -> bool:
    states = state_store.load()
    capability_id = skill_capability_id(name)
    state = states.get(capability_id)
    if state is None:
        return False
    states[capability_id] = state.__class__(
        capability_id=state.capability_id,
        kind=CapabilityKind.SKILL,
        name=state.name,
        path_or_ref=state.path_or_ref,
        source=state.source,
        scope=state.scope,
        temperature=CapabilityTemperature.COLD,
        hidden=True,
        asset_state=CapabilityAssetState.ARCHIVED,
        created_at=state.created_at,
        updated_at=_isoformat(datetime.now(UTC)),
        last_seen_at=state.last_seen_at,
        last_used_at=state.last_used_at,
        invocation_count=state.invocation_count,
        last_selected_at=state.last_selected_at,
    )
    state_store.save(states)
    return True


def _is_workspace_skill_path(path: Path, workspace: Path) -> bool:
    return _is_relative_to(
        path.resolve(),
        (workspace / "skills").resolve(),
    ) or _is_relative_to(path.resolve(), (workspace / ".claude" / "skills").resolve())


def _is_safe_skill_target(path: Path, workspace: Path, data_dir: Path | None = None) -> bool:
    roots = (
        (workspace / ".claude" / "skills").resolve(),
        (workspace / "skills").resolve(),
        *((user_skill_library_root(data_dir).resolve(),) if data_dir is not None else ()),
    )
    resolved = path.resolve()
    return any(_is_relative_to(resolved, root) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _existing_installed_at(
    manifest_store: InstalledSkillManifestStore,
    name: str,
) -> str:
    record = manifest_store.get(name)
    return record.installed_at if record is not None else ""


def _bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _extract_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            total_size = 0
            for member in archive.infolist():
                if _zip_member_is_link(member):
                    raise ValueError(f"archive member uses unsupported link: {member.filename}")
                if member.file_size > MAX_SKILL_ARCHIVE_MEMBER_BYTES:
                    raise ValueError(
                        f"archive member is too large: {member.filename}"
                    )
                total_size += member.file_size
                if total_size > MAX_SKILL_ARCHIVE_TOTAL_BYTES:
                    raise ValueError("archive contents are too large")
                target = destination / member.filename
                if not _is_relative_to(target.resolve(), destination.resolve()):
                    raise ValueError(f"archive member escapes destination: {member.filename}")
            archive.extractall(destination)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            total_size = 0
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError(
                        f"archive member uses unsupported link: {member.name}"
                    )
                if member.isdev() or member.isblk() or member.ischr() or member.isfifo():
                    raise ValueError(
                        f"archive member uses unsupported special file: {member.name}"
                    )
                if member.isfile():
                    if member.size > MAX_SKILL_ARCHIVE_MEMBER_BYTES:
                        raise ValueError(f"archive member is too large: {member.name}")
                    total_size += member.size
                    if total_size > MAX_SKILL_ARCHIVE_TOTAL_BYTES:
                        raise ValueError("archive contents are too large")
                target = destination / member.name
                if not _is_relative_to(target.resolve(), destination.resolve()):
                    raise ValueError(f"archive member escapes destination: {member.name}")
            archive.extractall(destination)
        return
    raise ValueError(f"unsupported archive type: {archive_path}")


def _zip_member_is_link(member: zipfile.ZipInfo) -> bool:
    return ((member.external_attr >> 16) & 0o170000) == 0o120000


def _download_to(tmp_dir: Path, url: str, filename: str) -> Path:
    path = tmp_dir / filename
    _validate_public_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Deepmate Skill Installer"},
    )
    try:
        print(f"Downloading skill bundle: {url}", file=sys.stderr)
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            final_url = getattr(response, "geturl", lambda: url)()
            _validate_public_url(str(final_url))
            path.write_bytes(_read_limited(response, MAX_SKILL_DOWNLOAD_BYTES))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OSError(_download_error_message(url, exc)) from exc
    return path


def _fetch_text(url: str) -> str:
    _validate_public_url(url)
    headers = {"User-Agent": "Deepmate Skill Installer"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token and urllib.parse.urlparse(url).netloc.lower() == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            final_url = getattr(response, "geturl", lambda: url)()
            _validate_public_url(str(final_url))
            data = _read_limited(response, MAX_SKILL_TEXT_BYTES)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OSError(_download_error_message(url, exc)) from exc
    return data.decode("utf-8", errors="replace")


def _download_error_message(url: str, exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    detail = str(reason or exc).strip() or exc.__class__.__name__
    return f"failed to download {url}: {detail}"


def _validate_public_url(url: str) -> None:
    from deepmate.tools.url_safety import validate_public_url

    validate_public_url(url)


def _read_limited(response: object, max_bytes: int) -> bytes:
    data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise OSError(f"remote skill response exceeds {max_bytes} bytes")
    return data


def _github_default_branch(owner: str, repo: str) -> str:
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        data = json.loads(_fetch_text(api_url))
    except (OSError, json.JSONDecodeError):
        return "main"
    if isinstance(data, Mapping):
        branch = data.get("default_branch")
        if isinstance(branch, str) and branch.strip():
            return branch.strip()
    return "main"


def _first_installable_link(html: str) -> str:
    links = re.findall(r"https?://[^\"'<>\\\s]+", html)
    for link in links:
        clean = link.rstrip(").,;")
        if _is_archive(urllib.parse.urlparse(clean).path):
            return clean
    for link in links:
        clean = link.rstrip(").,;")
        if "github.com" in urllib.parse.urlparse(clean).netloc.lower():
            return clean
    return ""


def _looks_like_skill_install_command(source: str) -> bool:
    clean = " ".join(source.strip().split()).lower()
    return clean.startswith("skill install:") or clean.startswith("skill install ")


def _looks_like_remote_skill_page(host: str) -> bool:
    return "skill" in host


def _contains_skill_install_instruction(text: str) -> bool:
    return "skill install" in text.lower()


def _is_archive(path_or_name: str) -> bool:
    clean = path_or_name.lower()
    return clean.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2"))


def _single_child_dir(path: Path) -> Path | None:
    children = tuple(child for child in path.iterdir() if child.is_dir())
    if len(children) == 1:
        return children[0]
    return None


def _compatibility(
    *,
    candidates: tuple[SkillInstallCandidate, ...],
    warnings: Iterable[str],
    fatal_errors: Iterable[str],
    approval_required: Iterable[str],
) -> str:
    if tuple(fatal_errors):
        return "incompatible"
    if tuple(approval_required) and not candidates:
        return "requires_approval"
    if tuple(warnings) or any(candidate.warnings for candidate in candidates):
        return "compatible_with_warnings"
    return "compatible"


def _inspection_error_message(inspection: SkillInspectionResult) -> str:
    if inspection.fatal_errors:
        return "; ".join(inspection.fatal_errors)
    return f"skill source is not installable: {inspection.source_ref}"


def _skill_document_content(document: SkillDocument) -> str:
    return "\n".join(
        (
            "<skill>",
            f"<name>{document.name.strip()}</name>",
            f"<description>{document.description.strip()}</description>",
            "<instructions>",
            _expand_skill_dir_placeholders(document.body.strip(), document.path.parent),
            "</instructions>",
            "</skill>",
        )
    )


def _expand_skill_dir_placeholders(body: str, skill_dir: Path) -> str:
    replacement = str(skill_dir)
    return (
        body.replace("$SKILL_DIR", replacement)
        .replace("${SKILL_DIR}", replacement)
        .replace("${SKILL_ROOT}", replacement)
    )


def _display_path(path: Path, root: Path) -> str:
    return display_path(path, root)


def _normalize_name(name: str) -> str:
    return normalize_name(name)


def _isoformat(value: datetime) -> str:
    return utc_isoformat(value)
