"""Skill discovery helpers for Deepmate."""

from deepmate.skills.catalog import SkillCard, SkillCatalog, load_skill_card
from deepmate.skills.install import (
    SkillInspectionResult,
    SkillBundleInstallResult,
    SkillInstallResult,
    SkillUninstallResult,
    SkillVerifyResult,
    format_skill_bundle_install_result,
    format_installed_skill_list,
    format_skill_inspection,
    format_skill_install_result,
    format_skill_uninstall_result,
    format_skill_verify_result,
    inspect_skill_source,
    install_skill_bundle,
    install_skill_source,
    uninstall_skill,
    update_skill_source,
    verify_skill_install,
)
from deepmate.skills.loader import SkillDocument, load_skill_document
from deepmate.skills.manifest import (
    InstalledSkillManifestStore,
    InstalledSkillRecord,
)

__all__ = [
    "InstalledSkillManifestStore",
    "InstalledSkillRecord",
    "SkillCard",
    "SkillCatalog",
    "SkillDocument",
    "SkillInspectionResult",
    "SkillBundleInstallResult",
    "SkillInstallResult",
    "SkillUninstallResult",
    "SkillVerifyResult",
    "format_installed_skill_list",
    "format_skill_bundle_install_result",
    "format_skill_inspection",
    "format_skill_install_result",
    "format_skill_uninstall_result",
    "format_skill_verify_result",
    "inspect_skill_source",
    "install_skill_bundle",
    "install_skill_source",
    "load_skill_card",
    "load_skill_document",
    "uninstall_skill",
    "update_skill_source",
    "verify_skill_install",
]
