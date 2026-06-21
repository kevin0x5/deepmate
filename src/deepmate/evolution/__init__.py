"""Lightweight self-evolution primitives."""

from deepmate.evolution.behavior import (
    BEHAVIOR_FILE,
    BEHAVIOR_HEADING,
    BehaviorHintDocument,
    extract_behavior_hints,
    profile_behavior_path,
    read_behavior_hint_documents,
    render_collaboration_hints,
    replace_behavior_hints_section,
    workspace_behavior_path,
)
from deepmate.evolution.changes import (
    EvolutionChange,
    EvolutionChangeStore,
    EvolutionRollbackResult,
)
from deepmate.evolution.evidence_mining import (
    EvolutionEvidenceBatch,
    ToolFailureEvidence,
    UserCorrectionEvidence,
    WorkflowEvidence,
    collect_evidence_from_records,
    tool_failure_candidates,
    user_correction_candidates,
    workflow_candidates,
)
from deepmate.evolution.failure_patterns import (
    FailurePattern,
    FailurePatternGuard,
    FailurePatternMatch,
    FailurePatternStore,
    update_failure_patterns_from_evidence,
)
from deepmate.evolution.generated_skills import (
    GeneratedSkillApplyResult,
    GeneratedSkillDraft,
    apply_generated_skill_draft,
    apply_generated_skill_patch,
    archive_generated_skill,
    generated_skill_drafts_from_workflows,
)
from deepmate.evolution.maintenance import (
    EvolutionFitnessMetrics,
    EvolutionMaintenanceResult,
    EvolutionMaintenanceState,
    apply_behavior_hint_change,
    run_evolution_maintenance,
)

__all__ = [
    "BEHAVIOR_FILE",
    "BEHAVIOR_HEADING",
    "BehaviorHintDocument",
    "EvolutionChange",
    "EvolutionChangeStore",
    "EvolutionEvidenceBatch",
    "EvolutionFitnessMetrics",
    "EvolutionMaintenanceResult",
    "EvolutionMaintenanceState",
    "EvolutionRollbackResult",
    "FailurePattern",
    "FailurePatternGuard",
    "FailurePatternMatch",
    "FailurePatternStore",
    "GeneratedSkillApplyResult",
    "GeneratedSkillDraft",
    "ToolFailureEvidence",
    "UserCorrectionEvidence",
    "WorkflowEvidence",
    "apply_behavior_hint_change",
    "apply_generated_skill_draft",
    "apply_generated_skill_patch",
    "archive_generated_skill",
    "collect_evidence_from_records",
    "extract_behavior_hints",
    "generated_skill_drafts_from_workflows",
    "profile_behavior_path",
    "read_behavior_hint_documents",
    "render_collaboration_hints",
    "replace_behavior_hints_section",
    "run_evolution_maintenance",
    "tool_failure_candidates",
    "update_failure_patterns_from_evidence",
    "user_correction_candidates",
    "workspace_behavior_path",
    "workflow_candidates",
]
