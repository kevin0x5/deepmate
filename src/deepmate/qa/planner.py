"""LLM-driven QA Audit planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from deepmate.domain import Message, MessageRole
from deepmate.providers import ModelConversationItem, ModelProvider, ModelRequest
from deepmate.qa.discovery import discover_project
from deepmate.qa.model import AuditCase, AuditPlan, ProjectProfile
from deepmate.qa.store import next_audit_id


QA_PLANNER_SYSTEM_PROMPT = """You are a senior QA architect agent.

Design a project-specific QA audit plan from the user's goal and the discovered
project profile. Do not emit a generic checklist. Infer realistic user journeys,
failure modes, automation opportunities, and evidence needs from the project
signals. Computer Use is available as a permissioned capability for real visual
and interaction testing, but it is only one possible tool.

Return only one valid JSON object with this schema:
{
  "scope": ["short audit scope item"],
  "risk_model": ["specific risk hypothesis"],
  "permissions": ["permission needed before running"],
  "cases": [
    {
      "case_id": "stable.slug.001",
      "title": "short title",
      "surface": "ui_surface|api_surface|command_surface|...",
      "risk_area": "short risk area",
      "priority": "high|medium|low",
      "persona": "target user",
      "scenario_brief": "specific scenario",
      "preconditions": ["optional"],
      "steps": ["concrete step or command"],
      "expected": ["observable expected behavior"],
      "runner": "shell|file|service|browser|computer|artifact|manual",
      "tools": ["shell", "browser", "computer_use"],
      "oracle": "exit_code|visual_interaction_review|evidence_review|...",
      "evidence_required": ["command_output", "screenshot", "log"],
      "cleanup": ["optional"],
      "blocked_if": ["optional"]
    }
  ]
}

Runner guidance:
- shell: existing local test/build/lint command that can run in a sandbox.
- file: inspect docs/config/source evidence.
- service: requires starting a local service or checking health.
- browser: web UI automation evidence.
- computer: real user visual/interaction testing through Computer Use.
- artifact: final report/decision synthesis.
- manual: cannot be automated safely yet.

Always include at least one artifact decision case. If the project exposes UI,
desktop, CLI, or TUI surfaces, include a Computer Use case focused on real user
experience. Keep the case set focused: usually 5-12 cases.
"""


def create_audit_plan(
    goal: str,
    *,
    workspace: str | Path,
    provider: ModelProvider,
    model: str,
    options: Mapping[str, object] | None = None,
) -> tuple[AuditPlan, tuple[AuditCase, ...], str]:
    """Create an LLM-generated QA audit plan and editable cases."""
    clean_goal = goal.strip() or "检查这个项目是否具备发布条件"
    project = discover_project(workspace)
    audit_id = next_audit_id(workspace, clean_goal)
    payload = _generate_plan_payload(
        clean_goal,
        project=project,
        provider=provider,
        model=model,
        options=options or {},
    )
    scope = _tuple_text(payload.get("scope")) or _scope_for(project, clean_goal)
    risks = _tuple_text(payload.get("risk_model")) or _risk_model(project, clean_goal)
    permissions = _tuple_text(payload.get("permissions")) or _permissions_for(project)
    plan = AuditPlan(
        audit_id=audit_id,
        goal=clean_goal,
        scope=scope,
        surfaces=project.surfaces,
        risk_model=risks,
        permissions=permissions,
        project=project,
    )
    cases = _cases_from_payload(payload.get("cases"))
    if not cases:
        raise ValueError("QA planner returned no valid test cases")
    cases = _ensure_artifact_case(cases)
    return plan, cases, render_plan_markdown(plan, cases)


def create_fallback_audit_plan(
    goal: str,
    *,
    workspace: str | Path,
) -> tuple[AuditPlan, tuple[AuditCase, ...], str]:
    """Create a deterministic fallback plan for tests and degraded repair paths."""
    clean_goal = goal.strip() or "检查这个项目是否具备发布条件"
    project = discover_project(workspace)
    audit_id = next_audit_id(workspace, clean_goal)
    plan = AuditPlan(
        audit_id=audit_id,
        goal=clean_goal,
        scope=_scope_for(project, clean_goal),
        surfaces=project.surfaces,
        risk_model=_risk_model(project, clean_goal),
        permissions=_permissions_for(project),
        project=project,
    )
    cases = _cases_for(project, clean_goal)
    return plan, cases, render_plan_markdown(plan, cases)


def _generate_plan_payload(
    goal: str,
    *,
    project: ProjectProfile,
    provider: ModelProvider,
    model: str,
    options: Mapping[str, object],
) -> Mapping[str, object]:
    request = ModelRequest(
        model=model,
        conversation=(
            ModelConversationItem.from_message(
                Message(role=MessageRole.SYSTEM, content=QA_PLANNER_SYSTEM_PROMPT)
            ),
            ModelConversationItem.from_message(
                Message(
                    role=MessageRole.USER,
                    content=json.dumps(
                        {
                            "goal": goal,
                            "project_profile": project.to_mapping(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
            ),
        ),
        options={**dict(options), "temperature": options.get("temperature", 0)},
    )
    response = provider.complete(request)
    return _json_object(response.content)


def _json_object(text: str) -> Mapping[str, object]:
    clean = _strip_fenced_json(text)
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"QA planner response must be valid JSON: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("QA planner response must be a JSON object")
    return payload


def _strip_fenced_json(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    return clean


def _cases_from_payload(value: object) -> tuple[AuditCase, ...]:
    if not isinstance(value, list):
        return ()
    cases: list[AuditCase] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        case = AuditCase.from_mapping(item)
        if not case.case_id or not case.title or case.case_id in seen:
            continue
        cases.append(case)
        seen.add(case.case_id)
    return tuple(cases)


def _ensure_artifact_case(cases: tuple[AuditCase, ...]) -> tuple[AuditCase, ...]:
    if any(case.runner == "artifact" for case in cases):
        return cases
    return (
        *cases,
        AuditCase(
            case_id="release.decision.001",
            title="Release or handoff decision is backed by evidence",
            surface="artifact_surface",
            risk_area="decision_quality",
            priority="high",
            persona="project owner",
            scenario_brief="Synthesize case results and residual risks into a decision report.",
            steps=("Review all collected case evidence.",),
            expected=("The report supports a clear ship / do-not-ship / needs-review decision.",),
            runner="artifact",
            tools=("render_html_report", "review_artifact"),
            oracle="report_review",
            evidence_required=("report.md", "report.html"),
        ),
    )


def _tuple_text(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return tuple(result)


def render_plan_markdown(plan: AuditPlan, cases: tuple[AuditCase, ...]) -> str:
    """Render a user-editable audit plan."""
    lines = [
        f"# QA Audit Plan: {plan.project.project_name}",
        "",
        "## Goal",
        plan.goal,
        "",
        "## Project Profile",
        f"- Project kinds: {_join(plan.project.project_kinds)}",
        f"- Surfaces: {_join(plan.surfaces)}",
        f"- Package managers: {_join(plan.project.package_managers) or '-'}",
        f"- Existing test commands: {_join(plan.project.test_commands) or '-'}",
        f"- Run commands: {_join(plan.project.run_commands) or '-'}",
        "",
        "## Audit Scope",
        *[f"- {item}" for item in plan.scope],
        "",
        "## Risk Model",
        *[f"- {item}" for item in plan.risk_model],
        "",
        "## Permission Preview",
        *[f"- {item}" for item in plan.permissions],
        "",
        "## Test Outline",
    ]
    for case in cases:
        lines.append(
            f"- [{case.priority}] {case.case_id}: {case.title} "
            f"({case.surface}, runner={case.runner})"
        )
    lines.extend(
        [
            "",
            "## User Review",
            "- Confirm the goal and direction before running.",
            "- Edit `audit.cases.jsonl` if a scenario, expected outcome, or runner is wrong.",
            "- GUI and real user experience checks require explicit Computer Use permission.",
            "",
            "## Output",
            "- Evidence will be saved under `evidence/`.",
            "- The final decision report will be generated as `report.html`.",
            "",
        ]
    )
    return "\n".join(lines)


def _scope_for(project: ProjectProfile, goal: str) -> tuple[str, ...]:
    scope = [
        "Validate the project from real user-facing surfaces, not only source layout.",
        "Run available deterministic checks before experience review.",
        "Collect reproducible evidence for every finding.",
    ]
    if "docs_surface" in project.surfaces:
        scope.append("Check that documentation and first-use examples match actual behavior.")
    if _has_real_ux_surface(project):
        scope.append("Include visual and interaction experience checks when permission is granted.")
    if "api_surface" in project.surfaces:
        scope.append("Validate API availability and error recovery paths.")
    if "data_surface" in project.surfaces:
        scope.append("Validate data input/output contracts and malformed input handling.")
    if "发布" in goal or "release" in goal.lower():
        scope.append("Produce a release-readiness recommendation.")
    return tuple(scope)


def _risk_model(project: ProjectProfile, goal: str) -> tuple[str, ...]:
    risks = [
        "The primary user path may fail even when unit tests pass.",
        "Errors may lack enough recovery guidance for a new user.",
        "Generated artifacts or docs may drift from runtime behavior.",
    ]
    if project.test_commands:
        risks.append("Existing tests may not cover install, configuration, and end-to-end use.")
    else:
        risks.append("No obvious test command was detected; smoke checks and manual coverage matter more.")
    if "install_surface" in project.surfaces:
        risks.append("Install or dependency setup may block first use.")
    if "service_surface" in project.surfaces:
        risks.append("Local services may start but fail health, logs, ports, or shutdown behavior.")
    if _has_real_ux_surface(project):
        risks.append("Visual hierarchy, focus, shortcuts, and feedback may be unusable in real interaction.")
    if "integration_surface" in project.surfaces:
        risks.append("External tools, credentials, permissions, or agent actions may fail late.")
    return tuple(risks)


def _permissions_for(project: ProjectProfile) -> tuple[str, ...]:
    permissions = [
        "Read workspace files for discovery and evidence.",
        "Write QA artifacts under qa/audits/.",
    ]
    if project.test_commands:
        permissions.append("Run existing local test/build commands.")
    if project.run_commands or "service_surface" in project.surfaces:
        permissions.append("Optionally start local services and inspect logs/ports.")
    if "ui_surface" in project.surfaces:
        permissions.append("Optionally use browser automation for Web UI checks.")
    if _has_real_ux_surface(project):
        permissions.append("Optionally use Computer Use for real Web UI/TUI/CLI/desktop interaction.")
    permissions.append("High-risk external actions still require separate approval.")
    return tuple(permissions)


def _cases_for(project: ProjectProfile, goal: str) -> tuple[AuditCase, ...]:
    cases: list[AuditCase] = []
    add = cases.append
    if "install_surface" in project.surfaces:
        add(
            AuditCase(
                case_id="install.first_use.001",
                title="First-use install and setup path is understandable",
                surface="install_surface",
                risk_area="onboarding",
                priority="high",
                persona="new user",
                scenario_brief="A new user follows the project entrypoint and reaches the first useful action.",
                steps=("Inspect README or manifest setup instructions.", "Check whether required commands and configuration are discoverable."),
                expected=("A new user can identify install, configuration, and first run steps.",),
                runner="file",
                tools=("file_read",),
                oracle="evidence_review",
                evidence_required=("manifest_or_readme_excerpt",),
            )
        )
    if project.test_commands:
        for index, command in enumerate(project.test_commands[:3], start=1):
            add(
                AuditCase(
                    case_id=f"test.existing.{index:03d}",
                    title=f"Existing check passes: {command}",
                    surface="test_surface",
                    risk_area="regression",
                    priority="high",
                    persona="maintainer",
                    scenario_brief=f"Run the project's detected check command: {command}",
                    steps=(command,),
                    expected=("Command exits successfully or produces actionable diagnostics.",),
                    runner="shell",
                    tools=("shell",),
                    oracle="exit_code",
                    evidence_required=("command_output",),
                    cleanup=(),
                )
            )
    if "command_surface" in project.surfaces:
        add(
            AuditCase(
                case_id="cli.help.001",
                title="Command entrypoint exposes usable help and failure guidance",
                surface="command_surface",
                risk_area="usability",
                priority="medium",
                persona="new user",
                scenario_brief="Inspect CLI entrypoints and help paths for discoverability and recovery guidance.",
                steps=("Identify likely CLI entrypoints.", "Check help text or documented first command."),
                expected=("Help text explains what to do next and does not require internal knowledge.",),
                runner="file",
                tools=("file_read", "shell_optional"),
                oracle="evidence_review",
                evidence_required=("entrypoint_or_help_evidence",),
            )
        )
    if "api_surface" in project.surfaces:
        add(
            AuditCase(
                case_id="api.availability.001",
                title="API surface has a smokeable availability path",
                surface="api_surface",
                risk_area="availability",
                priority="high",
                persona="integrator",
                scenario_brief="Validate that the API can be started or inspected and has an obvious health/error path.",
                steps=("Inspect API spec or server entrypoint.", "Check health/error documentation or smoke route."),
                expected=("API consumers have a reliable availability check and understandable errors.",),
                runner="service",
                tools=("file_read", "shell_optional", "http_optional"),
                oracle="deterministic_or_blocked",
                evidence_required=("spec_or_health_evidence",),
            )
        )
    if "ui_surface" in project.surfaces:
        add(
            AuditCase(
                case_id="ui.primary_flow.001",
                title="Primary UI path can be validated with browser evidence",
                surface="ui_surface",
                risk_area="frontend",
                priority="high",
                persona="target user",
                scenario_brief="Open the primary UI route, observe content, and capture evidence for the main user path.",
                steps=("Start or identify the UI.", "Use browser automation when available.", "Capture DOM/screenshot evidence."),
                expected=("The primary screen loads, exposes the main action, and provides understandable states.",),
                runner="browser",
                tools=("browser", "shell_optional"),
                oracle="dom_or_visual_review",
                evidence_required=("browser_snapshot", "screenshot"),
            )
        )
    if "desktop_surface" in project.surfaces:
        add(
            AuditCase(
                case_id="ux.desktop.001",
                title="Desktop UI supports real first-use interaction",
                surface="desktop_surface",
                risk_area="real_user_experience",
                priority="high",
                persona="new user",
                scenario_brief="Use Computer Use to observe and interact with the desktop UI like a real user.",
                steps=("Launch or locate the desktop app.", "Observe the visible UI.", "Perform the primary action.", "Capture before/after evidence."),
                expected=("The UI is visually clear, interactive, and recoverable when optional dependencies are missing.",),
                runner="computer",
                tools=("computer_use",),
                oracle="visual_experience_review",
                evidence_required=("screenshot", "accessibility_snapshot", "interaction_log"),
                blocked_if=("Computer Use permission is not granted.",),
            )
        )
    if _has_real_ux_surface(project):
        add(
            AuditCase(
                case_id="ux.real_interaction.001",
                title="Real user interaction is visually clear and recoverable",
                surface="experience_surface",
                risk_area="real_user_experience",
                priority="high",
                persona="new user",
                scenario_brief=(
                    "Use Computer Use to validate the most important visible workflow "
                    "as a real user would experience it, including layout, focus, "
                    "feedback, shortcuts, and error recovery."
                ),
                steps=(
                    "Identify the primary user-facing entrypoint from the plan and project evidence.",
                    "Launch or open that surface only after the user grants Computer Use permission.",
                    "Observe the first screen before interacting.",
                    "Perform the main user action or first-use path.",
                    "Capture before/after screenshots and an interaction log.",
                ),
                expected=(
                    "The first screen communicates purpose and next action without hidden setup knowledge.",
                    "Keyboard, mouse, focus, and visible feedback behave predictably.",
                    "Failure states are readable and offer a recoverable next step.",
                ),
                runner="computer",
                tools=("computer_use", "browser_optional", "shell_optional"),
                oracle="visual_interaction_review",
                evidence_required=("screenshot", "interaction_log", "accessibility_snapshot"),
                blocked_if=("Computer Use permission is not granted.",),
            )
        )
    if "data_surface" in project.surfaces:
        add(
            AuditCase(
                case_id="data.contract.001",
                title="Data input and output contracts are explicit",
                surface="data_surface",
                risk_area="data_quality",
                priority="medium",
                persona="data user",
                scenario_brief="Inspect data samples, schemas, or pipelines for reproducible input/output expectations.",
                steps=("Identify data inputs and outputs.", "Check schema, sample, or transformation documentation."),
                expected=("Valid and invalid data behavior is documented or testable.",),
                runner="file",
                tools=("file_read",),
                oracle="evidence_review",
                evidence_required=("schema_or_sample_evidence",),
            )
        )
    if "docs_surface" in project.surfaces:
        add(
            AuditCase(
                case_id="docs.behavior_match.001",
                title="Documentation matches the actual first useful workflow",
                surface="docs_surface",
                risk_area="docs_drift",
                priority="medium",
                persona="new user",
                scenario_brief="Compare documented commands, examples, and outputs against detected project capabilities.",
                steps=("Inspect README examples.", "Cross-check commands against manifests and scripts."),
                expected=("Documentation points to runnable, current commands and clear next steps.",),
                runner="file",
                tools=("file_read",),
                oracle="evidence_review",
                evidence_required=("readme_excerpt", "manifest_excerpt"),
            )
        )
    add(
        AuditCase(
            case_id="release.decision.001",
            title="Release or handoff decision is backed by evidence",
            surface="artifact_surface",
            risk_area="decision_quality",
            priority="high",
            persona="project owner",
            scenario_brief="Summarize case results, blocked coverage, and highest-impact findings into a decision report.",
            steps=("Review all collected case evidence.", "Classify failures, warnings, blocked areas, and residual risks."),
            expected=("The report supports a clear ship / do-not-ship / needs-review decision.",),
            runner="artifact",
            tools=("render_html_report", "review_artifact"),
            oracle="report_review",
            evidence_required=("report.md", "report.html"),
        )
    )
    return tuple(cases)


def _join(values: tuple[str, ...]) -> str:
    return ", ".join(values)


def _has_real_ux_surface(project: ProjectProfile) -> bool:
    return bool(
        {
            "ui_surface",
            "desktop_surface",
            "command_surface",
        }.intersection(project.surfaces)
    )
