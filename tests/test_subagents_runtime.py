from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from deepmate.app import AppSettings, ModelPurposeSettings
from deepmate.channels.cli import _format_trace_record, _subagent_executor
from deepmate.capabilities import combine_surfaces, from_native_tool_schemas, from_skill_cards
from deepmate.domain import Message, MessageRole, ProfileRef, RuntimeEvent
from deepmate.providers import (
    ModelConversationItem,
    ModelRequest,
    ModelResponse,
    ModelToolRequest,
    TokenUsage,
)
from deepmate.runtime import (
    AgentStepResult,
    HookAction,
    HookActionType,
    HookDefinition,
    HookEvent,
    HookLayer,
    HookRegistry,
    HookRuntimeContext,
    HookSignalStore,
    ToolAccessMode,
    UserTurnResult,
    run_user_turn,
    start_runtime_activation,
)
from deepmate.skills import SkillCard, load_skill_document
from deepmate.subagents import (
    READ_SUBAGENT_RESULT_TOOL_NAME,
    SubagentAssignment,
    SubagentOrchestrationPolicy,
    SubagentRunRequest,
    SubagentRunResult,
    SubagentRunStatus,
    SubagentRuntime,
    SubagentReviewStatus,
    SubagentWorkflowStatus,
    SubagentToolExecutor,
    SUBAGENT_WORKFLOW_TOOL_NAME,
    review_subagent_result,
    run_subagent_orchestration,
    subagent_tool_schema,
    run_subagent,
)
from deepmate.subagents.runtime import (
    DEFAULT_OUTPUT_CONTRACT,
    _AllowedMcpToolExecutor,
    _turn_result,
)
from deepmate.subagents.store import SubagentResultStore
from deepmate.tools import NativeToolRegistry, workspace_filesystem_tools
from deepmate.trace import TraceRecorder, trace_record_matches_session


class StubProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("stub provider received too many requests")
        return self.responses.pop(0)


class SubagentRuntimeTest(unittest.TestCase):
    def test_run_subagent_builds_child_prompt_and_filters_visible_tools(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            schemas = registry.schemas()
            surface = combine_surfaces(
                (
                    from_native_tool_schemas(schemas),
                    from_skill_cards(
                        (
                            SkillCard(
                                name="repo-review",
                                description="Review repository details.",
                                path=workspace / "skills" / "repo-review" / "SKILL.md",
                            ),
                            SkillCard(
                                name="private-process",
                                description="Private process details.",
                                path=workspace
                                / "skills"
                                / "private-process"
                                / "SKILL.md",
                            ),
                        )
                    ),
                )
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        content="child summary",
                        usage=TokenUsage(input_tokens=2, output_tokens=3),
                    )
                ]
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                capability_surface=surface,
                native_tools=registry,
                tool_schemas=schemas,
                selected_skill_documents=(
                    load_skill_document(workspace / "skills" / "repo-review"),
                ),
            )

            result = run_subagent(
                runtime,
                SubagentRunRequest(
                    goal="Inspect one file.",
                    input_context="Only inspect the README.",
                    output_contract="Return one sentence.",
                    acceptance_criteria=("Mention the README evidence.",),
                    allowed_tools=("read_text_file",),
                    max_steps=1,
                ),
            )

            self.assertEqual(result.status, SubagentRunStatus.COMPLETED)
            self.assertEqual(result.summary, "child summary")
            self.assertEqual(result.usage.total_tokens(), 5)
            request = provider.requests[0]
            self.assertEqual(request.model, "stub-model")
            self.assertEqual(
                tuple(schema["name"] for schema in request.tool_schemas),
                ("read_text_file",),
            )
            self.assertEqual(request.conversation[0].message.role, MessageRole.SYSTEM)
            child_prompt = request.conversation[-1].message.content
            self.assertIn("Goal:\nInspect one file.", child_prompt)
            self.assertIn("Input context:\nOnly inspect the README.", child_prompt)
            self.assertIn("Acceptance criteria:\n- Mention the README evidence.", child_prompt)
            self.assertIn("Output contract:\nReturn one sentence.", child_prompt)
            self.assertIn("Do not request recursive subagent delegation.", child_prompt)
            system_prompt = request.conversation[0].message.content
            self.assertNotIn("<selected_skills>", system_prompt)
            self.assertNotIn("repo-review", system_prompt)
            self.assertNotIn("private-process", system_prompt)
            self.assertIn("read_text_file", system_prompt)
            self.assertNotIn("list_directory", system_prompt)

    def test_run_subagent_injects_only_explicitly_allowed_selected_skill(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="child summary")])
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                selected_skill_documents=(
                    load_skill_document(workspace / "skills" / "repo-review"),
                    load_skill_document(workspace / "skills" / "private-process"),
                ),
            )

            result = run_subagent(
                runtime,
                SubagentRunRequest(
                    goal="Use review skill.",
                    allowed_tools=("repo-review",),
                    max_steps=1,
                ),
            )

            self.assertEqual(result.status, SubagentRunStatus.COMPLETED)
            system_prompt = provider.requests[0].conversation[0].message.content
            self.assertIn("<selected_skills>", system_prompt)
            self.assertIn("<name>repo-review</name>", system_prompt)
            self.assertNotIn("<name>private-process</name>", system_prompt)

    def test_run_subagent_uses_default_output_contract_when_missing(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="default contract used")])
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
            )

            result = run_subagent(
                runtime,
                SubagentRunRequest(
                    goal="Summarize a local observation.",
                    input_context="No tools needed.",
                    max_steps=1,
                ),
            )

            self.assertEqual(result.status, SubagentRunStatus.COMPLETED)
            child_prompt = provider.requests[0].conversation[-1].message.content
            self.assertIn("Output contract:\n" + DEFAULT_OUTPUT_CONTRACT, child_prompt)

    def test_run_subagent_executes_allowed_write_tool_and_collects_refs(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(
                workspace_filesystem_tools(workspace, include_write_tools=True)
            )
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="write_text_file",
                                id="call_1",
                                arguments={
                                    "path": "result.txt",
                                    "content": "hello",
                                    "overwrite": True,
                                },
                            ),
                        ),
                        usage=TokenUsage(input_tokens=5, output_tokens=1),
                    ),
                    ModelResponse(
                        content="wrote result.txt",
                        usage=TokenUsage(input_tokens=7, output_tokens=2),
                    ),
                ]
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
            )

            result = run_subagent(
                runtime,
                SubagentRunRequest(
                    goal="Write the result file.",
                    allowed_tools=("write_text_file",),
                    tool_access_mode=ToolAccessMode.WORKSPACE_WRITE,
                    max_steps=2,
                ),
            )

            self.assertEqual(result.status, SubagentRunStatus.COMPLETED)
            self.assertEqual(result.summary, "wrote result.txt")
            self.assertEqual(result.artifact_refs, ("result.txt",))
            self.assertIn("result.txt", result.evidence_refs)
            self.assertEqual(result.usage.total_tokens(), 15)
            self.assertEqual((workspace / "result.txt").read_text(encoding="utf-8"), "hello")

    def test_run_subagent_reports_max_steps_reached(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                id="call_1",
                                arguments={"path": "README.md"},
                            ),
                        )
                    )
                ]
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
            )

            result = run_subagent(
                runtime,
                SubagentRunRequest(
                    goal="Read once.",
                    allowed_tools=("read_text_file",),
                    max_steps=1,
                ),
            )

            self.assertEqual(result.status, SubagentRunStatus.MAX_STEPS_REACHED)
            self.assertEqual(result.error.code, "subagent_max_steps_reached")
            self.assertIn("README.md", result.evidence_refs)

    def test_subagent_turn_result_maps_interrupted_turn_to_cancelled(self) -> None:
        turn = UserTurnResult(
            steps=(
                AgentStepResult(
                    request=ModelRequest(
                        model="stub-model",
                        conversation=(
                            ModelConversationItem.from_message(
                                Message(role=MessageRole.USER, content="cancel")
                            ),
                        ),
                    ),
                    response=ModelResponse(content="interrupted"),
                    events=(
                        RuntimeEvent(
                            kind="turn_interrupted",
                            summary="Turn interrupted.",
                        ),
                    ),
                ),
            )
        )

        result = _turn_result(
            "run_cancelled",
            SubagentRunRequest(goal="Stop early.", max_steps=1),
            turn,
        )

        self.assertEqual(result.status, SubagentRunStatus.CANCELLED)

    def test_parent_agent_can_call_subagent_tool_and_replay_result(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="run_subagent",
                                id="call_parent",
                                arguments={
                                    "goal": "Inspect README only.",
                                    "input_context": "Use read_text_file if needed.",
                                    "allowed_tools": ["read_text_file", "run_subagent"],
                                    "max_steps": 1,
                                },
                            ),
                        )
                    ),
                    ModelResponse(
                        content="child inspected README",
                        usage=TokenUsage(input_tokens=4, output_tokens=2),
                    ),
                    ModelResponse(content="parent merged child result"),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                    native_tools=registry,
                    tool_schemas=registry.schemas(),
                )

            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                messages=(),
                model="stub-model",
                subagents=SubagentToolExecutor(
                    runtime_factory=runtime_factory,
                    default_allowed_tools=("read_text_file",),
                ),
                tool_schemas=(subagent_tool_schema(),),
                max_steps=2,
            )

            self.assertEqual(result.final_step().response.content, "parent merged child result")
            exchange = result.tool_exchanges[0]
            self.assertEqual(exchange.tool_results[0].name, "run_subagent")
            self.assertIn('"status":"completed"', exchange.tool_results[0].content)
            self.assertIn('"review":{"status":"accepted"', exchange.tool_results[0].content)
            self.assertIn("child inspected README", exchange.tool_results[0].content)
            child_request = provider.requests[1]
            self.assertEqual(
                tuple(schema["name"] for schema in child_request.tool_schemas),
                ("read_text_file",),
            )

    def test_parent_subagent_tool_returns_handle_and_can_read_details(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            store = SubagentResultStore.in_data_dir(workspace / "var", "session-a")
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                id="call_read",
                                arguments={"path": "README.md"},
                            ),
                        )
                    ),
                    ModelResponse(content="README inspected with evidence."),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                    native_tools=registry,
                    tool_schemas=registry.schemas(),
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                default_allowed_tools=("read_text_file",),
                result_store=store,
            )
            self.assertTrue(executor.has_tool(READ_SUBAGENT_RESULT_TOOL_NAME))
            self.assertIn(
                READ_SUBAGENT_RESULT_TOOL_NAME,
                tuple(schema["name"] for schema in executor.schemas()),
            )
            first = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_parent",
                    arguments={
                        "goal": "Inspect README only.",
                        "allowed_tools": ["read_text_file"],
                        "max_steps": 2,
                    },
                )
            )
            payload = json.loads(first.model_result.content)
            result_ref = payload["result_handle"]

            second = executor.execute(
                ModelToolRequest(
                    name=READ_SUBAGENT_RESULT_TOOL_NAME,
                    id="call_read_result",
                    arguments={"result_ref": result_ref},
                )
            )
            details = json.loads(second.model_result.content)
            result_path = (
                workspace
                / "var"
                / "subagents"
                / "session-a"
                / f"{result_ref}.json"
            )
            result_mode = result_path.stat().st_mode & 0o777

        self.assertFalse(first.model_result.is_error)
        self.assertEqual(details["result_ref"], result_ref)
        self.assertEqual(details["schema_version"], 2)
        self.assertEqual(details["result"]["summary"], "README inspected with evidence.")
        self.assertEqual(details["request"]["input_context_chars"], 0)
        self.assertIn("steps", details)
        self.assertEqual(details["steps"][0]["tool_results"][0]["name"], "read_text_file")
        self.assertEqual(result_mode, 0o600)

    def test_subagent_store_loads_legacy_record_without_schema_version(self) -> None:
        with _workspace() as workspace:
            store = SubagentResultStore.in_data_dir(workspace / "var", "session-a")
            record = store.save(
                request=SubagentRunRequest(goal="Inspect.", run_id="legacy"),
                result=SubagentRunResult(
                    run_id="legacy",
                    status=SubagentRunStatus.COMPLETED,
                    summary="saved",
                ),
                turn=None,
            )
            path = (
                workspace
                / "var"
                / "subagents"
                / "session-a"
                / f"{record.ref}.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.pop("schema_version")
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = store.load(record.ref)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.schema_version, 1)

    def test_subagent_runtime_removes_recursive_tools_even_when_explicitly_allowed(
        self,
    ) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            provider = StubProvider([ModelResponse(content="child summary")])
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                native_tools=registry,
                tool_schemas=(
                    *registry.schemas(),
                    subagent_tool_schema(),
                ),
            )

            result = run_subagent(
                runtime,
                SubagentRunRequest(
                    goal="Inspect README only.",
                    allowed_tools=("read_text_file", "run_subagent"),
                    max_steps=1,
                ),
            )

        self.assertEqual(result.status, SubagentRunStatus.COMPLETED)
        self.assertEqual(
            tuple(schema["name"] for schema in provider.requests[0].tool_schemas),
            ("read_text_file",),
        )

    def test_bound_subagent_executor_reads_current_session_result_store(self) -> None:
        with _workspace() as workspace:
            data_dir = workspace / "var"
            old_store = SubagentResultStore.in_data_dir(data_dir, "session-a")
            current_store = SubagentResultStore.in_data_dir(data_dir, "session-b")
            request = SubagentRunRequest(goal="Inspect result store.", run_id="shared")
            old_record = old_store.save(
                request=request,
                result=SubagentRunResult(
                    run_id="shared",
                    status=SubagentRunStatus.COMPLETED,
                    summary="old session result",
                ),
                turn=None,
            )
            current_record = current_store.save(
                request=request,
                result=SubagentRunResult(
                    run_id="shared",
                    status=SubagentRunStatus.COMPLETED,
                    summary="current session result",
                ),
                turn=None,
            )
            self.assertEqual(old_record.ref, current_record.ref)

            executor = SubagentToolExecutor(
                runtime_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("runtime factory should not be called")
                ),
                result_store=old_store,
            ).bind_runtime(
                capability_surface=None,
                native_tools=None,
                mcp_tools=None,
                tool_schemas=(),
                result_store=current_store,
            )
            result = executor.execute(
                ModelToolRequest(
                    name=READ_SUBAGENT_RESULT_TOOL_NAME,
                    id="call_read_result",
                    arguments={"result_ref": current_record.ref},
                )
            )
            details = json.loads(result.model_result.content)

        self.assertFalse(result.model_result.is_error)
        self.assertEqual(details["result"]["summary"], "current session result")
        self.assertNotIn("old session result", result.model_result.content)

    def test_subagent_executor_exposes_single_and_workflow_schemas(self) -> None:
        def runtime_factory() -> SubagentRuntime:
            raise AssertionError("runtime factory should not be called")

        executor = SubagentToolExecutor(
            runtime_factory=runtime_factory,
            workflow_policy=SubagentOrchestrationPolicy(max_child_steps=9),
        )

        self.assertEqual(executor.schema()["name"], "run_subagent")
        self.assertTrue(executor.has_tool("run_subagent"))
        self.assertTrue(executor.has_tool(SUBAGENT_WORKFLOW_TOOL_NAME))
        schemas = executor.schemas()
        self.assertEqual(
            tuple(schema["name"] for schema in schemas),
            ("run_subagent", SUBAGENT_WORKFLOW_TOOL_NAME),
        )
        workflow_schema = schemas[1]
        assignment_schema = workflow_schema["input_schema"]["properties"][
            "assignments"
        ]["items"]
        self.assertEqual(
            assignment_schema["properties"]["max_steps"]["maximum"],
            9,
        )

    def test_parent_agent_can_call_subagent_workflow_tool_and_replay_result(
        self,
    ) -> None:
        with _workspace() as workspace:
            store = SubagentResultStore.in_data_dir(workspace / "var", "session-a")
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name=SUBAGENT_WORKFLOW_TOOL_NAME,
                                id="call_workflow",
                                arguments={
                                    "plan_summary": "Inspect before implementing.",
                                    "assignments": [
                                        {
                                            "assignment_id": "inspect",
                                            "goal": "Inspect current design.",
                                            "max_steps": 1,
                                        },
                                        {
                                            "assignment_id": "implement",
                                            "goal": "Implement scoped change.",
                                            "depends_on": ["inspect"],
                                            "max_steps": 1,
                                        },
                                    ],
                                    "reflector": {
                                        "assignment_id": "reflect",
                                        "goal": "Check if result is ready.",
                                        "max_steps": 1,
                                    },
                                },
                            ),
                        )
                    ),
                    ModelResponse(
                        content="inspection complete",
                        usage=TokenUsage(input_tokens=2, output_tokens=3),
                    ),
                    ModelResponse(
                        content="implementation complete",
                        usage=TokenUsage(input_tokens=5, output_tokens=7),
                    ),
                    ModelResponse(
                        content="reflector pass",
                        usage=TokenUsage(input_tokens=11, output_tokens=13),
                    ),
                    ModelResponse(content="parent merged workflow result"),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                result_store=store,
            )
            result = run_user_turn(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                messages=(),
                model="stub-model",
                subagents=executor,
                tool_schemas=executor.schemas(),
                max_steps=2,
            )

            self.assertEqual(
                result.final_step().response.content,
                "parent merged workflow result",
            )
            exchange = result.tool_exchanges[0]
            tool_result = exchange.tool_results[0]
            self.assertEqual(tool_result.name, SUBAGENT_WORKFLOW_TOOL_NAME)
            self.assertFalse(tool_result.is_error)
            self.assertIn("status=completed", tool_result.refs)
            self.assertIn("child_runs=3", tool_result.refs)
            self.assertIn("accepted_results=2", tool_result.refs)
            payload = json.loads(tool_result.content)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(len(payload["accepted_results"]), 2)
            self.assertIn("result_handle", payload["accepted_results"][0])
            self.assertIn("result_handle", payload["assignment_runs"][0])
            self.assertEqual(payload["reflector_summary"], "reflector pass")
            self.assertEqual(payload["usage"]["input_tokens"], 18)
            dependency_prompt = provider.requests[2].conversation[-1].message.content
            self.assertIn("- inspect: inspection complete", dependency_prompt)
            reflector_prompt = provider.requests[3].conversation[-1].message.content
            self.assertIn(
                "Plan summary:\nInspect before implementing.",
                reflector_prompt,
            )
            self.assertIn("implementation complete", reflector_prompt)

    def test_subagent_workflow_tool_marks_blocked_result_as_error(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="changed the file")])

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                parent_tool_access_mode=ToolAccessMode.WORKSPACE_WRITE,
                workflow_policy=SubagentOrchestrationPolicy(max_child_runs=2),
            )
            result = executor.execute(
                ModelToolRequest(
                    name=SUBAGENT_WORKFLOW_TOOL_NAME,
                    id="call_1",
                    arguments={
                        "assignments": [
                            {
                                "assignment_id": "write",
                                "goal": "Change a file.",
                                "output_contract": (
                                    "Return changed file artifact refs."
                                ),
                                "tool_access_mode": "workspace_write",
                                "max_steps": 1,
                            }
                        ]
                    },
                )
            )

            self.assertTrue(result.model_result.is_error)
            self.assertEqual(result.error.code, "subagent_workflow_blocked")
            payload = json.loads(result.model_result.content)
            self.assertEqual(payload["status"], "blocked")
            self.assertIn("artifact_refs_for_write", result.model_result.content)

    def test_subagent_workflow_end_hook_records_signal(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="inspection complete")])
            store = HookSignalStore(workspace / "var" / "hooks" / "signals.jsonl")

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                hook_context=HookRuntimeContext.from_registry(
                    HookRegistry.from_hooks(
                        (
                            HookDefinition(
                                hook_id="workflow-signal",
                                event_name=HookEvent.SUBAGENT_WORKFLOW_END,
                                layer=HookLayer.SESSION,
                                actions=(
                                    HookAction(
                                        HookActionType.RECORD_EVOLUTION_SIGNAL,
                                        {
                                            "signal_kind": "workflow_result",
                                            "summary": "Workflow completed.",
                                        },
                                    ),
                                ),
                            ),
                        )
                    ),
                    signal_store=store,
                ),
            )

            result = executor.execute(
                ModelToolRequest(
                    name=SUBAGENT_WORKFLOW_TOOL_NAME,
                    id="call_1",
                    arguments={
                        "assignments": [
                            {
                                "assignment_id": "inspect",
                                "goal": "Inspect README.",
                                "max_steps": 1,
                            }
                        ]
                    },
                )
            )
            signals = store.load_recent()

        self.assertIsNone(result.error)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].hook_id, "workflow-signal")
        self.assertEqual(signals[0].event_name, HookEvent.SUBAGENT_WORKFLOW_END.value)
        self.assertTrue(
            any(event.kind == "subagent_workflow_hook_observed" for event in result.events)
        )

    def test_cli_subagent_executor_uses_subagent_worker_model_purpose(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="child summary")])
            settings = AppSettings(
                workspace=workspace,
                data_dir=workspace / "var",
                active_profile="default",
                trace_sink=workspace / "var" / "traces" / "trace.jsonl",
                default_provider="deepseek",
                model_purposes={
                    "subagent_worker": ModelPurposeSettings(
                        model="subagent-model",
                        thinking="disabled",
                        temperature=0,
                        max_tokens=1200,
                    )
                },
            )
            executor = _subagent_executor(
                provider=provider,
                settings=settings,
                profile=_profile(workspace),
                model="parent-model",
                capability_surface=None,
                native_tools=None,
                mcp_executor=None,
                tool_schemas=(),
                selected_skills=(),
                activation=None,
                provider_retry_policy=None,
                options={"thinking": {"type": "enabled"}, "max_tokens": 2048},
                trace_recorder=None,
                tool_access_mode=ToolAccessMode.READ_ONLY,
            )

            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Inspect README.",
                        "max_steps": 1,
                    },
                )
            )

            self.assertFalse(result.model_result.is_error)
            child_request = provider.requests[0]
            self.assertEqual(child_request.model, "subagent-model")
            self.assertEqual(
                child_request.options,
                {
                    "thinking": {"type": "disabled"},
                    "temperature": 0,
                    "max_tokens": 1200,
                },
            )

    def test_cli_subagent_executor_falls_back_when_purpose_missing(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="child summary")])
            settings = AppSettings(
                workspace=workspace,
                data_dir=workspace / "var",
                active_profile="default",
                trace_sink=workspace / "var" / "traces" / "trace.jsonl",
                default_provider="deepseek",
            )
            executor = _subagent_executor(
                provider=provider,
                settings=settings,
                profile=_profile(workspace),
                model="parent-model",
                capability_surface=None,
                native_tools=None,
                mcp_executor=None,
                tool_schemas=(),
                selected_skills=(),
                activation=None,
                provider_retry_policy=None,
                options={"thinking": {"type": "enabled"}, "max_tokens": 512},
                trace_recorder=None,
                tool_access_mode=ToolAccessMode.READ_ONLY,
            )

            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Inspect README.",
                        "max_steps": 1,
                    },
                )
            )

            self.assertFalse(result.model_result.is_error)
            child_request = provider.requests[0]
            self.assertEqual(child_request.model, "parent-model")
            self.assertEqual(
                child_request.options,
                {"thinking": {"type": "enabled"}, "max_tokens": 512},
            )

    def test_cli_subagent_executor_uses_local_parent_model_in_local_mode(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="child summary")])
            settings = AppSettings(
                workspace=workspace,
                data_dir=workspace / "var",
                active_profile="default",
                trace_sink=workspace / "var" / "traces" / "trace.jsonl",
                default_provider="local",
                model_purposes={
                    "subagent_worker": ModelPurposeSettings(
                        model="deepseek-v4-flash",
                        thinking="disabled",
                        max_tokens=1200,
                    )
                },
            )
            executor = _subagent_executor(
                provider=provider,
                settings=settings,
                profile=_profile(workspace),
                model="qwen3:4b",
                capability_surface=None,
                native_tools=None,
                mcp_executor=None,
                tool_schemas=(),
                selected_skills=(),
                activation=None,
                provider_retry_policy=None,
                options={"thinking": {"type": "enabled"}, "max_tokens": 512},
                trace_recorder=None,
                tool_access_mode=ToolAccessMode.READ_ONLY,
            )

            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Inspect README.",
                        "max_steps": 1,
                    },
                )
            )

            self.assertFalse(result.model_result.is_error)
            child_request = provider.requests[0]
            self.assertEqual(child_request.model, "qwen3:4b")
            self.assertEqual(child_request.options, {"max_tokens": 512})

    def test_subagent_executor_rejects_unexpected_tool_name(self) -> None:
        def runtime_factory() -> SubagentRuntime:
            raise AssertionError("runtime factory should not be called")

        executor = SubagentToolExecutor(runtime_factory=runtime_factory)
        result = executor.execute(
            ModelToolRequest(
                name="read_text_file",
                id="call_1",
                arguments={"goal": "Inspect README."},
            )
        )

        self.assertTrue(result.model_result.is_error)
        self.assertEqual(result.error.code, "subagent_tool_not_allowed")
        self.assertIn("run_subagent", result.model_result.content)

    def test_allowed_mcp_executor_returns_error_result_for_empty_tool_name(self) -> None:
        class FakeMcpExecutor:
            def has_tool(self, _name):
                return False

            def execute(self, _request):
                raise AssertionError("wrapped executor should not be called")

        executor = _AllowedMcpToolExecutor(FakeMcpExecutor(), {"server.tool"})  # type: ignore[arg-type]

        result = executor.execute(ModelToolRequest(name="", id="call_1"))

        self.assertIsNotNone(result.model_result)
        self.assertTrue(result.model_result.is_error)
        self.assertEqual(result.model_result.request_id, "call_1")
        self.assertEqual(result.model_result.name, "mcp_tool")
        self.assertEqual(result.error.code, "mcp_tool_not_allowed")

    def test_allowed_mcp_executor_exposes_schema_and_close_for_allowed_tools(self) -> None:
        class FakeMcpExecutor:
            def __init__(self) -> None:
                self.closed = False

            def has_tool(self, name):
                return name == "server.tool"

            def tool_schema(self, name):
                if name == "server.tool":
                    return {"name": name, "description": "Allowed tool"}
                return None

            def execute(self, _request):
                raise AssertionError("execute should not be called")

            def close(self):
                self.closed = True

        wrapped = FakeMcpExecutor()
        executor = _AllowedMcpToolExecutor(wrapped, {"server.tool"})  # type: ignore[arg-type]

        self.assertEqual(
            executor.tool_schema("server.tool"),
            {"name": "server.tool", "description": "Allowed tool"},
        )
        self.assertIsNone(executor.tool_schema("server.other"))
        executor.close()
        self.assertTrue(wrapped.closed)

    def test_subagent_result_review_detects_missing_evidence(self) -> None:
        review = review_subagent_result(
            SubagentRunRequest(
                goal="Inspect README.",
                output_contract="Return findings with evidence refs.",
            ),
            SubagentRunResult(
                run_id="run_1",
                status=SubagentRunStatus.COMPLETED,
                summary="README is present.",
            ),
        )

        self.assertEqual(review.status, SubagentReviewStatus.INCOMPLETE)
        self.assertEqual(review.missing, ("evidence_refs",))
        self.assertTrue(review.retryable)
        self.assertIn("evidence_refs", review.retry_instruction)

    def test_subagent_result_review_does_not_match_ref_inside_prefer(self) -> None:
        review = review_subagent_result(
            SubagentRunRequest(
                goal="Inspect README.",
                output_contract="Return concise findings and prefer direct language.",
            ),
            SubagentRunResult(
                run_id="run_1",
                status=SubagentRunStatus.COMPLETED,
                summary="README is present.",
            ),
        )

        self.assertEqual(review.status, SubagentReviewStatus.ACCEPTED)

    def test_parent_subagent_tool_marks_incomplete_result_as_error(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider(
                [
                    ModelResponse(content="summary only"),
                    ModelResponse(content="still summary only"),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                )

            executor = SubagentToolExecutor(runtime_factory=runtime_factory)
            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Inspect README.",
                        "output_contract": "Return findings with evidence refs.",
                        "allowed_tools": [],
                        "max_steps": 1,
                    },
                )
            )

            self.assertTrue(result.model_result.is_error)
            self.assertEqual(result.error.code, "subagent_result_incomplete")
            self.assertIn('"status":"incomplete"', result.model_result.content)
            self.assertIn('"missing":["evidence_refs"]', result.model_result.content)
            self.assertNotIn('"retry":{"attempts":2', result.model_result.content)
            self.assertEqual(result.events[-1].kind, "subagent_result_reviewed")
            self.assertEqual(len(provider.requests), 1)

    def test_subagent_tool_respects_explicit_empty_allowed_tools(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            provider = StubProvider([ModelResponse(content="no tools used")])

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                    native_tools=registry,
                    tool_schemas=registry.schemas(),
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                default_allowed_tools=("read_text_file",),
            )
            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Answer from supplied context only.",
                        "allowed_tools": [],
                        "max_steps": 1,
                    },
                )
            )

            self.assertFalse(result.model_result.is_error)
            self.assertEqual(provider.requests[0].tool_schemas, ())

    def test_subagent_filters_provider_shaped_tool_schemas_by_function_name(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="child summary")])
            schema = {
                "type": "function",
                "function": {
                    "name": "read_text_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object"},
                },
            }
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                tool_schemas=(schema,),
            )

            result = run_subagent(
                runtime,
                SubagentRunRequest(
                    goal="Inspect README.",
                    allowed_tools=("read_text_file",),
                    max_steps=1,
                ),
            )

            self.assertEqual(result.status, SubagentRunStatus.COMPLETED)
            self.assertEqual(provider.requests[0].tool_schemas, (schema,))

    def test_parent_subagent_tool_retries_once_when_result_becomes_mergeable(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            store = SubagentResultStore.in_data_dir(workspace / "var", "session-a")
            provider = StubProvider(
                [
                    ModelResponse(content="summary only"),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                id="call_read",
                                arguments={"path": "README.md"},
                            ),
                        )
                    ),
                    ModelResponse(content="summary with evidence"),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                    native_tools=registry,
                    tool_schemas=registry.schemas(),
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                default_allowed_tools=("read_text_file",),
                max_retries=1,
                result_store=store,
            )
            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Inspect README.",
                        "output_contract": "Return findings with evidence refs.",
                        "max_steps": 2,
                    },
                )
            )

            self.assertIsNone(result.error)
            self.assertFalse(result.model_result.is_error)
            self.assertIn('"review":{"status":"accepted"', result.model_result.content)
            self.assertIn('"retry":{"attempts":2', result.model_result.content)
            self.assertEqual(len(provider.requests), 3)
            payload = json.loads(result.model_result.content)
            self.assertTrue(payload["run_id"].endswith("-retry1"))
            self.assertTrue(payload["result_handle"].endswith("-retry1"))
            first_ref = payload["result_handle"].removesuffix("-retry1")
            first_record = store.load(first_ref)
            retry_record = store.load(payload["result_handle"])
            self.assertIsNotNone(first_record)
            self.assertIsNotNone(retry_record)
            self.assertEqual(
                retry_record.result["run_id"],
                f"{first_record.result['run_id']}-retry1",
            )

    def test_parent_subagent_tool_retries_after_max_steps_reached(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            provider = StubProvider(
                [
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                id="call_read",
                                arguments={"path": "README.md"},
                            ),
                        )
                    ),
                    ModelResponse(content="retry summary\n\nevidence_refs: README.md"),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                    native_tools=registry,
                    tool_schemas=registry.schemas(),
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                default_allowed_tools=("read_text_file",),
                max_retries=1,
            )
            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Inspect README.",
                        "allowed_tools": ["read_text_file"],
                        "max_steps": 1,
                    },
                )
            )

            self.assertIsNone(result.error)
            self.assertFalse(result.model_result.is_error)
            self.assertIn('"retry":{"attempts":2', result.model_result.content)
            self.assertEqual(len(provider.requests), 2)

    def test_read_subagent_result_returns_not_found_for_corrupt_record(self) -> None:
        with _workspace() as workspace:
            store = SubagentResultStore.in_data_dir(workspace / "var", "session-a")
            store_dir = store._session_dir
            store_dir.mkdir(parents=True, exist_ok=True)
            (store_dir / "subagent-result-0001.json").write_text("{bad json", encoding="utf-8")

            executor = SubagentToolExecutor(
                runtime_factory=lambda: SubagentRuntime(),
                result_store=store,
            )
            result = executor.execute(
                ModelToolRequest(
                    name="read_subagent_result",
                    id="call_1",
                    arguments={"result_ref": "subagent-result-0001"},
                )
            )

            self.assertEqual(result.error.code if result.error else "", "subagent_result_not_found")

    def test_subagent_result_store_failure_does_not_retry_completed_run(self) -> None:
        class FailingStore:
            def save(self, **kwargs):
                raise OSError("disk full")

        with _workspace() as workspace:
            provider = StubProvider(
                [
                    ModelResponse(content="summary only"),
                    ModelResponse(content="should not retry"),
                ]
            )

            def runtime_factory() -> SubagentRuntime:
                return SubagentRuntime(
                    provider=provider,
                    workspace=workspace,
                    profile=_profile(workspace),
                    model="stub-model",
                )

            executor = SubagentToolExecutor(
                runtime_factory=runtime_factory,
                max_retries=1,
                result_store=FailingStore(),
            )
            result = executor.execute(
                ModelToolRequest(
                    name="run_subagent",
                    id="call_1",
                    arguments={
                        "goal": "Inspect README.",
                        "output_contract": "Return findings with evidence refs.",
                        "allowed_tools": [],
                        "max_steps": 1,
                    },
                )
            )

        self.assertTrue(result.model_result.is_error)
        self.assertEqual(result.error.code, "subagent_result_unpersisted")
        self.assertIn("disk full", result.model_result.content)
        self.assertEqual(len(provider.requests), 1)

    def test_subagent_orchestration_runs_dependencies_and_reflector(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider(
                [
                    ModelResponse(
                        content="inspection complete",
                        usage=TokenUsage(input_tokens=2, output_tokens=3),
                    ),
                    ModelResponse(
                        content="implementation complete",
                        usage=TokenUsage(input_tokens=5, output_tokens=7),
                    ),
                    ModelResponse(
                        content="reflector pass",
                        usage=TokenUsage(input_tokens=11, output_tokens=13),
                    ),
                ]
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
            )

            result = run_subagent_orchestration(
                runtime,
                (
                    SubagentAssignment(
                        assignment_id="inspect",
                        goal="Inspect the current design.",
                        max_steps=1,
                    ),
                    SubagentAssignment(
                        assignment_id="implement",
                        goal="Implement the scoped change.",
                        depends_on=("inspect",),
                        max_steps=1,
                    ),
                ),
                plan_summary="Inspect before implementing.",
                reflector_assignment=SubagentAssignment(
                    assignment_id="reflect",
                    goal="Check whether the result is ready to deliver.",
                    max_steps=1,
                ),
                policy=SubagentOrchestrationPolicy(max_child_runs=3),
            )

            self.assertEqual(result.status, SubagentWorkflowStatus.COMPLETED)
            self.assertTrue(result.is_success())
            self.assertEqual(len(result.accepted_results), 2)
            self.assertEqual(result.reflector_summary, "reflector pass")
            self.assertEqual(result.usage.total_tokens(), 41)
            dependency_prompt = provider.requests[1].conversation[-1].message.content
            self.assertIn("- inspect: inspection complete", dependency_prompt)
            reflector_prompt = provider.requests[2].conversation[-1].message.content
            self.assertIn("Plan summary:\nInspect before implementing.", reflector_prompt)
            self.assertIn("inspection complete", reflector_prompt)
            self.assertIn("implementation complete", reflector_prompt)

    def test_subagent_orchestration_revises_once_when_evidence_missing(self) -> None:
        with _workspace() as workspace:
            registry = NativeToolRegistry(workspace_filesystem_tools(workspace))
            provider = StubProvider(
                [
                    ModelResponse(content="summary only"),
                    ModelResponse(
                        tool_requests=(
                            ModelToolRequest(
                                name="read_text_file",
                                id="call_read",
                                arguments={"path": "README.md"},
                            ),
                        )
                    ),
                    ModelResponse(content="summary with evidence"),
                ]
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                native_tools=registry,
                tool_schemas=registry.schemas(),
            )

            result = run_subagent_orchestration(
                runtime,
                (
                    SubagentAssignment(
                        assignment_id="inspect",
                        goal="Inspect README.",
                        output_contract="Return findings with evidence refs.",
                        allowed_tools=("read_text_file",),
                        max_steps=1,
                    ),
                ),
                policy=SubagentOrchestrationPolicy(max_child_runs=2),
            )

            self.assertEqual(result.status, SubagentWorkflowStatus.REVISED)
            self.assertEqual(result.revised_assignment_ids, ("inspect",))
            self.assertEqual(len(result.assignment_runs), 2)
            self.assertEqual(result.assignment_runs[1].request.max_steps, 3)
            self.assertIn("README.md", result.evidence_refs)
            retry_prompt = provider.requests[1].conversation[-1].message.content
            self.assertIn("Previous attempt was not mergeable.", retry_prompt)
            self.assertIn("evidence_refs", retry_prompt)

    def test_subagent_orchestration_blocks_workspace_write_without_artifact_refs(
        self,
    ) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="changed the file")])
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
            )

            result = run_subagent_orchestration(
                runtime,
                (
                    SubagentAssignment(
                        assignment_id="write",
                        goal="Change a file.",
                        output_contract="Return changed file artifact refs.",
                        tool_access_mode=ToolAccessMode.WORKSPACE_WRITE,
                        max_steps=1,
                    ),
                ),
                policy=SubagentOrchestrationPolicy(max_child_runs=2),
            )

            self.assertEqual(result.status, SubagentWorkflowStatus.BLOCKED)
            self.assertFalse(result.is_success())
            self.assertEqual(result.accepted_results, ())
            self.assertIn("artifact_refs_for_write", result.non_accepted_reviews[0].missing)

    def test_subagent_orchestration_runs_reflector_for_blocked_results(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider(
                [
                    ModelResponse(content="changed the file"),
                    ModelResponse(content="reflector saw blocking gap"),
                ]
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
            )

            result = run_subagent_orchestration(
                runtime,
                (
                    SubagentAssignment(
                        assignment_id="write",
                        goal="Change a file.",
                        output_contract="Return changed file artifact refs.",
                        tool_access_mode=ToolAccessMode.WORKSPACE_WRITE,
                        max_steps=1,
                    ),
                ),
                reflector_assignment=SubagentAssignment(
                    assignment_id="reflect",
                    goal="Check blocking gaps.",
                    max_steps=1,
                ),
                policy=SubagentOrchestrationPolicy(max_child_runs=2),
            )

            self.assertEqual(result.status, SubagentWorkflowStatus.BLOCKED)
            self.assertEqual(result.reflector_summary, "reflector saw blocking gap")
            self.assertEqual(len(provider.requests), 2)
            reflector_prompt = provider.requests[1].conversation[-1].message.content
            self.assertIn("Blocking gaps:", reflector_prompt)
            self.assertIn("artifact_refs_for_write", reflector_prompt)

    def test_subagent_orchestration_limits_workspace_write_child_runs(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider(
                [
                    ModelResponse(
                        content=(
                            "changed one file\n"
                            "artifact_refs: file://src/one.py\n"
                            "validation: checked"
                        )
                    ),
                    ModelResponse(
                        content=(
                            "changed another file\n"
                            "artifact_refs: file://src/two.py\n"
                            "validation: checked"
                        )
                    ),
                ]
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
            )

            result = run_subagent_orchestration(
                runtime,
                (
                    SubagentAssignment(
                        assignment_id="write_one",
                        goal="Change one file.",
                        output_contract="Return artifact refs and validation.",
                        tool_access_mode=ToolAccessMode.WORKSPACE_WRITE,
                        max_steps=1,
                    ),
                    SubagentAssignment(
                        assignment_id="write_two",
                        goal="Change another file.",
                        output_contract="Return artifact refs and validation.",
                        tool_access_mode=ToolAccessMode.WORKSPACE_WRITE,
                        max_steps=1,
                    ),
                ),
                policy=SubagentOrchestrationPolicy(
                    max_child_runs=4,
                    max_workspace_write_child_runs=1,
                    enable_reflector=False,
                ),
            )

            self.assertEqual(len(provider.requests), 1)
            self.assertEqual(result.status, SubagentWorkflowStatus.BLOCKED)
            self.assertEqual(len(result.assignment_runs), 1)
            self.assertIn(
                "write_two: workspace_write child run budget exhausted",
                result.blocking_gaps,
            )

    def test_subagent_orchestration_trace_matches_parent_session(self) -> None:
        with _workspace() as workspace:
            provider = StubProvider([ModelResponse(content="inspection complete")])
            sink = _MemoryTraceSink()
            activation = start_runtime_activation(
                session_id="session_1",
                workspace=workspace,
                profile=_profile(workspace),
            )
            runtime = SubagentRuntime(
                provider=provider,
                workspace=workspace,
                profile=_profile(workspace),
                model="stub-model",
                activation=activation,
                trace_recorder=TraceRecorder(sink),
            )

            result = run_subagent_orchestration(
                runtime,
                (
                    SubagentAssignment(
                        assignment_id="inspect",
                        goal="Inspect current design.",
                        max_steps=1,
                    ),
                ),
                plan_summary="Trace the orchestration.",
                policy=SubagentOrchestrationPolicy(max_child_runs=2),
            )

            self.assertEqual(result.status, SubagentWorkflowStatus.COMPLETED)
            kinds = tuple(record["kind"] for record in sink.records)
            self.assertIn("subagent_orchestration_started", kinds)
            self.assertIn("subagent_assignment_started", kinds)
            self.assertIn("subagent_run_finished", kinds)
            self.assertIn("subagent_assignment_reviewed", kinds)
            self.assertIn("subagent_orchestration_finished", kinds)
            self.assertTrue(
                all(
                    trace_record_matches_session(record, "session_1")
                    for record in sink.records
                    if str(record["kind"]).startswith("subagent_")
                )
            )
            reviewed = next(
                record
                for record in sink.records
                if record["kind"] == "subagent_assignment_reviewed"
            )
            formatted = _format_trace_record(reviewed)
            self.assertIn("assignment=inspect", formatted)
            self.assertIn("review=accepted", formatted)

    def test_show_session_matches_and_formats_parent_subagent_trace(self) -> None:
        record = {
            "recorded_at": "2026-05-29T00:00:00+00:00",
            "kind": "subagent_run_finished",
            "summary": "Subagent run finished.",
            "refs": [
                "subagent_run_id=run_1",
                "parent_session_id=session_1",
                "parent_activation_id=activation_1",
                "status=completed",
                "max_steps=3",
                "tool_access_mode=read_only",
                "artifact_refs=0",
                "evidence_refs=2",
            ],
        }

        self.assertTrue(trace_record_matches_session(record, "session_1"))
        formatted = _format_trace_record(record)
        self.assertIn("subagent_run_finished", formatted)
        self.assertIn("run=run_1", formatted)
        self.assertIn("status=completed", formatted)
        self.assertIn("evidence=2", formatted)


def _profile(workspace: Path) -> ProfileRef:
    return ProfileRef(name="default", uri=str(workspace / "profiles" / "default"))


class _workspace:
    def __enter__(self) -> Path:
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        (root / "profiles" / "default").mkdir(parents=True)
        (root / "skills" / "repo-review").mkdir(parents=True)
        (root / "skills" / "private-process").mkdir(parents=True)
        (root / "AGENTS.md").write_text("Workspace rules.", encoding="utf-8")
        (root / "README.md").write_text("Read me.", encoding="utf-8")
        (root / "profiles" / "default" / "identity.md").write_text(
            "Identity.",
            encoding="utf-8",
        )
        (root / "profiles" / "default" / "soul.md").write_text(
            "Style.",
            encoding="utf-8",
        )
        (root / "skills" / "repo-review" / "SKILL.md").write_text(
            "---\nname: repo-review\ndescription: Review repository details.\n---\nBody.",
            encoding="utf-8",
        )
        (root / "skills" / "private-process" / "SKILL.md").write_text(
            "---\nname: private-process\ndescription: Private process details.\n---\nBody.",
            encoding="utf-8",
        )
        return root

    def __exit__(self, exc_type, exc, tb) -> None:
        self._temp.cleanup()


class _MemoryTraceSink:
    def __init__(self) -> None:
        self.records = []

    def write(self, event) -> None:
        self.records.append(event.to_record())


if __name__ == "__main__":
    unittest.main()
