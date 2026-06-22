from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from deepmate.channels.cli import main
from deepmate.pet.copy import fallback_pet_copy, generate_pet_copy
from deepmate.pet.electron_host import electron_pet_command, electron_pet_missing_message
from deepmate.pet.events import (
    PetVisualState,
    event_for_care_reminder,
    event_for_task_achievement,
    event_for_turn_finished,
    event_for_turn_progress,
    event_for_turn_started,
    event_for_turn_waiting,
)
from deepmate.pet.learning import (
    LearningCandidate,
    _learning_request,
    interest_tags_from_texts,
    rank_learning_candidates,
)
from deepmate.pet.pets import built_in_pet, built_in_pet_ids
from deepmate.pet.policy import PetDisplayPolicy
from deepmate.pet.service import PetBackendService
from deepmate.pet.state import PetStateStore, PetUserAction, default_pet_profile
from deepmate.providers import ModelResponse


class _CopyProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ModelResponse(content=self.content)


class PetCompanionTests(unittest.TestCase):
    def test_state_store_selects_profile_and_writes_current_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)

            profile = store.select_pet("cat")
            self.assertEqual(profile.pet_id, "cat-lazy")
            self.assertEqual(store.load_profile().species, "cat")

            event = event_for_turn_started(
                workspace=Path(tmp),
                session_id="sess_1",
                prompt="实现桌宠状态文件",
                title="桌宠",
            )
            record = store.save_current_state(event)

            self.assertEqual(record["kind"], "task.started")
            self.assertEqual(store.load_current_state()["session_id"], "sess_1")
            self.assertTrue(store.events_path.exists())
            self.assertIn("task.started", store.events_path.read_text(encoding="utf-8"))

    def test_bad_current_state_json_falls_back_to_empty_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            store.current_state_path.parent.mkdir(parents=True)
            store.current_state_path.write_text("{not-json", encoding="utf-8")

            self.assertEqual(store.load_current_state(), {})
            self.assertEqual(store.offline_state()["kind"], "current_work.idle")
            self.assertEqual(store.offline_state()["state"], PetVisualState.IDLE.value)
            self.assertNotIn("idle.", store.offline_state()["summary"])

    def test_action_outbox_tracks_pending_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            store.append_action(
                PetUserAction(
                    action="open_current_work",
                    created_at="2026-06-10T12:00:00+08:00",
                    payload={"session_id": "sess_1", "title": "Current work"},
                )
            )

            pending = store.pending_actions()

            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0][1].action, "open_current_work")
            store.mark_actions_processed(pending[0][0])
            self.assertEqual(store.pending_actions(), ())

    def test_action_append_does_not_reenter_jsonl_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)

            store.append_action(
                PetUserAction(
                    action="open_current_work",
                    created_at="2026-06-10T12:00:00+08:00",
                    payload={"session_id": "sess_1"},
                )
            )

            self.assertEqual(len(store.pending_actions()), 1)

    def test_default_pet_profile_accepts_species_and_preset_id(self) -> None:
        self.assertEqual(default_pet_profile("squirrel").pet_id, "squirrel-lively")
        self.assertEqual(default_pet_profile("penguin-naive").species, "penguin")
        self.assertEqual(default_pet_profile("unknown").pet_id, "dog-happy")

    def test_progress_and_waiting_events_have_user_visible_states(self) -> None:
        progress = event_for_turn_progress(
            workspace=".",
            session_id="sess_1",
            title="Work",
            summary="Tool output compacted.",
        )
        waiting = event_for_turn_waiting(
            workspace=".",
            session_id="sess_1",
            title="Work",
            summary="Approval required.",
        )

        self.assertEqual(progress.kind, "task.progress")
        self.assertEqual(progress.state.value, "reporting")
        self.assertEqual(waiting.kind, "task.waiting")
        self.assertEqual(waiting.state.value, "waiting")

    def test_builtin_pixel_pets_have_distinct_palettes_and_required_states(self) -> None:
        pet_ids = built_in_pet_ids()
        self.assertEqual(
            set(pet_ids),
            {"dog-happy", "cat-lazy", "squirrel-lively", "penguin-naive"},
        )
        palettes = [tuple(sorted(built_in_pet(pet_id).palette.items())) for pet_id in pet_ids]
        self.assertEqual(len(set(palettes)), 4)
        for pet_id in pet_ids:
            pet = built_in_pet(pet_id)
            for state in (
                "idle",
                "thinking",
                "working",
                "waiting",
                "reporting",
                "celebrate",
                "blocked",
                "resting",
                "offline",
            ):
                self.assertIn(state, pet.frames)
            idle = pet.frames["idle"][0]
            self.assertGreaterEqual(len(idle), 14)
            self.assertGreaterEqual(len(idle[0]), 18)
            for frames in pet.frames.values():
                for frame in frames:
                    self.assertEqual(len(frame), 16)
                    self.assertEqual({len(row) for row in frame}, {20})

    def test_pet_service_publishes_frontend_ui_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            store.save_profile(default_pet_profile("cat"))
            store.save_current_state(
                event_for_turn_finished(
                    workspace=tmp,
                    session_id="sess_1",
                    title="测试任务",
                    summary="任务完成",
                )
            )

            snapshot = PetBackendService(store, poll_seconds=0.2).tick()

            self.assertIsNotNone(snapshot)
            ui_state = store.load_ui_state()
            self.assertEqual(ui_state["state"], "celebrate")
            self.assertEqual(ui_state["profile"]["pet_id"], "cat-lazy")
            self.assertTrue(ui_state["bubble"]["show"])
            self.assertTrue(ui_state["bubble"]["hold"])
            self.assertIn("Open current work", ui_state["actions"][0]["label"])

    def test_copy_cache_update_moves_key_before_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            store.save_copy_cache(
                {
                    f"k{index}": {
                        "text": str(index),
                        "source": "fallback",
                        "cached_at": "2026-06-20T00:00:00+00:00",
                    }
                    for index in range(200)
                }
            )
            service = PetBackendService(store, poll_seconds=0.2)

            service._store_copy_cache("k0", "fresh", "fallback")
            service._store_copy_cache("k200", "new", "fallback")

            cache = store.load_copy_cache()
            self.assertIn("k0", cache)
            self.assertIn("k200", cache)
            self.assertNotIn("k1", cache)
            self.assertEqual(len(cache), 200)

    def test_copy_cache_updates_are_atomic_under_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)

            def write_key(index: int) -> None:
                store.update_copy_cache(
                    f"k{index}",
                    {
                        "text": str(index),
                        "source": "test",
                        "cached_at": "2026-06-20T00:00:00+00:00",
                    },
                    limit=200,
                )

            threads = [threading.Thread(target=write_key, args=(index,)) for index in range(40)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            cache = store.load_copy_cache()
            self.assertEqual(len(cache), 40)
            self.assertEqual(set(cache), {f"k{index}" for index in range(40)})

    def test_electron_pet_frontend_assets_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "pet_ui/package.json").exists())
        self.assertTrue((root / "pet_ui/electron/main.js").exists())
        self.assertTrue((root / "pet_ui/electron/preload.js").exists())
        self.assertTrue((root / "pet_ui/renderer/pet.html").exists())
        self.assertTrue((root / "pet_ui/renderer/bubble.html").exists())
        self.assertTrue((root / "pet_ui/renderer/hud.html").exists())

    def test_electron_pet_frontend_is_transparent_pixel_surface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        main_js = (root / "pet_ui/electron/main.js").read_text(encoding="utf-8")
        preload_js = (root / "pet_ui/electron/preload.js").read_text(encoding="utf-8")
        pet_html = (root / "pet_ui/renderer/pet.html").read_text(encoding="utf-8")
        pet_js = (root / "pet_ui/renderer/pet.js").read_text(encoding="utf-8")
        hud_html = (root / "pet_ui/renderer/hud.html").read_text(encoding="utf-8")
        hud_js = (root / "pet_ui/renderer/hud.js").read_text(encoding="utf-8")
        hud_css = (root / "pet_ui/renderer/styles/hud.css").read_text(encoding="utf-8")
        pet_css = (root / "pet_ui/renderer/styles/pet.css").read_text(encoding="utf-8")
        bubble_js = (root / "pet_ui/renderer/bubble.js").read_text(encoding="utf-8")
        bubble_css = (root / "pet_ui/renderer/styles/bubble.css").read_text(encoding="utf-8")

        self.assertIn("frame: false", main_js)
        self.assertIn("transparent: true", main_js)
        self.assertIn("backgroundColor: '#00000000'", main_js)
        self.assertIn("alwaysOnTop: true", main_js)
        self.assertIn("skipTaskbar: true", main_js)
        self.assertIn("setIgnoreMouseEvents", main_js)
        self.assertIn("bubbleWindow", main_js)
        self.assertIn("hudWindow", main_js)
        self.assertIn("pet-renderer-ready", main_js)
        self.assertIn("const HUD_HEIGHT = 390", main_js)
        self.assertIn("fs.renameSync(temporary, file)", main_js)
        self.assertIn("pet_learning_state.json", main_js)
        self.assertIn("recordFeedback(value)", main_js)
        self.assertIn("suppressed_until", main_js)
        self.assertIn("refs: Array.isArray(state.refs)", main_js)
        self.assertIn("hx = clamp(hx", main_js)
        self.assertIn("ready(surface)", preload_js)
        self.assertIn("sendFeedback(value)", preload_js)
        self.assertIn('<canvas id="sprite"', pet_html)
        self.assertIn("BASE_FRAMES", pet_js)
        self.assertIn("FRAME_CACHE", pet_js)
        self.assertIn("buildFrameCache()", pet_js)
        self.assertIn("stateFrames", pet_js)
        self.assertIn("image-rendering: pixelated", pet_css)
        self.assertIn("openCurrentWork()", pet_js)
        self.assertIn("background: transparent", pet_css)
        self.assertIn('id="feedback"', hud_html)
        self.assertIn('id="timeline"', hud_html)
        self.assertIn("shouldShowFeedback(state)", hud_js)
        self.assertIn("sendFeedback('less_often')", hud_js)
        self.assertIn("timelineItems(state)", hud_js)
        self.assertIn(".feedback", hud_css)
        self.assertIn(".timeline", hud_css)
        self.assertIn("pet_profile.json", main_js)
        self.assertIn("muted_until", main_js)
        self.assertIn("...backendProfile,\n    ...localProfile", main_js)
        self.assertIn(
            "muted_until: localProfile.muted_until || backendProfile.muted_until",
            main_js,
        )
        self.assertIn("profile,", main_js)
        self.assertIn("REACTIONS", pet_js)
        self.assertIn("cleanReaction", pet_js)
        self.assertIn("renderInlineMarkdown", bubble_js)
        self.assertIn("document.createElement('strong')", bubble_js)
        self.assertIn("max-height: 130px", bubble_css)
        self.assertIn("setTimedMuteHours(1)", main_js)
        self.assertIn("Quiet 1h", main_js)
        self.assertIn("Quiet 1h", hud_js)

    def test_electron_pet_command_reports_missing_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = electron_pet_command(Path(tmp) / "var")

        if command is None:
            self.assertIn("npm --prefix pet_ui install", electron_pet_missing_message())
            self.assertIn("DEEPMATE_PET_ELECTRON", electron_pet_missing_message())
            self.assertIn("optional", electron_pet_missing_message())
            self.assertIn("pet_ui/node_modules/", electron_pet_missing_message())
        else:
            self.assertIn("--data-dir", command)

    def test_pet_cli_missing_frontend_reports_optional_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stderr = io.StringIO()

            with (
                patch("deepmate.pet.electron_host.electron_pet_command", return_value=None),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                exit_code = main(("--workspace", str(workspace), "--pet"))

        self.assertEqual(exit_code, 2)
        message = stderr.getvalue()
        self.assertIn("Desktop pet is optional", message)
        self.assertIn("npm --prefix pet_ui install", message)
        self.assertIn("pet_ui/node_modules/", message)

    def test_electron_pet_command_accepts_binary_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "electron"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            with patch.dict("os.environ", {"DEEPMATE_PET_ELECTRON": str(binary)}):
                command = electron_pet_command(Path(tmp) / "var")

        self.assertIsNotNone(command)
        self.assertEqual(command[0], str(binary))
        self.assertIn("--data-dir", command)

    def test_copy_generation_uses_fallback_without_provider(self) -> None:
        profile = default_pet_profile("penguin")
        event = event_for_turn_finished(
            workspace=".",
            session_id="sess_1",
            title="测试",
            summary="任务完成",
        )

        fallback = fallback_pet_copy(event, profile, max_chars=32)
        result = generate_pet_copy(event, profile, provider=None, model="deepseek")

        self.assertEqual(result.source, "fallback")
        self.assertEqual(result.text, fallback_pet_copy(event, profile))
        self.assertLessEqual(len(fallback), 32)

    def test_copy_generation_uses_provider_when_enabled(self) -> None:
        provider = _CopyProvider("我看到任务已经完成啦。")
        profile = default_pet_profile("dog")
        event = event_for_turn_finished(
            workspace=".",
            session_id="sess_1",
            title="测试",
            summary="任务完成",
        )

        result = generate_pet_copy(event, profile, provider=provider, model="pet-model")

        self.assertEqual(result.source, "llm")
        self.assertEqual(result.text, "我看到任务已经完成啦。")
        self.assertEqual(len(provider.requests), 1)

    def test_task_achievement_event_gets_visible_pet_bubble(self) -> None:
        event = event_for_task_achievement(
            workspace=".",
            session_id="sess_1",
            title="阶段收口",
            summary="成果文件已保存",
            path="task/achievements/stage.md",
        )
        decision = PetDisplayPolicy().decide(event)
        text = fallback_pet_copy(event, default_pet_profile("squirrel"))

        self.assertEqual(event.kind, "task.achievement")
        self.assertIn("path=task/achievements/stage.md", event.refs)
        self.assertTrue(decision.show_bubble)
        self.assertTrue(decision.hold)
        self.assertIn("这个阶段记下来了", text)

    def test_care_reminder_event_gets_low_priority_pet_bubble(self) -> None:
        event = event_for_care_reminder(summary="Take a short break.")
        decision = PetDisplayPolicy().decide(event)
        text = fallback_pet_copy(event, default_pet_profile("penguin"))

        self.assertEqual(event.kind, "care.reminder")
        self.assertEqual(event.state, PetVisualState.RESTING)
        self.assertTrue(decision.show_bubble)
        self.assertFalse(decision.hold)
        self.assertIn("Maybe pause", text)

    def test_pet_service_respects_profile_muted_until(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            muted_until = (
                datetime.now(timezone.utc).astimezone() + timedelta(hours=1)
            ).isoformat(timespec="seconds")
            store.save_profile(
                replace(default_pet_profile("dog"), muted_until=muted_until)
            )
            store.save_current_state(
                event_for_turn_finished(
                    workspace=tmp,
                    session_id="sess_1",
                    title="测试任务",
                    summary="任务完成",
                )
            )

            snapshot = PetBackendService(store, poll_seconds=0.2).tick()

            self.assertIsNotNone(snapshot)
            ui_state = store.load_ui_state()
            self.assertTrue(ui_state["muted"])
            self.assertFalse(ui_state["bubble"]["show"])

    def test_pet_care_only_fires_for_real_active_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            service = PetBackendService(
                store,
                poll_seconds=0.2,
                care_initial_delay=0,
            )

            snapshot = service.tick()

            self.assertIsNotNone(snapshot)
            ui_state = store.load_ui_state()
            self.assertEqual(ui_state["kind"], "current_work.idle")
            self.assertFalse(ui_state["bubble"]["show"])

    def test_pet_care_fires_for_long_running_work_once_per_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            started = event_for_turn_progress(
                workspace=tmp,
                session_id="sess_1",
                title="Long task",
                summary="Still checking files.",
            )
            old_created_at = (
                datetime.now(timezone.utc).astimezone() - timedelta(hours=3)
            ).isoformat(timespec="seconds")
            store.save_current_state(replace(started, created_at=old_created_at))
            service = PetBackendService(
                store,
                poll_seconds=0.2,
                care_initial_delay=0,
            )

            first = service.tick()
            service._last_care_at = 0
            second = service.tick()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertEqual(first.record["kind"], "care.reminder")
            self.assertEqual(second.record["kind"], "task.progress")

    def test_pet_care_uses_continuous_work_start_across_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            old_created_at = (
                datetime.now(timezone.utc).astimezone() - timedelta(hours=3)
            ).isoformat(timespec="seconds")
            store.save_current_state(
                replace(
                    event_for_turn_started(
                        workspace=tmp,
                        session_id="sess_1",
                        prompt="Long task",
                        title="Long task",
                    ),
                    created_at=old_created_at,
                )
            )
            service = PetBackendService(
                store,
                poll_seconds=0.2,
                care_initial_delay=2 * 60 * 60,
            )

            first = service.tick()
            store.save_current_state(
                event_for_turn_progress(
                    workspace=tmp,
                    session_id="sess_1",
                    title="Long task",
                    summary="Fresh progress should not reset long-work care.",
                )
            )
            service._last_care_at = 0
            second = service.tick()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertEqual(first.record["kind"], "task.started")
            self.assertEqual(second.record["kind"], "care.reminder")

    def test_learning_worker_rechecks_mute_before_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            store.save_profile(replace(default_pet_profile("cat"), learning_mode="low"))
            store.save_learning_state({"sources": ["https://example.test/feed"]})
            store.save_pet_state({"muted": True})
            service = PetBackendService(store, poll_seconds=0.2)

            with patch(
                "deepmate.pet.service.fetch_learning_candidates",
                return_value=(
                    LearningCandidate(
                        title="MCP patterns",
                        url="https://example.test/mcp",
                        summary="Useful context protocol notes.",
                    ),
                ),
            ):
                service._learning_worker()

            learning_state = store.load_learning_state()
            self.assertNotIn("last_suggestion", learning_state)
            self.assertNotIn("shown_urls", learning_state)
            self.assertEqual(store.load_ui_state(), {})

    def test_learning_feedback_suppression_blocks_learning_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            store.save_profile(replace(default_pet_profile("cat"), learning_mode="low"))
            store.save_learning_state(
                {
                    "sources": ["https://example.test/feed"],
                    "suppressed_until": {
                        "learning_suggestion": (
                            datetime.now(timezone.utc).astimezone() + timedelta(hours=1)
                        ).isoformat(timespec="seconds")
                    },
                }
            )
            service = PetBackendService(store, poll_seconds=0.2)

            with patch("deepmate.pet.service.fetch_learning_candidates") as fetch:
                service._learning_worker()

            fetch.assert_not_called()
            self.assertEqual(store.load_ui_state(), {})

    def test_care_feedback_suppression_blocks_care_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            old_created_at = (
                datetime.now(timezone.utc).astimezone() - timedelta(hours=3)
            ).isoformat(timespec="seconds")
            store.save_current_state(
                replace(
                    event_for_turn_progress(
                        workspace=tmp,
                        session_id="sess_1",
                        title="Long task",
                        summary="Still checking files.",
                    ),
                    created_at=old_created_at,
                )
            )
            store.save_learning_state(
                {
                    "suppressed_until": {
                        "proactive_care": (
                            datetime.now(timezone.utc).astimezone() + timedelta(hours=1)
                        ).isoformat(timespec="seconds")
                    }
                }
            )
            service = PetBackendService(
                store,
                poll_seconds=0.2,
                care_initial_delay=0,
            )

            snapshot = service.tick()

            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.record["kind"], "task.progress")

    def test_care_emitted_keys_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PetStateStore.in_data_dir(tmp)
            old_created_at = (
                datetime.now(timezone.utc).astimezone() - timedelta(hours=3)
            ).isoformat(timespec="seconds")
            store.save_current_state(
                replace(
                    event_for_turn_progress(
                        workspace=tmp,
                        session_id="sess_1",
                        title="Long task",
                        summary="Still checking files.",
                    ),
                    created_at=old_created_at,
                )
            )
            service = PetBackendService(
                store,
                poll_seconds=0.2,
                care_initial_delay=0,
            )
            service._care_emitted_keys = {f"2026-06-{index:02d}:long_work" for index in range(30)}

            snapshot = service.tick()

            self.assertIsNotNone(snapshot)
            self.assertLessEqual(len(service._care_emitted_keys), 14)

    def test_learning_candidates_rank_by_interest_tags(self) -> None:
        candidates = (
            LearningCandidate(title="Agent browser research", url="https://example.test/a"),
            LearningCandidate(title="Cooking tips", url="https://example.test/b"),
            LearningCandidate(title="MCP server patterns", url="https://example.test/c"),
        )

        ranked = rank_learning_candidates(candidates, interest_tags=("mcp",))

        self.assertEqual(ranked[0].title, "MCP server patterns")
        self.assertEqual(
            interest_tags_from_texts(("MCP cost control for coding agent",)),
            ("ai-agent", "coding-agent", "mcp", "cost-control"),
        )

    def test_learning_request_serializes_context_as_json(self) -> None:
        request = _learning_request(
            (
                LearningCandidate(
                    title="MCP patterns",
                    url="https://example.test/mcp",
                    summary="Useful context protocol notes.",
                ),
            ),
            interest_tags=("mcp",),
            current_work_summary="Deepmate task mode",
            profile=default_pet_profile("dog"),
            model="pet-model",
        )

        user_message = request.conversation[1].message
        self.assertIsNotNone(user_message)
        payload = json.loads(user_message.content)
        self.assertEqual(payload["pet"], "dog")
        self.assertEqual(payload["interest_tags"], ["mcp"])
        self.assertEqual(payload["candidates"][0]["url"], "https://example.test/mcp")

    def test_learning_candidates_reject_local_sources(self) -> None:
        from deepmate.pet.learning import fetch_learning_candidates

        with self.assertRaisesRegex(ValueError, "local network URLs"):
            fetch_learning_candidates("http://localhost:8000/")

    def test_learning_candidates_reject_oversized_sources(self) -> None:
        from deepmate.pet.learning import MAX_LEARNING_SOURCE_BYTES, fetch_learning_candidates

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size=-1):
                return b"x" * (size + 1)

        with (
            patch("deepmate.pet.learning.validate_public_url", return_value=None),
            patch("deepmate.pet.learning.urlopen", return_value=Response()),
        ):
            with self.assertRaisesRegex(
                ValueError,
                f"learning source exceeds {MAX_LEARNING_SOURCE_BYTES} bytes",
            ):
                fetch_learning_candidates("https://example.test/feed")

    def test_learning_sources_must_be_explicitly_configured(self) -> None:
        from deepmate.pet.service import _learning_sources

        self.assertEqual(_learning_sources({}), ())
        self.assertEqual(
            _learning_sources({"sources": ["https://example.com/", ""]}),
            ("https://example.com/",),
        )

    def test_pet_only_cli_commands_do_not_require_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--pet-select",
                        "penguin",
                    )
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("desktop pet updated", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(("--workspace", str(workspace), "--pet-status"))

            self.assertEqual(exit_code, 0)
            self.assertIn("pet_id: penguin-naive", stdout.getvalue())
            self.assertIn("current_kind:", stdout.getvalue())

    def test_pet_settings_and_actions_cli_do_not_require_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    (
                        "--workspace",
                        str(workspace),
                        "--pet-select",
                        "cat",
                        "--pet-learning",
                        "low",
                        "--pet-bubble",
                        "frugal",
                        "--pet-name",
                        "Milo",
                        "--no-pet-proactive-care",
                    )
                )

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("pet_id: cat-lazy", output)
            self.assertIn("learning_mode: low", output)
            self.assertIn("learning_sources: (none configured)", output)
            self.assertIn("bubble_generation: frugal", output)
            self.assertIn("proactive_care: false", output)

            store = PetStateStore.in_data_dir(workspace / "var")
            self.assertNotIn("sources", store.load_learning_state())

            store.append_action(
                PetUserAction(
                    action="open_current_work",
                    created_at="2026-06-10T12:00:00+08:00",
                    payload={"session_id": "missing", "title": "Current work"},
                )
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(("--workspace", str(workspace), "--pet-actions"))

            self.assertEqual(exit_code, 0)
            self.assertIn("pending: 1", stdout.getvalue())
            self.assertIn("Current work", stdout.getvalue())

    def test_pet_open_action_is_consumed_when_interactive_starts_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config_dir = workspace / "config"
            config_dir.mkdir(parents=True)
            (config_dir / "deepmate.yaml").write_text(
                "provider:\n  default: stub\n",
                encoding="utf-8",
            )
            (config_dir / "providers.yaml").write_text(
                "\n".join(
                    (
                        "providers:",
                        "  stub:",
                        "    base_url: http://example.test",
                        "    default_model: stub-main",
                        "    api_key_env: DEEPMATE_TEST_MISSING_KEY",
                    )
                ),
                encoding="utf-8",
            )
            store = PetStateStore.in_data_dir(workspace / "var")
            store.append_action(
                PetUserAction(
                    action="open_current_work",
                    created_at="2026-06-10T12:00:00+08:00",
                    payload={"session_id": "sess_1", "title": "Current work"},
                )
            )

            captured = {}

            def fake_run_tui_mode(**kwargs):
                captured.update(kwargs)
                return 0

            stderr = io.StringIO()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("deepmate.channels.cli.run_tui_mode", fake_run_tui_mode),
                redirect_stderr(stderr),
            ):
                exit_code = main(("--workspace", str(workspace), "--interactive"))

            self.assertEqual(exit_code, 0)
            self.assertNotIn("DEEPMATE_TEST_MISSING_KEY", stderr.getvalue())
            self.assertFalse(captured["provider_api_key_available"])
            self.assertEqual(captured["provider_name"], "stub")
            self.assertEqual(captured["provider_api_key_env"], "DEEPMATE_TEST_MISSING_KEY")
            self.assertEqual(captured["remote_provider_name"], "stub")
            self.assertEqual(len(store.pending_actions()), 0)


if __name__ == "__main__":
    unittest.main()
