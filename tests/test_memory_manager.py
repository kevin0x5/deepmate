from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.memory import (
    MemoryPatch,
    MemoryPatchOperation,
    apply_memory_patch,
)


class MemoryManagerTests(unittest.TestCase):
    def test_apply_patch_preserves_headings_and_applies_exact_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)
            (profile_dir / "user.md").write_text(
                "# User\n\n- 用户偏好中文。\n",
                encoding="utf-8",
            )

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="replace",
                            target="user",
                            replace_ref="用户偏好中文。",
                            content="用户偏好直接、克制的中文回答。",
                        ),
                    )
                ),
            )

            self.assertTrue(result.changed())
            content = (profile_dir / "user.md").read_text(encoding="utf-8")
            self.assertIn("# User", content)
            self.assertIn("- 用户偏好直接、克制的中文回答。", content)
            self.assertNotIn("- 用户偏好中文。", content)

    def test_apply_patch_preserves_entries_after_empty_bullet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)
            (profile_dir / "user.md").write_text(
                "# User\n\n- \n- 用户偏好中文。\n",
                encoding="utf-8",
            )

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="replace",
                            target="user",
                            replace_ref="用户偏好中文。",
                            content="用户偏好中文直接回答。",
                        ),
                    )
                ),
            )

            self.assertTrue(result.changed())
            content = (profile_dir / "user.md").read_text(encoding="utf-8")
            self.assertIn("- \n", content)
            self.assertIn("- 用户偏好中文直接回答。", content)
            self.assertNotIn("- 用户偏好中文。\n", content)

    def test_apply_patch_skips_existing_duplicate_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)
            (profile_dir / "memory.md").write_text(
                "- 保持回答克制。\n",
                encoding="utf-8",
            )

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="write_memory",
                            content="保持回答克制。",
                        ),
                    )
                ),
            )

            self.assertFalse(result.changed())
            self.assertIn("duplicate_existing_content", result.skipped)
            content = (profile_dir / "memory.md").read_text(encoding="utf-8")
            self.assertEqual(content.count("- 保持回答克制。"), 1)

    def test_apply_patch_deduplicates_normalized_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)
            (profile_dir / "memory.md").write_text(
                "- 保持回答克制。\n",
                encoding="utf-8",
            )

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="write_memory",
                            content="保持 回答 克制",
                        ),
                    )
                ),
            )

            self.assertFalse(result.changed())
            self.assertIn("duplicate_existing_content", result.skipped)
            self.assertEqual(
                (profile_dir / "memory.md").read_text(encoding="utf-8"),
                "- 保持回答克制。\n",
            )

    def test_budget_blocked_patch_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)
            (profile_dir / "memory.md").write_text(
                "- Existing.\n",
                encoding="utf-8",
            )

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="write_memory",
                            content="This is a long memory item that exceeds budget.",
                        ),
                    )
                ),
                hot_profile_token_budget=1,
            )

            self.assertFalse(result.changed())
            self.assertTrue(result.budget_blocked)
            self.assertEqual(
                (profile_dir / "memory.md").read_text(encoding="utf-8"),
                "- Existing.\n",
            )

    def test_invalid_patch_schema_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(action="replace", target="user"),
                    )
                ),
            )

            self.assertFalse(result.changed())
            self.assertIn("missing_replace_ref", result.skipped)

    def test_patch_action_and_target_are_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="WRITE_MEMORY",
                            target="MEMORY",
                            content="保持输出克制。",
                        ),
                    )
                ),
            )

            self.assertTrue(result.changed())
            self.assertIn(
                "- 保持输出克制。",
                (profile_dir / "memory.md").read_text(encoding="utf-8"),
            )

    def test_write_project_memory_writes_project_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_profile = root / "global" / "profiles" / "default"
            project_profile = root / "workspace" / "profiles" / "default"
            global_profile.mkdir(parents=True)
            project_profile.mkdir(parents=True)
            (global_profile / "memory.md").write_text("- Global principle.\n", encoding="utf-8")
            (project_profile / "memory.md").write_text("- Project fact.\n", encoding="utf-8")

            result = apply_memory_patch(
                global_profile,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="write_project_memory",
                            content="This project uses pnpm.",
                        ),
                    )
                ),
                project_profile_dir=project_profile,
            )

            self.assertTrue(result.changed())
            self.assertEqual(
                (global_profile / "memory.md").read_text(encoding="utf-8"),
                "- Global principle.\n",
            )
            self.assertIn(
                "- This project uses pnpm.",
                (project_profile / "memory.md").read_text(encoding="utf-8"),
            )

    def test_malformed_confidence_does_not_block_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_dir = Path(temp_dir)

            result = apply_memory_patch(
                profile_dir,
                MemoryPatch(
                    operations=(
                        MemoryPatchOperation(
                            action="write_user",
                            content="用户偏好直接回答。",
                            confidence="not-a-number",  # type: ignore[arg-type]
                        ),
                    )
                ),
            )

            self.assertTrue(result.changed())
            self.assertIsNone(result.applied_operations[0].confidence)


if __name__ == "__main__":
    unittest.main()
