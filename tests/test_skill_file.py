from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepmate.skills.skill_file import read_skill_markdown


class SkillFileTests(unittest.TestCase):
    def test_reads_nested_frontmatter_and_block_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SKILL.md"
            path.write_text(
                "\n".join(
                    (
                        "---",
                        "name: demo",
                        "description: >",
                        "  Multi line",
                        "  folded text.",
                        "allowed-tools:",
                        "  - read",
                        "  - search",
                        "metadata:",
                        "  owner: team",
                        "  enabled: true",
                        "  setup: |",
                        "    run local checks",
                        "    before install",
                        "---",
                        "# Demo",
                        "",
                        "Use this skill for demos.",
                    )
                ),
                encoding="utf-8",
            )

            metadata, body = read_skill_markdown(path)

        self.assertEqual(metadata["name"], "demo")
        self.assertEqual(metadata["description"], "Multi line folded text.")
        self.assertEqual(metadata["allowed-tools"], ["read", "search"])
        self.assertEqual(
            metadata["metadata"],
            {
                "owner": "team",
                "enabled": True,
                "setup": "run local checks\nbefore install",
            },
        )
        self.assertEqual(body, "# Demo\n\nUse this skill for demos.")


if __name__ == "__main__":
    unittest.main()
