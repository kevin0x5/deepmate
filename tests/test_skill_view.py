from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepmate.capabilities import from_skill_cards
from deepmate.channels.skill_view import (
    discover_builtin_skill_cards,
    discover_skill_cards,
    discover_workspace_skill_cards,
    select_skill_documents,
    workspace_skill_catalog,
)
from deepmate.skills import SkillCatalog


class SkillViewTests(unittest.TestCase):
    def test_builtin_work_kits_are_discoverable_without_workspace_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            cards, warnings = discover_skill_cards(workspace)
            names = tuple(card.name for card in cards)

        self.assertEqual(warnings, ())
        self.assertIn("research-brief", names)
        self.assertIn("prd", names)
        self.assertIn("html-report", names)
        self.assertIn("product-advisor", names)
        self.assertIn("delivery-advisor", names)

    def test_workspace_skill_overrides_builtin_work_kit_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            skill_dir = workspace / "skills" / "prd"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: prd\ndescription: Workspace PRD.\n---\nProject-specific PRD.",
                encoding="utf-8",
            )

            cards, warnings = discover_skill_cards(workspace)
            catalog, catalog_warnings = workspace_skill_catalog(workspace)
            self.assertIsNotNone(catalog)
            assert catalog is not None
            document = select_skill_documents(catalog, ("prd",), workspace)[0]

        self.assertEqual(tuple(card.name for card in cards).count("prd"), 1)
        self.assertEqual(warnings, ())
        self.assertEqual(catalog_warnings, ())
        self.assertIn("Project-specific PRD", document.body)

    def test_builtin_skill_cards_are_stable_work_kits(self) -> None:
        cards, warnings = discover_builtin_skill_cards()

        self.assertEqual(warnings, ())
        self.assertEqual(
            tuple(card.name for card in cards),
            (
                "architecture-advisor",
                "data-advisor",
                "delivery-advisor",
                "html-report",
                "prd",
                "product-advisor",
                "research-advisor",
                "research-brief",
                "tech-diagram",
                "technical-architecture",
                "ux-advisor",
            ),
        )

    def test_builtin_advisor_can_be_loaded_as_skill_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            catalog, warnings = workspace_skill_catalog(workspace)
            self.assertIsNotNone(catalog)
            assert catalog is not None

            document = select_skill_documents(
                catalog,
                ("product-advisor",),
                workspace,
            )[0]

        self.assertEqual(warnings, ())
        self.assertEqual(document.name, "product-advisor")
        self.assertIn("Product Advisor", document.body)
        self.assertIn("Subagent Use", document.body)

    def test_builtin_advisors_enter_surface_as_summaries_only(self) -> None:
        cards, warnings = discover_builtin_skill_cards()
        surface = from_skill_cards(cards)
        refs = {ref.name: ref for ref in surface.list_refs()}

        self.assertEqual(warnings, ())
        self.assertIn("product-advisor", refs)
        self.assertIn("Product advisor", refs["product-advisor"].description)
        self.assertNotIn("Subagent Use", refs["product-advisor"].description)

    def test_discovery_skips_os_errors_and_keeps_other_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            skill_dir = workspace / "skills"
            good_dir = skill_dir / "good"
            bad_dir = skill_dir / "bad"
            good_dir.mkdir(parents=True)
            bad_dir.mkdir(parents=True)
            (good_dir / "SKILL.md").write_text(
                "---\nname: good\ndescription: Good skill.\n---\nBody.",
                encoding="utf-8",
            )
            (bad_dir / "SKILL.md").write_text(
                "---\nname: bad\ndescription: Bad skill.\n---\nBody.",
                encoding="utf-8",
            )

            def fake_load_skill_card(path: Path):
                if path.parent.name == "bad":
                    raise PermissionError("denied")
                from deepmate.skills import load_skill_card

                return load_skill_card(path)

            with patch(
                "deepmate.channels.skill_view.load_skill_card",
                side_effect=fake_load_skill_card,
            ):
                cards, warnings = discover_workspace_skill_cards(workspace)

        self.assertEqual(tuple(card.name for card in cards), ("good",))
        self.assertTrue(
            any(warning.code == "skill_discovery_failed" for warning in warnings)
        )

    def test_discovers_workspace_compatible_project_skills_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            skill_dir = workspace / ".claude" / "skills" / "product-review"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    (
                        "---",
                        "name: product-review",
                        "when_to_use: >",
                        "  Use when reviewing product specs and roadmap docs.",
                        "metadata:",
                        "  short-description: Product review",
                        "---",
                        "# Product Review",
                        "",
                        "Review product plans for scope, risks, and tradeoffs.",
                    )
                ),
                encoding="utf-8",
            )

            cards, warnings = discover_workspace_skill_cards(workspace)

        self.assertEqual(warnings, ())
        self.assertEqual(tuple(card.name for card in cards), ("product-review",))
        self.assertIn("Review product plans", cards[0].description)
        self.assertIn("Use when reviewing product specs", cards[0].description)
        self.assertEqual(
            cards[0].metadata["metadata"],
            {"short-description": "Product review"},
        )

    def test_disable_model_invocation_hides_from_surface_but_allows_explicit_load(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            skill_dir = workspace / ".claude" / "skills" / "manual-only"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    (
                        "---",
                        "name: manual-only",
                        "description: Manual-only skill.",
                        "disable-model-invocation: true",
                        "---",
                        "Only use when explicitly selected.",
                    )
                ),
                encoding="utf-8",
            )
            cards, _warnings = discover_workspace_skill_cards(workspace)
            catalog = SkillCatalog(cards)

            surface = from_skill_cards(cards)
            selected = select_skill_documents(catalog, ("manual-only",), workspace)

        self.assertTrue(surface.is_empty())
        self.assertEqual(selected[0].name, "manual-only")
        self.assertIn("explicitly selected", selected[0].body)


if __name__ == "__main__":
    unittest.main()
