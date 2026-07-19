from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.loader import (
    EvalConfigurationError,
    _minimal_schema_validation,
    _safe_asset_path,
    case_set_digest,
    fixture_digest,
    load_case,
    load_cases,
    policy_digest,
    select_cases,
    validate_repository_assets,
)


class EvalCatalogTests(unittest.TestCase):
    def test_catalog_has_exact_delivery_counts_and_unique_ids(self) -> None:
        cases = load_cases()
        self.assertEqual(len(cases), 45)
        self.assertEqual(len({case.id for case in cases}), 45)
        self.assertEqual(validate_repository_assets(cases), ())
        self.assertEqual(len(select_cases(cases, preset="pr")), 30)
        self.assertEqual(
            len([case for case in select_cases(cases, preset="pr") if case.is_safety()]),
            15,
        )

    def test_all_cases_reference_known_okrs_and_have_stable_digests(self) -> None:
        cases = load_cases()
        self.assertEqual(case_set_digest(cases), case_set_digest(load_cases()))
        self.assertEqual(fixture_digest(cases), fixture_digest(load_cases()))
        self.assertEqual(policy_digest(cases), policy_digest(load_cases()))
        self.assertTrue(all(len(case.digest) == 64 for case in cases))

    def test_case_loader_rejects_paths_outside_catalog(self) -> None:
        with self.assertRaises(EvalConfigurationError):
            load_case(__file__)

    def test_empty_selection_is_invalid(self) -> None:
        with self.assertRaises(EvalConfigurationError):
            select_cases(load_cases(), case_id="dm-missing-case-999")

    def test_offline_catalog_is_not_silently_selected_for_online_runs(self) -> None:
        with self.assertRaises(EvalConfigurationError):
            select_cases(load_cases(), preset="pr", execution_mode="online")

    def test_dependency_free_schema_validation_rejects_invalid_core_fields(self) -> None:
        loaded = next(iter(load_cases()))
        self.assertIsNotNone(loaded.path)
        case = loaded.to_record()
        case.pop("path", None)
        case.pop("digest", None)
        case["suite"] = "unknown"
        with self.assertRaises(EvalConfigurationError):
            _minimal_schema_validation(case, loaded.path or Path("case.yaml"))

    def test_asset_validation_rejects_symlink_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(EvalConfigurationError):
                _safe_asset_path(link, root, "test asset")


if __name__ == "__main__":
    unittest.main()
