from __future__ import annotations

import unittest
from pathlib import Path

from evals.privacy import committed_asset_paths, sanitize_value, scan_text, scan_value


class EvalPrivacyTests(unittest.TestCase):
    def test_scanner_detects_credential_address_path_and_marker_without_echoing(self) -> None:
        samples = (
            "api" + "_key=" + "syntheticvalue123456",
            "203" + ".0.113.7",
            "/" + "Users" + "/example/private",
            "user" + "name=" + "synthetic-account",
            "session" + "_id=" + "synthetic-session",
            "project" + "_name=" + "synthetic-project",
        )
        kinds = {finding.kind for sample in samples for finding in scan_text(sample)}
        self.assertTrue(
            {
                "api_key",
                "ipv4",
                "user_path",
                "account",
                "session_id",
                "private_project",
            }.issubset(kinds)
        )
        marker = "runtime-private-marker"
        findings = scan_value({"output": marker}, canaries=(marker,))
        self.assertEqual(findings[0].kind, "canary")
        self.assertNotIn(marker, findings[0].message)

    def test_sanitizer_removes_trial_roots_and_runtime_markers(self) -> None:
        marker = "runtime-private-marker"
        sanitized = sanitize_value(
            {"path": "/temporary/trial/workspace", "value": marker},
            replacements={"/temporary/trial": "<trial-root>"},
            canaries=(marker,),
        )
        self.assertEqual(sanitized["path"], "<trial-root>/workspace")
        self.assertEqual(sanitized["value"], "<redacted-private-marker>")

    def test_path_replacement_does_not_hide_canary_findings(self) -> None:
        marker = "runtime-private-marker"
        findings = scan_value(
            {"path": "/temporary/trial", "value": marker},
            replacements={"/temporary/trial": "<trial-root>"},
            canaries=(marker,),
        )
        self.assertEqual([finding.kind for finding in findings], ["canary"])

    def test_committed_asset_scan_covers_framework_code(self) -> None:
        root = Path(__file__).resolve().parents[1]
        relative = {path.relative_to(root).as_posix() for path in committed_asset_paths(root)}
        self.assertIn("runner.py", relative)
        self.assertIn("tests/test_privacy.py", relative)
        self.assertNotIn("privacy.py", relative)


if __name__ == "__main__":
    unittest.main()
