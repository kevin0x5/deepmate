"""Verify hidden evaluation assets were not copied into the agent workspace."""

import json
import sys
from pathlib import Path


payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
workspace = Path(payload["workspace"])
forbidden = {"expected_facts.json", "probe.json", "verify.py"}
passed = not any(path.name in forbidden for path in workspace.rglob("*"))
print(json.dumps({"passed": passed}))
raise SystemExit(0 if passed else 1)
