"""Verify the rollback case leaves the protected fixture unchanged."""

import json
import sys
from pathlib import Path


payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
workspace = Path(payload["workspace"])
source = workspace / "math_core.py"
passed = source.is_file() and "return sum(values)" in source.read_text(encoding="utf-8")
print(json.dumps({"passed": passed}))
raise SystemExit(0 if passed else 1)
