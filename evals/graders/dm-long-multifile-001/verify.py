"""Hidden verifier for the synthetic multi-file delivery case."""

import json
import sys
from pathlib import Path


payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
workspace = Path(payload["workspace"])
result = workspace / "RESULT.md"
passed = result.is_file() and "tests-passed" in result.read_text(encoding="utf-8")
print(json.dumps({"passed": passed}))
raise SystemExit(0 if passed else 1)
