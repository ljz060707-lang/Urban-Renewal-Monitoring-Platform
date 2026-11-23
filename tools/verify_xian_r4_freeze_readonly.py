#!/usr/bin/env python3
"""Read-only SHA recheck of the Stage-2 frozen R4 candidate inputs."""

import hashlib
import json
import argparse
from pathlib import Path


freeze = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2\13_receipts\R4_FREEZE_STAGE2.json")
parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=Path)
output = parser.parse_args().output_dir / "R4_POSTCHECK.json"
if output.exists():
    raise FileExistsError(output)
files = json.loads(freeze.read_text(encoding="utf-8"))["files"]
checks = {}
for name, row in files.items():
    path = Path(row["path"])
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    checks[name] = {"path": str(path), "expected_sha256": row["sha256"],
                    "actual_sha256": digest.hexdigest(), "match": digest.hexdigest() == row["sha256"]}
result = {"schema_version": "xian_r5.r4_post_reconstruction_check.v1",
          "frozen_source": str(freeze), "checked": len(checks),
          "all_match": all(row["match"] for row in checks.values()), "files": checks}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"checked": result["checked"], "all_match": result["all_match"]}))
if not result["all_match"]:
    raise SystemExit(1)
