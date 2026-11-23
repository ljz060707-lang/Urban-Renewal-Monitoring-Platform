"""Promote only diagnostic-smoke-passing candidates into the final freeze table."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--suggested", type=Path, required=True)
    parser.add_argument("--screen-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_receipt.read_text(encoding="utf-8"))
    screen = json.loads(args.screen_receipt.read_text(encoding="utf-8"))
    rows = read_csv(args.suggested)
    if not rows or any(row.get("selection_status") != "SMOKE_SCREEN_PASS" for row in rows):
        raise RuntimeError("Refusing to freeze a selection containing smoke-screen failures.")
    write_csv(args.output / "地块选择冻结表.csv", rows)
    receipt = {
        **base,
        "status": "SELECTION_FROZEN_AFTER_B0_SMOKE_SCREEN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_method": "spatially stratified low-resolution proxy shortlist plus B0 center-tile smoke; diagnostic only",
        "smoke_screen_receipt": str(args.screen_receipt),
        "smoke_screen_candidates_tested": screen["candidates_tested"],
        "smoke_screen_pass_count": screen["smoke_pass_count"],
        "smoke_screen_suggested_pass_count": screen["suggested_pass_count"],
        "selected_count": len(rows),
        "selected_blocks": rows,
    }
    (args.output / "地块选择冻结回执.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "selected_count": len(rows), "smoke_screen_pass_count": screen["smoke_pass_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
