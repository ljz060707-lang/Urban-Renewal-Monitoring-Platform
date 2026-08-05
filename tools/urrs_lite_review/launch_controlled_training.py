from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))
from urrs_lite_review.core import DATA_ROOT, ReviewDB, atomic_write_json, now_iso  # noqa: E402

CONFIRMATIONS = {"pilot": "START PILOT", "controlled": "START CONTROLLED TRAINING"}
ALLOWED_COMMAND_IDS = {"TR1_HEAD_ONLY", "SC0_RGB_ONLY", "SC1_RGB_PLUS_CA_PROBABILITIES"}


def is_allowed_command(command_id: str, command: list[str] | None = None) -> bool:
    if command_id not in ALLOWED_COMMAND_IDS: return False
    # The browser/API can submit only an identifier. An optional command is
    # accepted for offline tests only when it exactly matches a fixed token.
    return command is None or command in (["python", "-m", "urrs_lite_review.training_stub"], ["echo", "URRS_CONTROLLED_TRAINING"])


def launch(project: str, gate_name: str, confirmation: str, command_id: str, command: list[str] | None = None, execute: bool = False) -> dict:
    db = ReviewDB()
    try:
        gate = db.conn.execute("SELECT status,details_json FROM training_gates WHERE gate_id=?", (gate_name,)).fetchone()
        receipt = {"created_at": now_iso(), "project": project, "gate": gate_name, "command_id": command_id, "confirmation_required": CONFIRMATIONS.get("pilot" if "pilot" in gate_name else "controlled"), "holdout_reads": 0, "current_best_registry_modified": False, "status": "blocked", "reason": None}
        expected = CONFIRMATIONS["pilot" if "pilot" in gate_name else "controlled"]
        if not gate or gate["status"] != "pass": receipt["reason"] = "training gate is not passed"
        elif confirmation != expected: receipt["reason"] = "confirmation phrase mismatch"
        elif "pilot" in gate_name.lower() and command_id != "TR1_HEAD_ONLY": receipt["reason"] = "Pilot Gate permits TR1_HEAD_ONLY only"
        elif not is_allowed_command(command_id, command): receipt["reason"] = "command is not on the whitelist"
        elif execute and command:
            proc = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True, timeout=3600, check=False)
            receipt.update({"status": "started" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]})
        else: receipt.update({"status": "ready_for_explicit_execution", "reason": "dry-run receipt; no training process started"})
        atomic_write_json(DATA_ROOT / "TRAINING_START_RECEIPT.json", receipt)
        return receipt
    finally: db.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--project", default="nanjing_building"); p.add_argument("--gate", default="building_pilot"); p.add_argument("--confirm", default=""); p.add_argument("--command-id", default="TR1_HEAD_ONLY", choices=sorted(ALLOWED_COMMAND_IDS)); p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv); receipt = launch(args.project, args.gate, args.confirm, args.command_id, execute=args.execute); print(json.dumps(receipt, ensure_ascii=False, indent=2)); return 0 if receipt["status"] != "blocked" else 2


if __name__ == "__main__": raise SystemExit(main())
