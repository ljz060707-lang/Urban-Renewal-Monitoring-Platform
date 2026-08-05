"""Process-level fallback bridge for the URRS recovery workflow.

This runner is intentionally small: it does not elevate privileges, mutate
system configuration, download data, or choose arbitrary project stages.  It
exists only to make an absolute, logged subprocess invocation reproducible when
the Codex shell helper is unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one logged local URRS command")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.command or args.command[0] == "--":
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    else:
        command = args.command
    if not command:
        parser.error("a command is required after --")

    cwd = Path(args.cwd).resolve()
    log_path = Path(args.log).resolve()
    if not cwd.is_dir():
        raise SystemExit(f"cwd does not exist: {cwd}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_temp = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Temp" / "urrs_codex_recovery"
    recovery_temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TEMP"] = str(recovery_temp)
    env["TMP"] = str(recovery_temp)
    env["PYTHONUTF8"] = "1"

    started = time.time()
    record = {
        "command": [str(item) for item in command],
        "cwd": str(cwd),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timeout_seconds": args.timeout,
        "temp": str(recovery_temp),
    }
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
            check=False,
        )
        record.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
            }
        )
    except subprocess.TimeoutExpired as exc:
        record.update(
            {
                "returncode": 124,
                "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
                "timed_out": True,
            }
        )
    record["duration_seconds"] = round(time.time() - started, 3)
    log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(record.get("stdout", ""))
    sys.stderr.write(record.get("stderr", ""))
    return int(record.get("returncode", 1))


if __name__ == "__main__":
    raise SystemExit(main())
