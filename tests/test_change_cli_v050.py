from __future__ import annotations

import json

from urban_monitor.change.cli import _mark_failed_run, build_parser


def test_change_cli_commands_exist_and_v040_parser_still_routes():
    parser = build_parser()
    commands = {action.dest: set(action.choices) for action in parser._actions if action.dest == "command"}["command"]
    assert {"change-doctor","change-prepare","change-train","change-all","change-demo","change-review","change-retrain","change-labels","change-finetune"} <= commands
    from urban_monitor.cli import build_parser as old_parser
    assert old_parser().parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["change-all", "--quick"]).quick is True


def test_failed_change_run_is_auditable_and_does_not_create_latest(tmp_path):
    run = tmp_path / "run-1"
    run.mkdir()
    try:
        raise RuntimeError("synthetic failure")
    except RuntimeError as exc:
        _mark_failed_run(run, "change-all", exc)

    manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_stage"] == "change-all"
    assert manifest["error_message"] == "synthetic failure"
    assert (run / "logs" / "traceback.txt").is_file()
    assert not (tmp_path / "latest.json").exists()
