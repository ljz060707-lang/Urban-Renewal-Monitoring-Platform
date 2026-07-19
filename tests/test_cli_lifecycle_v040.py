import json
from pathlib import Path
from types import SimpleNamespace

import urban_monitor.cli as cli


def test_failed_run_does_not_update_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    run = cli._unique_run("demo")
    try:
        raise RuntimeError("intentional failure")
    except RuntimeError as exc:
        cli.fail_run(run, "prepare", exc)
    manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed" and manifest["failure_stage"] == "prepare"
    assert not (tmp_path / "outputs" / "demo" / "latest.json").exists()


def test_successful_run_updates_latest_without_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    run = cli._unique_run("demo")
    checkpoint = run / "training" / "checkpoints" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"best")
    cli.complete_run(run, checkpoint)
    latest_path = tmp_path / "outputs" / "demo" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["status"] == "success" and latest["best_checkpoint"] == str(checkpoint)
    assert not latest_path.is_symlink()


def test_label_change_invalidates_data_hash(tmp_path):
    image = tmp_path / "image.tif"
    label = tmp_path / "label.gpkg"
    image.write_bytes(b"image")
    label.write_bytes(b"label-v1")
    config = {
        "data": {}, "tiling": {}, "split": {}, "crs": {},
    }
    project = SimpleNamespace(images={"2024-01-01": image}, labels={"2024-01-01": label}, project_dir=tmp_path, config=config)
    first = cli.compute_data_hash(project)
    label.write_bytes(b"label-v2")
    assert cli.compute_data_hash(project) != first


def test_force_flag_exists_on_all_user_commands():
    parser = cli.build_parser()
    for command in ("prepare", "train", "evaluate", "predict", "track", "all", "demo"):
        args = parser.parse_args([command, "--aoi", "demo", "--force"] if command != "demo" else [command, "--force"])
        assert args.force is True
