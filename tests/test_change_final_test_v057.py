from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _checkpoint(path: Path, torch, training: dict | None = None) -> Path:
    torch.save({"config": {"training": training or {"patch_size": 4, "batch_size": 8, "num_workers": 0}, "model": {}}, "model": {}}, path)
    return path


def test_final_test_reads_only_test_and_forwards_once(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    import urban_monitor.change.final_test as final_test

    run = tmp_path / "candidate"; run.mkdir()
    checkpoint = _checkpoint(tmp_path / "model.pth", torch)
    index = tmp_path / "index.csv"
    pd.DataFrame([
        {"sample_id": "train-1", "split": "train", "has_binary": True, "has_directional": True},
        {"sample_id": "val-1", "split": "validation", "has_binary": True, "has_directional": True},
        {"sample_id": "test-1", "split": "test", "has_binary": True, "has_directional": True},
    ]).to_csv(index, index=False)

    seen_splits = []

    class Dataset:
        def __init__(self, frame, transform):
            seen_splits.extend(frame["split"].tolist())
            assert set(frame["split"].str.casefold()) == {"test"}
        def __len__(self): return 1
        def __getitem__(self, index):
            target = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
            return {"t1": torch.zeros((3, 2, 2)), "t2": torch.zeros((3, 2, 2)),
                    "binary": target, "new": target, "demolition": target}

    class Model:
        calls = 0
        def to(self, device): return self
        def eval(self): return self
        def load_state_dict(self, state): pass
        def __call__(self, t1, t2):
            type(self).calls += 1
            logits = torch.tensor([[[[8.0, -8.0], [-8.0, 8.0]]]], device=t1.device)
            return {"binary_change": logits, "new_building": logits, "demolition": logits}

    monkeypatch.setattr(final_test, "PublicChangeDataset", Dataset)
    monkeypatch.setattr(final_test, "build_change_model", lambda config: Model())
    output = final_test.run_final_test(
        checkpoint, index, run, device_name="cpu",
        thresholds={"binary": .50, "new": .60, "demolition": .35}, confirm_final_test=True,
        cli_config={"provided": "config"},
    )
    assert seen_splits == ["test"]
    assert Model.calls == 1
    assert set(output["head"]) == {"binary", "new", "demolition"}
    assert set(output["sample_count"]) == {1}
    assert {"precision", "recall", "f1", "iou", "pr_auc", "false_positive_pixels", "false_negative_pixels",
            "true_positive_pixels", "predicted_positive_pixel_ratio", "actual_positive_pixel_ratio", "sample_count"} <= set(output.columns)
    assert (run / "metrics" / "final_test_metrics.csv").is_file()
    manifest = json.loads((run / "FINAL_TEST_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["test_used"] is True
    assert manifest["test_sample_count"] == 1
    assert manifest["thresholds"] == {"binary": .5, "new": .6, "demolition": .35}
    assert manifest["single_forward_pass"] is True


def test_final_test_requires_confirmation_and_refuses_repeat(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    import urban_monitor.change.final_test as final_test

    run = tmp_path / "candidate"; run.mkdir()
    checkpoint = _checkpoint(tmp_path / "model.pth", torch)
    index = tmp_path / "index.csv"
    pd.DataFrame([{"sample_id": "test-1", "split": "test"}]).to_csv(index, index=False)
    monkeypatch.setattr(final_test, "PublicChangeDataset", lambda *args, **kwargs: pytest.fail("test images must not be read"))
    with pytest.raises(PermissionError, match="confirm-final-test"):
        final_test.run_final_test(checkpoint, index, run, device_name="cpu", thresholds={"binary": .5, "new": .6, "demolition": .35}, confirm_final_test=False)
    (run / "FINAL_TEST_MANIFEST.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(final_test, "build_change_model", lambda config: pytest.fail("repeat must reject before model construction"))
    with pytest.raises(FileExistsError, match="refusing to read test again"):
        final_test.run_final_test(checkpoint, index, run, device_name="cpu", thresholds={"binary": .5, "new": .6, "demolition": .35}, confirm_final_test=True)


@pytest.mark.parametrize("training", [
    {"patch_size": 4, "batch_size": "auto", "num_workers": "auto", "cache_max_gb": "auto"},
    {"patch_size": 4, "batch_size": 1, "num_workers": 0, "cache_max_gb": 0},
])
def test_final_test_normalizes_auto_and_explicit_runtime_values(tmp_path: Path, monkeypatch, training):
    torch = pytest.importorskip("torch")
    import urban_monitor.change.final_test as final_test
    run = tmp_path / "candidate"; (run / "logs").mkdir(parents=True)
    (run / "logs" / "environment_directional.json").write_text('{"batch_size": 1}', encoding="utf-8")
    checkpoint = _checkpoint(tmp_path / "model.pth", torch, training)
    index = tmp_path / "index.csv"; pd.DataFrame([{"sample_id": "test", "split": "test", "has_binary": True, "has_directional": True}]).to_csv(index, index=False)

    class Dataset:
        def __init__(self, frame, transform): assert frame["split"].tolist() == ["test"]
        def __len__(self): return 1
        def __getitem__(self, index):
            target = torch.ones((1, 2, 2))
            return {"t1": torch.zeros((3, 2, 2)), "t2": torch.zeros((3, 2, 2)), "binary": target, "new": target, "demolition": target}
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def load_state_dict(self, state): pass
        def __call__(self, t1, t2):
            logits = torch.ones((len(t1), 1, 2, 2), device=t1.device)
            return {"binary_change": logits, "new_building": logits, "demolition": logits}
    monkeypatch.setattr(final_test, "PublicChangeDataset", Dataset)
    monkeypatch.setattr(final_test, "build_change_model", lambda config: Model())
    final_test.run_final_test(checkpoint, index, run, device_name="cpu", thresholds={"binary": .5, "new": .6, "demolition": .3}, confirm_final_test=True)
    manifest = json.loads((run / "FINAL_TEST_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["resolved_runtime"]["resolved_batch_size"] == 1


def test_invalid_runtime_config_fails_before_constructing_test_dataset(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    import urban_monitor.change.final_test as final_test
    run = tmp_path / "candidate"; run.mkdir()
    checkpoint = _checkpoint(tmp_path / "model.pth", torch, {"patch_size": 4, "batch_size": "invalid", "num_workers": "auto"})
    index = tmp_path / "index.csv"; pd.DataFrame([{"sample_id": "test", "split": "test"}]).to_csv(index, index=False)
    monkeypatch.setattr(final_test, "PublicChangeDataset", lambda *args, **kwargs: pytest.fail("test images must not be read"))
    with pytest.raises(ValueError, match="Invalid integer configuration: training.batch_size='invalid'"):
        final_test.run_final_test(checkpoint, index, run, device_name="cpu", thresholds={"binary": .5, "new": .6, "demolition": .3}, confirm_final_test=True)
    assert not (run / "FINAL_TEST_MANIFEST.json").exists()


def test_binary_only_final_test_skips_directional_heads_and_needs_only_binary_threshold(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    import urban_monitor.change.final_test as final_test
    run = tmp_path / "candidate"; run.mkdir()
    checkpoint = _checkpoint(tmp_path / "model.pth", torch)
    index = tmp_path / "index.csv"
    pd.DataFrame([
        {"sample_id": "train", "split": "train", "has_binary": True, "has_directional": False, "has_semantic": False},
        {"sample_id": "validation", "split": "validation", "has_binary": True, "has_directional": False, "has_semantic": False},
        {"sample_id": "test", "split": "test", "has_binary": True, "has_directional": False, "has_semantic": False},
    ]).to_csv(index, index=False)

    class Dataset:
        def __init__(self, frame, transform): assert frame["split"].tolist() == ["test"]
        def __len__(self): return 1
        def __getitem__(self, index):
            return {"t1": torch.zeros((3, 2, 2)), "t2": torch.zeros((3, 2, 2)),
                    "binary": torch.ones((1, 2, 2)), "new": None, "demolition": None}
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def load_state_dict(self, state): pass
        def __call__(self, t1, t2):
            logits = torch.ones((len(t1), 1, 2, 2), device=t1.device)
            return {"binary_change": logits, "new_building": logits, "demolition": logits}
    monkeypatch.setattr(final_test, "PublicChangeDataset", Dataset)
    monkeypatch.setattr(final_test, "build_change_model", lambda config: Model())
    output = final_test.run_final_test(
        checkpoint, index, run, device_name="cpu",
        thresholds={"binary": .75, "new": None, "demolition": None}, confirm_final_test=True,
    )
    binary = output.loc[output["head"] == "binary"].iloc[0]
    assert binary["status"] == "available" and binary["threshold"] == .75
    for head in ("new", "demolition"):
        row = output.loc[output["head"] == head].iloc[0]
        assert row["status"] == "skipped_no_supervision" and pd.isna(row["threshold"])
        assert pd.isna(row["f1"])
    receipt = json.loads((run / "FINAL_TEST_MANIFEST.json").read_text(encoding="utf-8"))
    assert receipt["supervision_mode"] == "binary_only"
    assert receipt["active_heads"] == ["binary"]
    assert receipt["new_threshold"] is None and receipt["demolition_threshold"] is None
    assert receipt["validation_sample_count"] == 1
    assert receipt["test_used_for_threshold_selection"] is False


def test_final_test_rejects_checkpoint_hash_before_test_dataset(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    import urban_monitor.change.final_test as final_test
    run = tmp_path / "candidate"; run.mkdir()
    checkpoint = _checkpoint(tmp_path / "model.pth", torch)
    index = tmp_path / "index.csv"
    pd.DataFrame([{"sample_id": "test", "split": "test", "has_binary": True, "has_directional": False}]).to_csv(index, index=False)
    monkeypatch.setattr(final_test, "PublicChangeDataset", lambda *args, **kwargs: pytest.fail("test images must not be read"))
    with pytest.raises(ValueError, match="Checkpoint SHA256 mismatch"):
        final_test.run_final_test(
            checkpoint, index, run, device_name="cpu", thresholds={"binary": .75},
            confirm_final_test=True, expected_checkpoint_sha256="0" * 64,
        )
    assert not (run / "FINAL_TEST_MANIFEST.json").exists()


def test_final_test_cli_allows_missing_directional_thresholds_and_scan_rejects_test():
    from urban_monitor.change.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["change-final-test", "--confirm-final-test"])
    with pytest.raises(SystemExit):
        parser.parse_args(["change-threshold-scan", "--split", "test"])
    parsed = parser.parse_args([
        "change-final-test", "--confirm-final-test", "--binary-threshold", ".50",
    ])
    assert parsed.confirm_final_test is True
    assert (parsed.binary_threshold, parsed.new_threshold, parsed.demolition_threshold) == (.5, None, None)
