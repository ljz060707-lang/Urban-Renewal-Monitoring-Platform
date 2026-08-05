from pathlib import Path

import pytest
import yaml


def test_change_aware_crop_hits_requested_demolition_region():
    torch = pytest.importorskip("torch")
    from urban_monitor.change.augmentations import ChangePairTransform
    mask = torch.zeros((1, 512, 512)); mask[:, 430:450, 430:450] = 1
    sample = {
        "t1": torch.zeros((3, 512, 512)), "t2": torch.zeros((3, 512, 512)),
        "binary": mask.clone(), "new": torch.zeros_like(mask), "demolition": mask.clone(),
        "semantic_t1": None, "semantic_t2": None, "fn_binary": None, "fn_new": None,
        "fn_demolition": mask.clone(), "target_head": "demolition", "hard_sample": True,
    }
    transform = ChangePairTransform(256, training=True, brightness=0, contrast=0, noise_std=0,
                                    change_aware={"enabled": True, "target_coverage_probability": 1.0, "boundary_probability": 0.0, "jitter_px": 0})
    output = transform(sample)
    assert output["hard_patch_hit"] is True
    assert bool((output["demolition"] > 0).any())
    assert output["crop_target_head"] == "demolition"


def test_ablation_plan_starts_with_one_epoch_and_isolates_test():
    root = Path(__file__).parents[1]
    plan = yaml.safe_load((root / "experiments" / "change_ablation_plan.yaml").read_text(encoding="utf-8"))
    assert plan["smoke_policy"]["initial_epochs"] == 1
    assert plan["split_policy"]["test_used_for_development"] is False
    assert all(item["epochs"] <= 1 for item in plan["experiments"].values())


def test_threshold_selection_rejects_test_before_inference(tmp_path: Path):
    from urban_monitor.change.thresholds import scan_thresholds
    with pytest.raises(ValueError, match="validation-only"):
        scan_thresholds(tmp_path / "missing.pth", tmp_path / "missing.csv", tmp_path, split="test")


def test_binary_only_threshold_scan_skips_unsupported_heads_without_reading_test(tmp_path: Path, monkeypatch):
    import json
    import numpy as np
    import urban_monitor.change.thresholds as thresholds

    checkpoint = tmp_path / "binary.pth"; checkpoint.write_bytes(b"binary-only")
    seen = []
    def cached(*args, **kwargs):
        seen.append(kwargs["split"])
        return {
            "binary": (np.array([.9, .1], dtype=np.float32), np.array([1, 0], dtype=np.uint8)),
            "new": (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint8)),
            "demolition": (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.uint8)),
        }, {"validation_samples": 1, "selection": {"test_images_read": 0}, "cache_hit": False, "inference_seconds": .01}
    monkeypatch.setattr(thresholds, "validation_probabilities", cached)
    output = thresholds.scan_thresholds(checkpoint, tmp_path / "index.csv", tmp_path / "run", device_name="cpu")
    assert seen == ["validation"]
    assert set(output["head"]) == {"binary"}
    root = tmp_path / "run" / "metrics" / "threshold_scans" / __import__("hashlib").sha256(b"binary-only").hexdigest()
    summary = json.loads((root / "threshold_scan.json").read_text(encoding="utf-8"))
    assert summary["heads"]["binary"]["status"] == "available" and "best" in summary["heads"]["binary"]
    assert summary["heads"]["new"] == {"status": "skipped_no_supervision"}
    assert summary["heads"]["demolition"] == {"status": "skipped_no_supervision"}
    report = (root / "THRESHOLD_DIAGNOSTIC.md").read_text(encoding="utf-8")
    assert "new: skipped_no_supervision" in report and "demolition: skipped_no_supervision" in report
