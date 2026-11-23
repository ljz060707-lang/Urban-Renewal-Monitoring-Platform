"""Synthetic only: no public payloads, real labels, real models or cloud reads."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import xian_cloud_public_final as final
from xian_cloud_s_train import _metrics as frozen_metrics, _ratios as frozen_ratios


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def record(index=0, dataset="BANDON", role="FINAL_TEST", size=(9, 11)):
    rid = f"{dataset}_{role}_{index}"
    return dict(id=rid, dataset=dataset, split=role, native_size=list(size),
                parent_scene=f"{dataset}/{role}/parent{index % 20}", city="SYNTHETIC_CITY", time_group="SYNTHETIC_TIME",
                image=f"data/{dataset.lower()}/{role}/{rid}.png",
                label=f"data/{dataset.lower()}/{role}/{rid}_label.png",
                image_sha256=digest(rid), label_sha256=digest(rid + "label"),
                prior_trained_parent=False, official_split="val" if role == "EXPOSED_DEV" else "train",
                label_kind="BANDON_ROOF_FACADE_0_1_2_255" if dataset == "BANDON" else "BONAI_ROOF_0_1_255",
                facade_available=dataset == "BANDON", GSD_m=None)


def metadata():
    return dict(old_protected_tests_included=False,
                records=[record(i, dataset, role) for (dataset, role), count in final.EXPECTED_COUNTS.items() for i in range(count)])


@pytest.fixture
def frozen_tree(tmp_path, monkeypatch):
    """Small synthetic files, none of which are real model or dataset assets."""
    payload, release = tmp_path / "payload", tmp_path / "RELEASE"
    release.mkdir()
    for relative, contents, constant in (
        ("weights/B0.pth", "SYNTHETIC_B0_NOT_A_MODEL", "B0_SHA256"),
        ("code/tools/xian_cloud_s_train.py", "SYNTHETIC_TRAINER_NOT_EXECUTED", "S_TRAINER_SHA256"),
        ("code/src/xian_r5/cloud_joint/segmentation.py", "SYNTHETIC_ADAPTER_NOT_EXECUTED", "S_ADAPTER_SHA256"),
    ):
        path = payload / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        monkeypatch.setattr(final, constant, final._sha(path))
    files = {}
    for name in ("S_parent_state.pth", "S_state.pth", "A_state.pth"):
        path = release / name
        path.write_text("SYNTHETIC_WEIGHTS_" + name, encoding="utf-8")
        files[name] = final._sha(path)
    manifest = dict(frozen_before_FINAL_test=True, GT_required_for_inference=False,
                    release_scope="JOINT_RESEARCH_CANDIDATE_NOT_PRODUCTION_ADMITTED", files=files)
    (release / "MODEL_FREEZE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (payload / "config").mkdir()
    (payload / "config/segmentation_data.json").write_text(json.dumps(metadata()), encoding="utf-8")
    return payload, release, manifest


def test_selected_exact334_roles_metadata_only():
    selected = final.select_public_records(metadata())
    assert len(selected) == 334
    assert final.Counter((r["dataset"], r["split"]) for r in selected) == final.PUBLIC_COUNTS
    assert not any(r["split"] in {"TRAIN", "TUNE"} for r in selected)


@pytest.mark.parametrize("kind", ["counts", "old_tests", "parent", "prior", "duplicate", "test_path", "official_test", "label_schema"])
def test_metadata_rejects_role_or_provenance_drift(kind):
    manifest = metadata()
    rows = manifest["records"]
    selected = next(r for r in rows if r["split"] == "FINAL_TEST")
    if kind == "counts":
        rows.pop()
    elif kind == "old_tests":
        manifest["old_protected_tests_included"] = True
    elif kind == "parent":
        selected["parent_scene"] = rows[0]["parent_scene"]
    elif kind == "prior":
        selected["prior_trained_parent"] = None
    elif kind == "duplicate":
        selected["image_sha256"] = rows[0]["image_sha256"]
    elif kind == "test_path":
        selected["image"] = "old_locked20/test/protected.png"
    elif kind == "official_test":
        selected["official_split"] = "test"
    else:
        selected["label_kind"] = "FOOTPRINT_NOT_ROOF"
    with pytest.raises(ValueError):
        final.select_public_records(manifest)


@pytest.mark.parametrize("flag", [False, None, "true", 1])
def test_freeze_flag_must_be_literal_true_before_data(frozen_tree, flag):
    payload, release, manifest = frozen_tree
    manifest["frozen_before_FINAL_test"] = flag
    (release / "MODEL_FREEZE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze"):
        final.verify_release(payload, release)
    assert not (release.parent / final.LEDGER_NAME).exists()


def test_frozen_model_hash_mismatch_rejected_before_claim(frozen_tree):
    payload, release, _ = frozen_tree
    (release / "S_state.pth").write_text("CHANGED_SYNTHETIC_ONLY", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        final.verify_release(payload, release)
    assert not (release.parent / final.LEDGER_NAME).exists()


def test_verify_all_frozen_models_not_only_the_used_S(frozen_tree):
    payload, release, _ = frozen_tree
    final.verify_release(payload, release)
    (release / "A_state.pth").write_text("CHANGED_SYNTHETIC_A", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        final.verify_release(payload, release)


def test_once_ledger_refuses_another_new_output_and_existing_output(tmp_path):
    release = tmp_path / "RELEASE"
    release.mkdir()
    output = tmp_path / "EVAL1"
    ledger = final.claim_once(release, output, {"synthetic": True})
    original = ledger.read_bytes()
    with pytest.raises(FileExistsError):
        final.claim_once(release, tmp_path / "EVAL2", {"synthetic": True})
    with pytest.raises(FileExistsError):
        final.claim_once(release, output, {"synthetic": True})
    assert ledger.read_bytes() == original


@pytest.mark.parametrize("relative", ["../oldtest/a", "/absolute/a", "data/../TRAIN/a", "C:/locked/a", "data\\test\\a"])
def test_path_confinement(tmp_path, relative):
    with pytest.raises(ValueError):
        final._path_in(tmp_path, relative)


@pytest.mark.parametrize("mode", ["normal", "all_ignore", "no_roofs", "empty_prediction"])
def test_roof_metrics_equal_frozen_native_protocol(mode):
    label = np.zeros((32, 40), dtype=np.uint8)
    label[8:20, 10:25] = 1
    label[14:22, 25:29] = 2
    label[:3] = 255
    pred = np.zeros(label.shape, dtype=bool)
    pred[9:21, 10:24] = True
    if mode == "all_ignore":
        label[:] = 255
    elif mode == "no_roofs":
        label[label == 1] = 0
    elif mode == "empty_prediction":
        pred[:] = False
    target = np.where(label == 255, 255, label == 1).astype(np.uint8)
    counts = final._roof_counts(pred, target)
    assert counts == frozen_metrics(pred, target)
    assert final._ratios(counts) == frozen_ratios(counts)


def test_E0_facade_null_and_trained_heads_confusion_both_state():
    label = np.array([[0, 1, 2, 255], [0, 1, 2, 0]], dtype=np.uint8)
    prediction = dict(roof=np.array([[0, 1, 1, 1], [0, 1, 0, 0]], dtype=bool),
                      facade=np.array([[0, 0, 1, 1], [1, 1, 1, 0]], dtype=bool))
    row = record(size=label.shape)
    baseline = final.score_prediction(row, "E0", prediction, label)
    for key in ("facade_counts", "facade_iou", "facade_precision", "facade_recall", "gt3_prediction4_confusion"):
        assert baseline[key] is None
    assert "UNTRAINED_RANDOM" in baseline["facade_status"]
    trained = final.score_prediction(row, "E3", prediction, label)
    matrix = np.array(trained["gt3_prediction4_confusion"])
    assert matrix.shape == (3, 4) and matrix.sum() == 7
    assert matrix[:, 3].sum() == 2
    assert trained["facade_status"] == final.FACADE_SEMANTICS
    assert trained["physical_instance_small_roof_recall"] is None


def test_bonai_no_facade_GT_and_empty_valid_not_false_accuracy():
    label = np.full((8, 8), 255, dtype=np.uint8)
    prediction = dict(roof=np.ones((8, 8), dtype=bool), facade=np.ones((8, 8), dtype=bool))
    row = final.score_prediction(record(dataset="BONAI", role="EXPOSED_DEV"), "E1", prediction, label)
    assert row["facade_iou"] is None and row["gt3_prediction4_confusion"] is None
    assert row["roof_iou"] is None and row["absolute_roof_area_error_pixels"] is None
    assert row["label_status"] == "NO_VALID_LABEL_PIXELS"
    aggregate = final.aggregate_rows([row])
    assert aggregate["mean_absolute_roof_area_error_pixels"] is None
    assert aggregate["images_with_valid_labels"] == 0


def test_area_MAE_keeps_opposite_errors_and_roles_separate():
    label = np.zeros((8, 8), dtype=np.uint8)
    label[3, 3] = 1
    more, less = label.astype(bool), np.zeros((8, 8), dtype=bool)
    more[4, 4] = True
    rows = [final.score_prediction(record(i), "E3", dict(roof=p, facade=less), label) for i, p in enumerate((more, less))]
    summary = final.aggregate_rows(rows)
    assert summary["mean_signed_roof_area_error_pixels"] == 0
    assert summary["mean_absolute_roof_area_error_pixels"] == 1
    assert summary["roof_pixel_MAE"] == 1/64
    rows[1]["split"] = "EXPOSED_DEV"
    groups = final.grouped_report(rows)
    assert set(groups) == {"BANDON/FINAL_TEST/E3", "BANDON/EXPOSED_DEV/E3"}
    assert all(set(g["groups"]) == {"source_group", "parent_scene", "city", "time_group"} for g in groups.values())


class PixelModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, rgb):
        assert not torch.is_grad_enabled()
        self.calls += 1
        z = rgb[:, :1].float() - 128
        return dict(roof_logits=z, facade_logits=z)


def test_nonoverlapping512_edge_tiles_stitch_native_and_dual_heads_independent():
    models = {name: PixelModel() for name in final.EXPERIMENTS}
    rgb = np.zeros((513, 515, 3), dtype=np.uint8)
    rgb[::2, ::3, 0] = 255
    rgb[0, 0, 0] = 128  # sigmoid == .5 must remain false.
    pred, count = final.predict_native(models, rgb, lambda x: x.float(), "cpu")
    assert count == 4
    for name in models:
        assert models[name].calls == 4
        np.testing.assert_array_equal(pred[name]["roof"], rgb[:, :, 0] > 128)
        np.testing.assert_array_equal(pred[name]["facade"], pred[name]["roof"])


def test_wrong_shape_or_nonfinite_logits_fail_before_label():
    class Bad(PixelModel):
        def forward(self, rgb):
            return dict(roof_logits=torch.zeros(1, 1, 32, 32), facade_logits=torch.zeros(1, 1, 512, 512))
    with pytest.raises(ValueError, match="shaped"):
        final.predict_native({name: Bad() for name in final.EXPERIMENTS}, np.zeros((8, 8, 3), np.uint8), lambda x: x.float(), "cpu")


def test_streaming_image_label_read_once_and_only_after_all_predictions(tmp_path, monkeypatch):
    payload, output = tmp_path / "payload", tmp_path / "OUT"
    payload.mkdir()
    output.mkdir()
    row = record(size=(9, 11))
    calls = []
    models = {name: PixelModel() for name in final.EXPERIMENTS}
    def load(path, expected_sha, *, label=False):
        if label:
            assert all(m.calls == 1 for m in models.values())
            assert len(list(output.glob("predictions/*/*.png"))) == 6
        calls.append("label" if label else "image")
        return np.zeros((9, 11), np.uint8) if label else np.zeros((9, 11, 3), np.uint8)
    monkeypatch.setattr(final, "_load_image_once", load)
    rows = final.evaluate_records(payload, [row], models, lambda x: x.float(), "cpu", output)
    assert calls == ["image", "label"]
    assert len(rows) == 3
    events = [json.loads(x)["event"] for x in (output / "READ_EVENTS.jsonl").read_text().splitlines()]
    assert events == ["RGB_READ_INTENT", "ALL_PREDICTION_ARRAYS_GENERATED_BEFORE_LABEL", "LABEL_READ_INTENT_AFTER_PREDICTION"]


def test_TRAIN_role_rejected_without_image_access(tmp_path, monkeypatch):
    def deny(*a, **k):
        raise AssertionError("TRAIN pixels must never be read")
    monkeypatch.setattr(final, "_load_image_once", deny)
    with pytest.raises(ValueError, match="allowlisted"):
        final.evaluate_records(tmp_path, [record(role="TRAIN")], {}, None, "cpu", tmp_path)


def test_failure_consumes_global_read_claim_without_retry(frozen_tree, monkeypatch):
    payload, release, _ = frozen_tree
    calls = []
    def fail(*args):
        calls.append("runtime")
        raise RuntimeError("SYNTHETIC_FAILURE_BEFORE_ANY_PIXELS")
    monkeypatch.setattr(final, "_load_runtime", fail)
    with pytest.raises(RuntimeError, match="SYNTHETIC_FAILURE"):
        final.run_public_final(payload, release, release.parent / "EVAL1")
    assert (release.parent / final.LEDGER_NAME).exists()
    assert (release.parent / "EVAL1/FAILED_NO_RETRY.json").exists()
    with pytest.raises(FileExistsError):
        final.run_public_final(payload, release, release.parent / "EVAL2")
    assert calls == ["runtime"]


def test_synthetic_one_image_end_to_end_frozen_gate_and_artifacts(frozen_tree, monkeypatch):
    payload, release, _ = frozen_tree
    row = final.select_public_records(metadata())[0]
    monkeypatch.setattr(final, "select_public_records", lambda data: [row])  # Test seam only; CLI has no count override.
    calls = []
    class Trainer:
        @staticmethod
        def make_s_model(root, checkpoint):
            calls.append(None if checkpoint is None else checkpoint.name)
            return PixelModel()
    monkeypatch.setattr(final, "_load_runtime", lambda root: (Trainer(), SimpleNamespace(normalize_rgb=lambda x: x.float()), "cpu"))
    def load(path, expected_sha, *, label=False):
        assert (release.parent / final.LEDGER_NAME).exists()
        return np.zeros(tuple(row["native_size"]), np.uint8) if label else np.zeros((*row["native_size"], 3), np.uint8)
    monkeypatch.setattr(final, "_load_image_once", load)
    output = release.parent / "RELEASE_PUBLIC_EVAL"
    result = final.run_public_final(payload, release, output)
    assert calls == [None, "S_parent_state.pth", "S_state.pth"]
    assert result["status"] == "COMPLETE" and result["model_image_evaluations"] == 3
    assert (output / "COMPLETE.json").exists()
    report = json.loads((output / "PUBLIC_FINAL_METRICS.json").read_text())
    assert report["gt_in_forward"] is False
    assert len(report["grouped"]) == 3
    assert (output / "ARTIFACT_SHA256_MANIFEST.json").exists()
