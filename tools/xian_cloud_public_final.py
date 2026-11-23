"""One-shot PUBLIC native S evaluation after model freeze; never train/select.

CLI: --payload PAYLOAD --release-dir ROOT/RELEASE --output ROOT/RELEASE_PUBLIC_EVAL
Only this process may consume the new PUBLIC FINAL. A durable, exclusive read
ledger in ROOT is consumed BEFORE any image/label read, including failed runs.
No resume/retry/threshold search. E0/E1/E3 here compare native S roof heads;
no A displacement or ten-metre integration is applied to public native images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import uuid

import numpy as np

sys.dont_write_bytecode = True
SIDE = 512
EXPERIMENTS = ("E0", "E1", "E3")
EXPECTED_COUNTS = {("BANDON", "TRAIN"): 1601, ("BONAI", "TRAIN"): 32,
                   ("BANDON", "TUNE"): 59, ("BANDON", "CALIBRATION"): 59,
                   ("BANDON", "FINAL_TEST"): 57, ("BANDON", "EXPOSED_DEV"): 206,
                   ("BONAI", "EXPOSED_DEV"): 12}
PUBLIC_COUNTS = {("BANDON", "FINAL_TEST"): 57, ("BANDON", "CALIBRATION"): 59,
                 ("BANDON", "EXPOSED_DEV"): 206, ("BONAI", "EXPOSED_DEV"): 12}
B0_SHA256 = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
S_TRAINER_SHA256 = "62bb74da06131dbba6161286ad253ce201a7cef1e95589d184cbb081539a1434"
S_ADAPTER_SHA256 = "3adbe2a525d74e5daa71cee2975616dddf6a8ed22370eba8894258ee6aa1447b"
FACADE_SEMANTICS = "BANDON_GEOMETRIC_PROTOCOL_NOT_EXACT_VISIBLE_FACADE_GT"
LEDGER_NAME = "PUBLIC_FINAL_READ_LEDGER.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _path_in(root, relative):
    """Confined explicit manifest path; reject aliases/symlink redirects."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Nonempty relative manifest path required")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in {"..", "."} for p in relative.split("/")) or "\\" in relative or ":" in relative:
        raise ValueError("Confined POSIX-relative path required")
    root = Path(root).resolve()
    raw = root / relative
    resolved = raw.resolve()
    if root not in resolved.parents or resolved != raw:
        raise ValueError("Path escapes/redirects the declared manifest role")
    return resolved


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_release(payload, release_dir):
    """Read only metadata/weights/code. No public image or label is accessed."""
    payload, release = Path(payload).resolve(), Path(release_dir).resolve()
    path = release / "MODEL_FREEZE_MANIFEST.json"
    manifest = _read_json(path)
    if manifest.get("frozen_before_FINAL_test") is not True or manifest.get("GT_required_for_inference") is not False:
        raise ValueError("Explicit pre-FINAL model freeze and GT-free inference required")
    files = manifest.get("files")
    required = {"S_parent_state.pth", "S_state.pth"}
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Frozen S parent and S release SHA entries required")
    snapshots = {str(path): _sha(path)}
    for relative, expected in files.items():
        if not _is_sha(expected):
            raise ValueError("Invalid frozen model SHA256")
        model_path = _path_in(release, relative)
        actual = _sha(model_path)
        if actual != expected:
            raise ValueError("Model changed after freeze: " + relative)
        snapshots[str(model_path)] = actual
    for relative, expected in (("weights/B0.pth", B0_SHA256),
                               ("code/tools/xian_cloud_s_train.py", S_TRAINER_SHA256),
                               ("code/src/xian_r5/cloud_joint/segmentation.py", S_ADAPTER_SHA256)):
        source = _path_in(payload, relative)
        if _sha(source) != expected:
            raise ValueError("Frozen B0/S implementation mismatch: " + relative)
        snapshots[str(source)] = expected
    return manifest, snapshots


def select_public_records(manifest):
    """Metadata-only allowlist. No TRAIN/TUNE/protected payload is opened."""
    if manifest.get("old_protected_tests_included") is not False:
        raise ValueError("Manifest must explicitly exclude old protected tests")
    records = manifest["records"]
    if Counter((r["dataset"], r["split"]) for r in records) != EXPECTED_COUNTS:
        raise ValueError("Frozen public manifest role counts changed")
    ids, parents, rgb_roles = set(), defaultdict(set), defaultdict(set)
    for row in records:
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("Unique source record IDs required")
        ids.add(row["id"])
        if not _is_sha(row["image_sha256"]) or not _is_sha(row["label_sha256"]):
            raise ValueError("Pinned RGB/label SHA required")
        role = row["split"]
        rgb_roles[row["image_sha256"]].add(role)
        if row["dataset"] == "BANDON" and role != "EXPOSED_DEV":
            parent = row.get("parent_scene")
            if not isinstance(parent, str) or not parent:
                raise ValueError("Original parent-scene provenance required")
            parents[role].add(parent)
            if role in {"TUNE", "CALIBRATION", "FINAL_TEST"} and row.get("prior_trained_parent") is not False:
                raise ValueError("New holdout has old/unknown fine-tuning exposure")
    roles = ("TRAIN", "TUNE", "CALIBRATION", "FINAL_TEST")
    for i, role in enumerate(roles):
        for other in roles[i + 1:]:
            if parents[role] & parents[other]:
                raise ValueError("Parent scene crosses frozen roles")
    if any(len(parents[r]) != 20 for r in roles[1:]):
        raise ValueError("New holdout requires the frozen20 parent scenes per role")
    if any(len(value & set(roles)) > 1 for value in rgb_roles.values()):
        raise ValueError("Duplicate RGB payload crosses train/holdout roles")
    selected = [r for r in records if (r["dataset"], r["split"]) in PUBLIC_COUNTS]
    if Counter((r["dataset"], r["split"]) for r in selected) != PUBLIC_COUNTS:
        raise ValueError("Expected PUBLIC57/CALIBRATION59/EXPOSED206+12")
    for row in selected:
        dataset, role = row["dataset"], row["split"]
        prefix = "data/" + dataset.lower() + "/" + role + "/"
        if any(not isinstance(row[k], str) or not row[k].startswith(prefix) for k in ("image", "label")):
            raise ValueError("Image/label not inside the declared public role directory")
        if dataset == "BANDON":
            if row.get("label_kind") != "BANDON_ROOF_FACADE_0_1_2_255" or row.get("facade_available") is not True:
                raise ValueError("Explicit BANDON roof/facade schema required")
            expected_split = "val" if role == "EXPOSED_DEV" else "train"
            if row.get("official_split") != expected_split:
                raise ValueError("Public holdouts must come from official train, not old test")
        elif row.get("label_kind") != "BONAI_ROOF_0_1_255" or row.get("facade_available") is not False:
            raise ValueError("Explicit BONAI roof-only schema required")
    return sorted(selected, key=lambda r: (r["dataset"], r["split"], r["id"]))


def claim_once(release_dir, output, registration):
    """One irreversible read claim per run root, not merely per output name.

    A failed/partial run consumes the claim. No automatic recovery or deletion.
    A different NEW output or release subdirectory cannot re-open this root.
    """
    release, output = Path(release_dir).resolve(), Path(output).resolve()
    if output.parent != release.parent or output == release:
        raise ValueError("NEW public output must be a sibling of RELEASE")
    output.mkdir(exist_ok=False)
    ledger = release.parent / LEDGER_NAME
    _dump_new(ledger, dict(registration, output=str(output), claimed_at=_now(),
                           claim="PUBLIC_FINAL_ACCESS_CONSUMED_NO_RETRY", pid=os.getpid()))
    return ledger


def _load_image_once(path, expected_sha, *, label=False):
    from PIL import Image
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Public payload SHA mismatch: " + str(path))
    with Image.open(io.BytesIO(data)) as image:
        if label:
            result = np.asarray(image).copy()
            if result.ndim != 2 or result.dtype.kind not in "ui":
                raise ValueError("Native integer single-band label required")
        else:
            result = np.asarray(image.convert("RGB")).copy()
    return result


def predict_native(models, rgb, normalize_rgb, device):
    """All E0/E1/E3 arrays exist before the caller may load the label."""
    import torch
    if tuple(models) != EXPERIMENTS or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Ordered E0/E1/E3 models and native RGB uint8 required")
    h, w = rgb.shape[:2]
    predictions = {name: {head: np.zeros((h, w), dtype=bool) for head in ("roof", "facade")} for name in models}
    tiles = 0
    with torch.inference_mode():
        for y in range(0, h, SIDE):
            for x in range(0, w, SIDE):
                th, tw = min(SIDE, h - y), min(SIDE, w - x)
                tile = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
                tile[:th, :tw] = rgb[y:y+th, x:x+tw]
                image = torch.from_numpy(tile.transpose(2, 0, 1).copy())[None].to(device)
                normalized = normalize_rgb(image)
                for experiment, model in models.items():
                    context = torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()
                    with context:
                        result = model(normalized)
                    for head in ("roof", "facade"):
                        logits = result[head + "_logits"]
                        if tuple(logits.shape) != (1, 1, SIDE, SIDE) or not bool(torch.isfinite(logits).all()):
                            raise ValueError("Nonfinite/wrong-shaped native " + head + " logits")
                        predictions[experiment][head][y:y+th, x:x+tw] = (
                            logits[0, 0, :th, :tw].float().sigmoid() > .5).cpu().numpy()
                tiles += 1
    return predictions, tiles


def _roof_counts(pred, target):
    """Equivalent to frozen S _metrics: inner8 boundary,2px,valid5x5core."""
    from scipy import ndimage as ndi
    valid, gt = target != 255, target == 1
    core = ndi.binary_erosion(valid, structure=np.ones((5, 5)), border_value=0)
    gb = gt & ~ndi.binary_erosion(gt, structure=np.ones((3, 3)), border_value=0) & core
    pb = pred & ~ndi.binary_erosion(pred, structure=np.ones((3, 3)), border_value=0) & core
    tp, fp, fn = int((pred & gt & valid).sum()), int((pred & ~gt & valid).sum()), int((~pred & gt & valid).sum())
    return dict(tp=tp, fp=fp, fn=fn, valid_pixels=int(valid.sum()), gt_roof_pixels=tp+fn,
                predicted_roof_pixels=tp+fp, signed_roof_area_error_pixels=fp-fn,
                absolute_roof_area_error_pixels=abs(fp-fn), pred_boundary=int(pb.sum()),
                gt_boundary=int(gb.sum()),
                matched_pred_boundary=int((pb & ndi.binary_dilation(gb, structure=np.ones((5, 5)))).sum()),
                matched_gt_boundary=int((gb & ndi.binary_dilation(pb, structure=np.ones((5, 5)))).sum()),
                physical_instance_metrics=None)


def _div(a, b):
    return a / b if b else None


def _ratios(counts):
    precision = _div(counts["matched_pred_boundary"], counts["pred_boundary"])
    recall = _div(counts["matched_gt_boundary"], counts["gt_boundary"])
    boundary = (2*precision*recall/(precision+recall)
                if precision is not None and recall is not None and precision+recall > 0
                else (0. if counts["gt_boundary"] else None))
    return dict(roof_iou=_div(counts["tp"], counts["tp"]+counts["fp"]+counts["fn"]),
                roof_precision=_div(counts["tp"], counts["tp"]+counts["fp"]),
                roof_recall=_div(counts["tp"], counts["tp"]+counts["fn"]),
                roof_boundary_f1_2px=boundary)


def score_prediction(record, experiment, prediction, label):
    """Only called after predictions. No GT tensor ever reaches a model."""
    codes = {0, 1, 2, 255} if record["dataset"] == "BANDON" else {0, 1, 255}
    if label.ndim != 2 or not set(np.unique(label).tolist()) <= codes:
        raise ValueError("Label codes disagree with explicit native schema")
    roof, facade = prediction["roof"], prediction["facade"]
    if roof.shape != label.shape or facade.shape != label.shape or roof.dtype != bool or facade.dtype != bool:
        raise ValueError("Native prediction/label shape or dtype mismatch")
    target = np.where(label == 255, 255, label == 1).astype(np.uint8)
    counts = _roof_counts(roof, target)
    row = {k: record.get(k) for k in ("id", "dataset", "split", "parent_scene", "city", "time_group", "actual_date", "nominal_year")}
    row.update(experiment=experiment, source_group=record.get("source_group") or record.get("parent_scene") or record["id"],
               counts=counts, **_ratios(counts), roof_pixel_MAE=_div(counts["fp"]+counts["fn"], counts["valid_pixels"]),
               signed_roof_area_error_pixels=counts["signed_roof_area_error_pixels"] if counts["valid_pixels"] else None,
               absolute_roof_area_error_pixels=counts["absolute_roof_area_error_pixels"] if counts["valid_pixels"] else None,
               label_status="MEASURED" if counts["valid_pixels"] else "NO_VALID_LABEL_PIXELS",
               area_unit="NATIVE_PIXEL_NOT_METRES", physical_instance_small_roof_recall=None,
               facade_counts=None, facade_iou=None, facade_precision=None, facade_recall=None,
               gt3_prediction4_confusion=None, background_neither_head_metrics=None,
               facade_status="UNTRAINED_RANDOM_AUX_NO_SEMANTIC_BASELINE" if experiment == "E0" else "NO_FACADE_GT")
    if experiment != "E0" and record["dataset"] == "BANDON":
        valid, gt = label != 255, label == 2
        tp, fp, fn = int((facade & gt & valid).sum()), int((facade & ~gt & valid).sum()), int((~facade & gt & valid).sum())
        states = roof.astype(np.int64) + 2 * facade.astype(np.int64)
        confusion = np.bincount((label[valid].astype(np.int64)*4+states[valid]), minlength=12).reshape(3, 4)
        row.update(facade_counts=dict(tp=tp, fp=fp, fn=fn, valid_pixels=int(valid.sum())),
                   facade_iou=_div(tp, tp+fp+fn), facade_precision=_div(tp, tp+fp), facade_recall=_div(tp, tp+fn),
                   facade_status=FACADE_SEMANTICS, gt3_prediction4_confusion=confusion.tolist())
    return row


def aggregate_rows(rows):
    """Micro counts plus per-image area errors; opposite errors never cancel MAE."""
    numeric = ("tp", "fp", "fn", "valid_pixels", "gt_roof_pixels", "predicted_roof_pixels",
               "signed_roof_area_error_pixels", "absolute_roof_area_error_pixels", "pred_boundary",
               "gt_boundary", "matched_pred_boundary", "matched_gt_boundary")
    counts = {key: sum(r["counts"][key] for r in rows) for key in numeric}
    valid_rows = [r for r in rows if r["counts"]["valid_pixels"]]
    facade_rows = [r for r in rows if r["facade_counts"] is not None]
    fc = {key: sum(r["facade_counts"][key] for r in facade_rows) for key in ("tp", "fp", "fn", "valid_pixels")} if facade_rows else None
    confusion = np.sum([r["gt3_prediction4_confusion"] for r in facade_rows], axis=0).tolist() if facade_rows else None
    return dict(images=len(rows), images_with_valid_labels=len(valid_rows), counts=counts, **_ratios(counts),
                roof_pixel_MAE=_div(counts["fp"]+counts["fn"], counts["valid_pixels"]),
                mean_signed_roof_area_error_pixels=_div(sum(r["signed_roof_area_error_pixels"] for r in valid_rows), len(valid_rows)),
                mean_absolute_roof_area_error_pixels=_div(sum(r["absolute_roof_area_error_pixels"] for r in valid_rows), len(valid_rows)),
                facade_images_with_GT=len(facade_rows), facade_counts=fc,
                facade_iou=_div(fc["tp"], fc["tp"]+fc["fp"]+fc["fn"]) if fc else None,
                facade_precision=_div(fc["tp"], fc["tp"]+fc["fp"]) if fc else None,
                facade_recall=_div(fc["tp"], fc["tp"]+fc["fn"]) if fc else None,
                gt3_prediction4_confusion=confusion, physical_instance_small_roof_recall=None,
                area_unit="NATIVE_PIXEL_NOT_METRES")


def grouped_report(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split"], row["experiment"])].append(row)
    result = {}
    for (dataset, role, experiment), items in sorted(groups.items()):
        per = {}
        for dimension in ("source_group", "parent_scene", "city", "time_group"):
            subgroups = defaultdict(list)
            for row in items:
                key = row.get(dimension)
                subgroups[str(key) if key is not None else "UNKNOWN"].append(row)
            per[dimension] = {key: aggregate_rows(value) for key, value in sorted(subgroups.items())}
        result["/".join((dataset, role, experiment))] = dict(summary=aggregate_rows(items), groups=per)
    return result


def evaluate_records(payload, records, models, normalize_rgb, device, output):
    """Streaming implementation seam; run_public_final owns the one-shot gate.

    Cache one decoded RGB, all three prediction pairs, then one native label.
    No dataset-wide pixel arrays; metrics records alone remain in memory.
    """
    from PIL import Image
    output, payload = Path(output), Path(payload)
    rows = []
    for index, record in enumerate(records):
        # Defense in depth even for internal callers: never visit TRAIN/TUNE.
        if (record["dataset"], record["split"]) not in PUBLIC_COUNTS:
            raise ValueError("Only allowlisted PUBLIC evaluation roles may be read")
        event = dict(index=index, id=record["id"], dataset=record["dataset"], split=record["split"])
        image_path, label_path = _path_in(payload, record["image"]), _path_in(payload, record["label"])
        _append(output / "READ_EVENTS.jsonl", dict(event, event="RGB_READ_INTENT", at=_now()))
        rgb = _load_image_once(image_path, record["image_sha256"])
        if list(rgb.shape[:2]) != record["native_size"]:
            raise ValueError("RGB dimensions changed from frozen native metadata")
        predictions, tiles = predict_native(models, rgb, normalize_rgb, device)
        image_key = hashlib.sha256(record["id"].encode()).hexdigest()[:24]
        prediction_files = {}
        for experiment in EXPERIMENTS:
            for head in ("roof", "facade"):
                relative = f"predictions/{experiment}/{image_key}_{head}.png"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    Image.fromarray(predictions[experiment][head].astype(np.uint8)*255).save(stream, format="PNG")
                prediction_files[experiment + "/" + head] = dict(path=relative, sha256=_sha(destination))
        _append(output / "READ_EVENTS.jsonl", dict(event, event="ALL_PREDICTION_ARRAYS_GENERATED_BEFORE_LABEL", at=_now(), tiles=tiles))
        _append(output / "READ_EVENTS.jsonl", dict(event, event="LABEL_READ_INTENT_AFTER_PREDICTION", at=_now()))
        label = _load_image_once(label_path, record["label_sha256"], label=True)
        for experiment in EXPERIMENTS:
            row = score_prediction(record, experiment, predictions[experiment], label)
            row.update(tiles_512_nonoverlapping=tiles, native_shape=list(label.shape),
                       prediction_files={key: value for key, value in prediction_files.items() if key.startswith(experiment + "/")},
                       input_image_sha256=record["image_sha256"], label_sha256=record["label_sha256"])
            _append(output / "PER_IMAGE_METRICS.jsonl", row)
            rows.append(row)
        del rgb, label, predictions
        print(json.dumps(dict(status="PUBLIC_IMAGE_COMPLETED", index=index+1, total=len(records))), flush=True)
    return rows


def _load_runtime(payload):
    import torch
    payload = Path(payload)
    sys.path[:0] = [str(payload / "code/src"), str(payload / "code/tools")]
    spec = importlib.util.spec_from_file_location("_xian_public_frozen_s", payload / "code/tools/xian_cloud_s_train.py")
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    api = trainer._adapter(payload)
    if Path(api.__file__).resolve() != (payload / "code/src/xian_r5/cloud_joint/segmentation.py").resolve():
        raise ValueError("Refuse an adapter cached from a different payload")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261020)
    if device == "cuda":
        torch.cuda.manual_seed_all(20261020)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return trainer, api, device


def run_public_final(payload, release_dir, output):
    """The sole production entry: frozen gate -> exclusive claim -> read/evaluate."""
    payload, release, output = (Path(p).resolve() for p in (payload, release_dir, output))
    if output == payload or payload in output.parents or output in payload.parents:
        raise ValueError("Evaluation output must not overlap the original payload")
    manifest, snapshots = verify_release(payload, release)
    data_path = _path_in(payload, "config/segmentation_data.json")
    data = _read_json(data_path)
    records = select_public_records(data)
    # Resolve only selected public data paths; no TRAIN/TUNE/old-test file scan.
    for row in records:
        for key in ("image", "label"):
            _path_in(payload, row[key])
    snapshots[str(data_path)] = _sha(data_path)
    registration = dict(schema="XIAN_PUBLIC_NATIVE_FINAL_V1", run_id=str(uuid.uuid4()),
        registered_before_any_public_pixels=True, payload=str(payload), release=str(release),
        model_freeze_manifest_sha256=snapshots[str(release / "MODEL_FREEZE_MANIFEST.json")],
        snapshots=snapshots, evaluator_sha256=_sha(Path(__file__)), experiments=list(EXPERIMENTS),
        records=[{k: row.get(k) for k in ("id", "dataset", "split", "image", "label", "image_sha256", "label_sha256")} for row in records],
        role_counts={"/".join(k): v for k, v in PUBLIC_COUNTS.items()}, threshold="INDEPENDENT_SIGMOID_STRICTLY_GT_0.5",
        native_side=512, stride=512, padding="ZERO_RGB_EDGE_ONLY_CROP_BACK_TO_NATIVE",
        boundary_protocol="inner8connected;Chebyshev2px;valid5x5core",
        facade_semantics=FACADE_SEMANTICS, E0_facade_semantics="UNTRAINED_RANDOM_AUX_NO_METRICS",
        confusion_rows=["GT_BG", "GT_ROOF", "GT_FACADE"],
        confusion_columns=["BG_NEITHER_HEAD", "ROOF_ONLY", "FACADE_ONLY", "BOTH"],
        parent_disjoint=True, spatial_independence="UNKNOWN", foundation_pretraining_exposure="UNKNOWN",
        exposed_dev_status="PREVIOUSLY_EXPOSED_NOT_BLIND_TEST", calibration_use="FROZEN_MODEL_REPORT_ONLY_NO_RECALIBRATION",
        E3_scope="RELEASE_S_NATIVE_ONLY_NO_A_WARP", model_release_scope=manifest.get("release_scope"),
        GSD_conversion="NONE_NATIVE_PIXELS_ONLY", physical_instance_small_roof_recall=None,
        weight_selection_after_final=False, threshold_selection_after_final=False, training=False)
    ledger = claim_once(release, output, registration)
    _dump_new(output / "PREREGISTRATION.json", dict(registration, read_ledger=str(ledger)))
    try:
        trainer, api, device = _load_runtime(payload)
        models = {"E0": trainer.make_s_model(payload, None).to(device).eval(),
                  "E1": trainer.make_s_model(payload, release / "S_parent_state.pth").to(device).eval(),
                  "E3": trainer.make_s_model(payload, release / "S_state.pth").to(device).eval()}
        for path, expected in snapshots.items():
            if _sha(path) != expected:
                raise ValueError("Frozen inputs changed during model construction")
        rows = evaluate_records(payload, records, models, api.normalize_rgb, device, output)
        for path, expected in snapshots.items():
            if _sha(path) != expected:
                raise ValueError("Frozen inputs changed during evaluation; do not accept results")
        _dump_new(output / "PUBLIC_FINAL_METRICS.json", dict(status="COMPLETED_SINGLE_PUBLIC_FINAL_PROCESS",
                  images=len(records), model_image_evaluations=len(rows), grouped=grouped_report(rows),
                  protocol="PREREGISTRATION.json", selection_or_training_performed=False,
                  new_public_final_only=True, old_protected_tests_accessed=False,
                  gt_in_forward=False, absolute_survey_accuracy_claim=False))
        artifacts = {p.relative_to(output).as_posix(): _sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
        _dump_new(output / "ARTIFACT_SHA256_MANIFEST.json", artifacts)
        _dump_new(output / "COMPLETE.json", dict(status="COMPLETE", at=_now(), read_ledger=str(ledger),
                                                  images=len(records), model_image_evaluations=len(rows), device=device))
        del models
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        return dict(status="COMPLETE", output=str(output), images=len(records), model_image_evaluations=len(rows))
    except BaseException as exc:
        _dump_new(output / "FAILED_NO_RETRY.json", dict(status="FAILED_OR_PARTIAL_FINAL_ACCESS_CONSUMED",
                   error_type=type(exc).__name__, error=str(exc), at=_now(), read_ledger=str(ledger),
                   automatic_retry_allowed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run_public_final(args.payload, args.release_dir, args.output)), flush=True)


if __name__ == "__main__":
    main()
