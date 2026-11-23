"""Budget-bound native-roof training. No training or data reads on import.

Public interface: make_s_model(payload_root, checkpoint=None),
run_s(payload_root, output, protocol). See S/TRAINER_REPORT.md for protocol.
Only TRAIN image/label files and TUNE evaluation files may be opened. The full
manifest's other roles are metadata only. All writes stay in a NEW output dir.
"""
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 20261020
SIDE = 512
MIN_EPOCHS = 8
MAX_EPOCHS = 60
TRAIN_DEADLINE = "2026-10-20T21:31:06+08:00"
TOTAL_DEADLINE = "2026-10-20T22:16:06+08:00"
EXPECTED_COUNTS = {("BANDON", "TRAIN"): 1601, ("BONAI", "TRAIN"): 32,
                   ("BANDON", "TUNE"): 59, ("BANDON", "CALIBRATION"): 59,
                   ("BANDON", "FINAL_TEST"): 57,
                   ("BANDON", "EXPOSED_DEV"): 206, ("BONAI", "EXPOSED_DEV"): 12}


def _now():
    return datetime.now(timezone.utc)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _relative(root, relative):
    """Reject traversal, absolute/Windows paths and symlink escapes."""
    relative = str(relative)
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative or ":" in relative:
        raise ValueError("payload path must be confined POSIX-relative: " + relative)
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError("payload path escapes root")
    return result


def _adapter(root):
    # Main Agent may place this tools file under payload/code/tools or tools/.
    candidates = [Path(root) / "code/src", Path(__file__).resolve().parents[1] / "src"]
    for candidate in candidates:
        if (candidate / "xian_r5/cloud_joint/segmentation.py").is_file():
            sys.path.insert(0, str(candidate))
            return importlib.import_module("xian_r5.cloud_joint.segmentation")
    raise FileNotFoundError("pack code/src/xian_r5/cloud_joint/segmentation.py")


def make_s_model(payload_root, checkpoint=None):
    """Return CPU model. Build cfg.model ONLY; verify/strict-load original B0.

    Fixed payload paths: code/RSBuilding/opencd/, config/B0_model.py,
    weights/B0.pth, code/src/xian_r5/cloud_joint/segmentation.py. Optional S
    checkpoint is a trusted locally produced full checkpoint or full state_dict;
    original B0 SHA is still checked. No old dataloaders/runtime are imported.
    Caller controls device, train/eval and AMP. No files are written here.
    """
    root = Path(payload_root).resolve()
    api = _adapter(root)
    sys.path.insert(0, str(_relative(root, "code/RSBuilding")))
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import opencd.models  # noqa: F401 registry only
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from opencd.registry import MODELS
    init_default_scope("opencd")
    cfg = Config.fromfile(str(_relative(root, "config/B0_model.py")))
    model = api.RoofFacadeSegmenter.from_b0(lambda: MODELS.build(cfg.model),
                                           _relative(root, "weights/B0.pth"))
    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = state.get("model", state.get("state_dict", state))
        if state.get("_extra_state", {}).get("initialization_sha256") != api.B0_SHA256:
            raise ValueError("S checkpoint lacks original B0 provenance")
        model.load_state_dict(state, strict=True)
    return model.eval()


class BudgetStop(RuntimeError):
    pass


def _timestamp(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("deadline must have explicit timezone")
    return dt.astimezone(timezone.utc)


class _Budget:
    def __init__(self, root, protocol, origin):
        hours = min(3.0, float(protocol.get("max_s_hours", 3.0)))
        if hours <= 0:
            raise ValueError("max_s_hours must be positive")
        deadlines = [_timestamp(TRAIN_DEADLINE), origin + timedelta(hours=hours),
                     _timestamp(protocol.get("hard_training_deadline", TRAIN_DEADLINE)),
                     _timestamp(protocol.get("total_deadline", TOTAL_DEADLINE)) - timedelta(minutes=45)]
        budget_path = Path(root) / "config/budget.json"
        if budget_path.is_file():
            with budget_path.open(encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("last_new_training_update_deadline"):
                deadlines.append(_timestamp(saved["last_new_training_update_deadline"]))
            if saved.get("hard_deadline"):
                deadlines.append(_timestamp(saved["hard_deadline"]) - timedelta(minutes=45))
        self.deadline = min(deadlines)
        self.monotonic_end = time.monotonic() + max(0, (self.deadline - _now()).total_seconds())

    def check(self):
        if _now() >= self.deadline or time.monotonic() >= self.monotonic_end:
            raise BudgetStop("S stage/global training deadline reached")


def _json_write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _atomic_save(path, state):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def _validate_records(records, production=True):
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate record IDs")
    counts = Counter((r["dataset"], r["split"]) for r in records)
    if production and counts != EXPECTED_COUNTS:
        raise ValueError("manifest role counts differ from frozen 1601+32/59/59/57/206+12")
    parents, payload_roles = defaultdict(set), defaultdict(set)
    for r in records:
        if r["split"] not in {"TRAIN", "TUNE", "CALIBRATION", "FINAL_TEST", "EXPOSED_DEV"}:
            raise ValueError("unknown split")
        if r["dataset"] not in {"BANDON", "BONAI"}:
            raise ValueError("unsupported native label source")
        for field in ("image", "label", "native_size", "parent_scene", "image_sha256", "label_sha256"):
            if field not in r:
                raise ValueError("missing manifest field: " + field)
        if r["dataset"] == "BANDON":
            parents[r["split"]].add(r["parent_scene"])
            if r["split"] in {"TUNE", "CALIBRATION", "FINAL_TEST"} and r.get("prior_trained_parent") is not False:
                raise ValueError("new holdout parent has old/unknown training exposure")
        payload_roles[r["image_sha256"]].add(r["split"])
    roles = ("TRAIN", "TUNE", "CALIBRATION", "FINAL_TEST")
    for i, role in enumerate(roles):
        for other in roles[i + 1:]:
            if parents[role] & parents[other]:
                raise ValueError("BANDON parent crosses frozen roles")
    if production and any(len(parents[k]) != 20 for k in roles[1:]):
        raise ValueError("each new holdout must have exactly20 parent scenes")
    # EXPOSED_DEV is not a selection source and never promoted to independent.
    if any(len(v & set(roles)) > 1 for v in payload_roles.values()):
        raise ValueError("duplicate RGB payload crosses training/selection/holdout roles")
    return counts


def _select_train(records, cap):
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError("max_train_images_per_parent must be None or positive int")
    groups = defaultdict(list)
    for record in records:
        if record["split"] == "TRAIN":
            key = (record["dataset"], record.get("source_group") or record["parent_scene"])
            groups[key].append(record)
    selected, unused = [], []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: _digest([SEED, key, r["id"], r["image_sha256"]]))
        selected.extend(rows if cap is None else rows[:cap])
        unused.extend([] if cap is None else rows[cap:])
    selected.sort(key=lambda r: r["id"])
    summary = {"max_train_images_per_parent": cap, "selection_rule": "FIXED_SHA_NO_LABEL_FILTER",
               "selected_images": len(selected), "selected_parents": len(groups),
               "unused_frames": len(unused), "selected_ids": [r["id"] for r in selected],
               "unused_ids": [r["id"] for r in unused],
               "by_dataset": {name: {"selected_images": sum(r["dataset"] == name for r in selected),
                                      "selected_parents": sum(k[0] == name for k in groups),
                                      "unused_frames": sum(r["dataset"] == name for r in unused)}
                              for name in ("BANDON", "BONAI")},
               "bonai_quota": "SAME_CAP_PER_SOURCE_GROUP_ELSE_PARENT_SCENE",
               "full_1601_bandon_training_claim": cap is None}
    return selected, summary


def _grid(records):
    cells = []
    for index, r in enumerate(records):
        h, w = map(int, r["native_size"])
        if h < 1 or w < 1:
            raise ValueError("invalid native dimensions")
        for y in range(0, h, SIDE):
            for x in range(0, w, SIDE):
                cells.append((index, y, x, f'{r["id"]}:y{y}:x{x}:s{SIDE}'))
    return cells


def _epoch_order(cells, epoch):
    # Shuffle images then each image's full grid: native image cache stays useful.
    groups = defaultdict(list)
    for i, cell in enumerate(cells):
        groups[cell[0]].append(i)
    rng = random.Random(SEED + epoch)
    image_order = sorted(groups)
    rng.shuffle(image_order)
    order = []
    for image in image_order:
        patches = groups[image][:]
        rng.shuffle(patches)
        order.extend(patches)
    return order


class _NativeCrops(Dataset):
    def __init__(self, root, records, api, *, augment=False, verify_sha=True, cache_images=2):
        if any(r["split"] not in {"TRAIN", "TUNE"} for r in records):
            raise ValueError("loader cannot access FINAL/CALIBRATION/EXPOSED_DEV")
        self.root, self.records, self.api = Path(root), records, api
        self.cells = _grid(records)
        self.augment, self.epoch = augment, 0
        self.verify_sha, self.checked = verify_sha, set()
        self.cache, self.cache_images = OrderedDict(), max(1, int(cache_images))

    def image(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        r = self.records[index]
        # Guard again at the actual I/O seam, before resolving/opening files.
        if r["split"] not in {"TRAIN", "TUNE"}:
            raise ValueError("protected split reached image loader")
        ip, lp = _relative(self.root, r["image"]), _relative(self.root, r["label"])
        if self.verify_sha and index not in self.checked:
            if _sha(ip) != r["image_sha256"] or _sha(lp) != r["label_sha256"]:
                raise ValueError("image/label SHA mismatch: " + r["id"])
            self.checked.add(index)
        with Image.open(ip) as image:
            rgb = np.array(image.convert("RGB"))
        with Image.open(lp) as image:
            label = np.array(image)
        if tuple(label.shape) != tuple(r["native_size"]) or rgb.shape != (*label.shape, 3):
            raise ValueError("native image/label dimensions differ from manifest")
        if r["dataset"] == "BANDON":
            if r.get("label_kind") != "BANDON_ROOF_FACADE_0_1_2_255" or not np.isin(label, [0, 1, 2, 255]).all():
                raise ValueError("invalid BANDON roof/facade schema")
        elif r.get("label_kind") != "BONAI_ROOF_0_1_255" or not np.isin(label, [0, 1, 255]).all():
            raise ValueError("BONAI must be explicit native roof, not footprint")
        self.cache[index] = (rgb, label)
        while len(self.cache) > self.cache_images:
            self.cache.popitem(last=False)
        return rgb, label

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        image, y, x, crop_id = self.cells[index]
        rgb, label = self.image(image)
        h, w = min(SIDE, label.shape[0] - y), min(SIDE, label.shape[1] - x)
        im = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
        lab = np.full((SIDE, SIDE), 255, dtype=np.int64)
        im[:h, :w], lab[:h, :w] = rgb[y:y+h, x:x+w], label[y:y+h, x:x+w]
        if self.augment:
            rng = random.Random(_digest([SEED, self.epoch, crop_id]))
            rotation, flip = rng.randrange(4), rng.randrange(2)
            im, lab = np.rot90(im, rotation), np.rot90(lab, rotation)
            if flip:
                im, lab = im[:, ::-1], lab[:, ::-1]
        lab = torch.from_numpy(lab.copy())
        if self.records[image]["dataset"] == "BANDON":
            roof, facade = self.api.bandon_targets(lab)
        else:
            roof, facade = lab, torch.full_like(lab, 255)
        return {"rgb": torch.from_numpy(im.transpose(2, 0, 1).copy()),
                "roof": roof, "facade": facade, "index": index, "crop_id": crop_id}


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device):
    if device.type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None}
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}


class _BufferGuard:
    def __init__(self, model):
        self.initial = {n: b.detach().clone() for n, b in model.named_buffers()}

    def check(self, model):
        current = dict(model.named_buffers())
        if current.keys() != self.initial.keys():
            raise RuntimeError("model buffer set changed")
        # One host synchronization per check, not one per tensor.
        equal = [torch.all(current[n] == b) for n, b in self.initial.items()]
        if equal and not torch.stack(equal).all().item():
            raise RuntimeError("frozen model buffer changed")


def _gradients(model):
    active, missing, forbidden = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:
                forbidden.append(name)
        elif param.grad is not None:
            active.append((name, param))
        else:
            missing.append(name)
    if forbidden:
        raise RuntimeError("frozen parameters received gradient: " + str(forbidden[:5]))
    if not active:
        raise RuntimeError("no trainable parameter received gradient")
    finite = torch.stack([p.grad.isfinite().all() for _, p in active]).all().item()
    return {"finite": finite, "active_tensors": len(active), "missing_tensors": len(missing),
            "missing_names": missing, "frozen_gradient_tensors": 0}


def _optimizer(model, previous=None):
    groups = model.parameter_groups(head_lr=3e-4, backbone_lr=1e-5)
    for group in groups:
        group["base_lr"] = group["lr"]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    if previous is not None:
        # Stage transition preserves head Adam moments and adds new backbone state.
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param in previous.state:
                    optimizer.state[param] = previous.state[param]
    return optimizer


def _schedule(optimizer, state):
    step, warmup, total = state["successful_updates"], state["warmup_steps"], state["total_steps"]
    if step < warmup:
        factor = .1 + .9 * step / max(1, warmup)
    else:
        fraction = min(1., (step - warmup) / max(1, total - warmup))
        factor = .01 + .99 * .5 * (1 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor * state["plateau_multiplier"]


def _seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _metrics(pred, target):
    from scipy import ndimage as ndi
    valid, gt = target != 255, target == 1
    core = ndi.binary_erosion(valid, structure=np.ones((5, 5)), border_value=0)
    gb = gt & ~ndi.binary_erosion(gt, structure=np.ones((3, 3)), border_value=0) & core
    pb = pred & ~ndi.binary_erosion(pred, structure=np.ones((3, 3)), border_value=0) & core
    tp, fp, fn = int((pred & gt & valid).sum()), int((pred & ~gt & valid).sum()), int((~pred & gt & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "valid_pixels": int(valid.sum()),
            "gt_roof_pixels": tp + fn, "predicted_roof_pixels": tp + fp,
            "signed_roof_area_error_pixels": fp - fn,
            "absolute_roof_area_error_pixels": abs(fp - fn),
            "pred_boundary": int(pb.sum()), "gt_boundary": int(gb.sum()),
            "matched_pred_boundary": int((pb & ndi.binary_dilation(gb, structure=np.ones((5, 5)))).sum()),
            "matched_gt_boundary": int((gb & ndi.binary_dilation(pb, structure=np.ones((5, 5)))).sum()),
            "physical_instance_metrics": None}


def _ratios(counts):
    def div(a, b):
        return a / b if b else None
    precision = div(counts["matched_pred_boundary"], counts["pred_boundary"])
    recall = div(counts["matched_gt_boundary"], counts["gt_boundary"])
    boundary = (2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall > 0 else
                (0. if counts["gt_boundary"] else None))
    return {"roof_iou": div(counts["tp"], counts["tp"] + counts["fp"] + counts["fn"]),
            "roof_precision": div(counts["tp"], counts["tp"] + counts["fp"]),
            "roof_recall": div(counts["tp"], counts["tp"] + counts["fn"]),
            "roof_boundary_f1_2px": boundary}


def _facade_metrics(pred, target):
    valid, gt = target != 255, target == 1
    tp, fp, fn = int((pred & gt & valid).sum()), int((pred & ~gt & valid).sum()), int((~pred & gt & valid).sum())
    return {"facade_tp": tp, "facade_fp": fp, "facade_fn": fn,
            "facade_valid_pixels": int(valid.sum())}


def _aggregate_metrics(rows):
    keys = ("tp", "fp", "fn", "valid_pixels", "gt_roof_pixels", "predicted_roof_pixels",
            "signed_roof_area_error_pixels", "absolute_roof_area_error_pixels",
            "pred_boundary", "gt_boundary", "matched_pred_boundary", "matched_gt_boundary",
            "facade_tp", "facade_fp", "facade_fn", "facade_valid_pixels")
    counts = {key: sum(row.get(key, 0) for row in rows) for key in keys}
    n = len(rows)
    ft, ff, fn = (counts[key] for key in ("facade_tp", "facade_fp", "facade_fn"))
    def div(a, b):
        return a / b if b else None
    return {"images": n, "counts": counts, **_ratios(counts),
            "facade_precision": div(ft, ft + ff), "facade_recall": div(ft, ft + fn),
            "facade_iou": div(ft, ft + ff + fn),
            "mean_absolute_roof_area_error_pixels": div(counts["absolute_roof_area_error_pixels"], n),
            "mean_signed_roof_area_error_pixels": div(counts["signed_roof_area_error_pixels"], n),
            "physical_instance_metrics": None}


def _evaluate(model, dataset, api, device, budget, epoch, checkpoint_tick=None, on_image=None):
    if any(r["split"] != "TUNE" or r["dataset"] != "BANDON" for r in dataset.records):
        raise ValueError("selection evaluation restricted to new BANDON TUNE")
    model.eval()
    rows = []
    with torch.no_grad():
        for index, record in enumerate(dataset.records):
            budget.check()
            rgb, label = dataset.image(index)
            pred = np.zeros(label.shape, dtype=bool)
            facade_pred = np.zeros(label.shape, dtype=bool)
            for y in range(0, label.shape[0], SIDE):
                for x in range(0, label.shape[1], SIDE):
                    budget.check()
                    if checkpoint_tick:
                        checkpoint_tick()
                    tile = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
                    h, w = min(SIDE, label.shape[0]-y), min(SIDE, label.shape[1]-x)
                    tile[:h, :w] = rgb[y:y+h, x:x+w]
                    image = torch.from_numpy(tile.transpose(2, 0, 1).copy())[None].to(device)
                    with _autocast(device):
                        outputs = model(api.normalize_rgb(image))
                        z, fz = outputs["roof_logits"], outputs["facade_logits"]
                    if not torch.isfinite(z).all() or not torch.isfinite(fz).all():
                        raise FloatingPointError("nonfinite TUNE prediction")
                    pred[y:y+h, x:x+w] = (z[0, 0, :h, :w].float().sigmoid() > .5).cpu().numpy()
                    facade_pred[y:y+h, x:x+w] = (fz[0, 0, :h, :w].float().sigmoid() > .5).cpu().numpy()
            target, facade_target = api.bandon_targets(torch.from_numpy(label.astype(np.int64)))
            counts = _metrics(pred, target.numpy())
            fc = _facade_metrics(facade_pred, facade_target.numpy())
            ft, ff, fn = (fc[key] for key in ("facade_tp", "facade_fp", "facade_fn"))
            rows.append({"id": record["id"], "parent_scene": record["parent_scene"],
                         "city": record.get("city", "UNKNOWN"), "time_group": record.get("time_group", "UNKNOWN"),
                         **counts, **_ratios(counts), **fc,
                         "facade_precision": ft / (ft + ff) if ft + ff else None,
                         "facade_recall": ft / (ft + fn) if ft + fn else None,
                         "facade_iou": ft / (ft + ff + fn) if ft + ff + fn else None})
            if on_image:
                on_image(rows[-1])
    budget.check()
    parents, city_times = defaultdict(list), defaultdict(list)
    for row in rows:
        parents[row["parent_scene"]].append(row)
        city_times[f'{row["city"]}/{row["time_group"]}'].append(row)
    return {"epoch": epoch, "role": "TUNE", "complete": True, **_aggregate_metrics(rows),
            "parent_scenes": len({r["parent_scene"] for r in rows}),
            "independence": "PARENT_DISJOINT_ONLY_ACTUAL_SPATIAL_OVERLAP_UNKNOWN",
            "per_image": rows,
            "per_parent": {k: _aggregate_metrics(v) for k, v in sorted(parents.items())},
            "per_city_time": {k: _aggregate_metrics(v) for k, v in sorted(city_times.items())},
            "facade_semantics": "BANDON_OFFICIAL_GEOMETRIC_AUXILIARY_NOT_EXACT_VISIBLE_SURFACE_GT",
            "physical_instance_metrics": None, "threshold": "sigmoid>0.5",
            "boundary_protocol": "inner8connected;Chebyshev2px;valid5x5core"}


def _candidate(metric, baseline, best, protocol):
    keys = ("roof_iou", "roof_recall", "roof_boundary_f1_2px")
    if not metric.get("complete") or any(metric.get(k) is None or baseline.get(k) is None for k in keys):
        return False
    return (metric["roof_iou"] >= baseline["roof_iou"] + protocol["iou_min_gain"]
            and metric["roof_recall"] >= baseline["roof_recall"] - protocol["recall_tolerance"]
            and metric["roof_boundary_f1_2px"] >= baseline["roof_boundary_f1_2px"] - protocol["boundary_tolerance"]
            and metric["roof_iou"] >= best["roof_iou"] + protocol["iou_min_gain"])


def _freeze_protocol(protocol):
    p = {"batch_size": 4, "smoke_batch_sizes": [4], "max_epochs": 60,
         "num_workers": 0, "max_s_hours": 3., "warmup_steps": 200,
         "max_train_images_per_parent": None, "facade_source_weight": 1.,
         "iou_min_gain": .001, "recall_tolerance": .01, "boundary_tolerance": 0.,
         "hard_training_deadline": TRAIN_DEADLINE, "total_deadline": TOTAL_DEADLINE,
         "device": "cuda", **dict(protocol)}
    if (isinstance(p["batch_size"], bool) or p["batch_size"] not in (4, 8)
            or not isinstance(p["batch_size"], int)
            or not p["smoke_batch_sizes"] or not set(p["smoke_batch_sizes"]) <= {4, 8}):
        raise ValueError("batch size/smoke candidates must be 4 or8")
    if p["batch_size"] not in p["smoke_batch_sizes"]:
        raise ValueError("selected batch must be included in real smoke")
    if (not isinstance(p["max_epochs"], int) or not MIN_EPOCHS <= p["max_epochs"] <= MAX_EPOCHS
            or not isinstance(p["num_workers"], int) or p["num_workers"] < 0
            or not isinstance(p["warmup_steps"], int) or p["warmup_steps"] < 0):
        raise ValueError("epoch bounds must be8..60 and workers nonnegative")
    if p.get("seed", SEED) != SEED:
        raise ValueError("seed frozen at20261020")
    if not 0 <= float(p["facade_source_weight"]) <= 1:
        raise ValueError("invalid facade source weight")
    for key in ("iou_min_gain", "recall_tolerance", "boundary_tolerance"):
        if not math.isfinite(float(p[key])) or p[key] < 0:
            raise ValueError("candidate thresholds must be finite nonnegative")
    # Resumption may tighten deadlines, never alter the optimization contract.
    keys = ("batch_size", "max_epochs", "warmup_steps", "max_train_images_per_parent",
            "facade_source_weight", "iou_min_gain", "recall_tolerance", "boundary_tolerance")
    p["training_contract_sha256"] = _digest({k: p[k] for k in keys})
    return p


def _smoke(root, dataset, api, device, p, budget, output):
    rows = []
    selected_pass = False
    for size in p["smoke_batch_sizes"]:
        budget.check()
        model = optimizer = scaler = None
        out = losses = reference = restored = rgb = roof = facade = stream = guard = None
        try:
            _seed()
            model = make_s_model(root).to(device).set_stage("heads").train()
            budget.check()
            optimizer = _optimizer(model)
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda", init_scale=256.)
            guard = _BufferGuard(model)
            load_start = time.perf_counter()
            # Prefer a supervised TRAIN crop; an all-IGNORE crop is not a smoke.
            items, inspected = [], 0
            for i in _epoch_order(dataset.cells, 0):
                budget.check()
                item = dataset[i]
                inspected += 1
                if (item["roof"] != 255).any():
                    items.append(item)
                if len(items) == size:
                    break
            if len(items) < size:
                raise RuntimeError("not enough supervised TRAIN crops for selected smoke batch")
            load_seconds = time.perf_counter() - load_start
            rgb = torch.stack([r["rgb"] for r in items]).to(device)
            roof = torch.stack([r["roof"] for r in items]).to(device)
            facade = torch.stack([r["facade"] for r in items]).to(device)
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with _autocast(device):
                out = model(api.normalize_rgb(rgb))
                losses = api.native_segmentation_loss(out, roof, facade_target=facade,
                    facade_source_weight=p["facade_source_weight"], boundary_trusted=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            gradients = _gradients(model)
            if not gradients["finite"]:
                raise FloatingPointError("nonfinite real smoke gradients")
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
            guard.check(model)
            _sync(device)
            seconds = time.perf_counter() - start
            budget.check()
            # No optimizer/scaler update: smoke never contributes a formal step.
            model.eval()
            with torch.no_grad(), _autocast(device):
                reference = model(api.normalize_rgb(rgb[:1]))
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            model.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True), strict=True)
            with torch.no_grad(), _autocast(device):
                restored = model(api.normalize_rgb(rgb[:1]))
            for key in reference:
                torch.testing.assert_close(reference[key], restored[key], rtol=0, atol=0)
            budget.check()
            rows.append({"batch_size": size, "status": "PASS_REAL_FORWARD_BACKWARD_STRICT_RELOAD",
                         "optimizer_updates": 0, "crop_ids": [r["crop_id"] for r in items],
                         "inspected_train_crops": inspected, "load_wait_seconds": load_seconds,
                         "forward_backward_seconds": seconds, "images_per_second": size / seconds,
                         "loss": float(losses["loss"].detach()), "gradients": gradients,
                         "memory": _memory(device), "strict_reload_bitwise_equal": True})
            selected_pass |= size == p["batch_size"]
        except BudgetStop:
            _json_write(output / "smoke.json", {"batches": rows, "formal_updates": 0, "budget_stop": True})
            raise
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            rows.append({"batch_size": size, "status": "CUDA_OOM", "optimizer_updates": 0})
        finally:
            del model, optimizer, scaler, out, losses, reference, restored, rgb, roof, facade, stream, guard
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _json_write(output / "smoke.json", {"batches": rows, "formal_updates": 0,
                                       "chosen_batch_size": p["batch_size"]})
    if not selected_pass:
        raise RuntimeError("selected batch failed smoke; freeze a smaller batch in NEW run before TUNE")


def run_s(payload_root, output, protocol):
    """Train only admitted TRAIN; select on TUNE; return actual, bounded status.

    Refuses existing output. Resume via protocol.resume_from into another NEW
    output; original stage start/deadlines, order/cursor/RNG and Adam state stay
    effective. No automatic all-backbone unfreeze or automatic data/batch shrink.
    Epoch counts are complete native512 grid coverage of explicitly selected
    images, including pure background. Under8 + deadline =>
    BUDGET_STOP_NOT_CONVERGED. No physical-instance accuracy is synthesized.
    """
    called_at = _now()
    root, output = Path(payload_root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside read-only payload")
    output.mkdir(parents=True, exist_ok=False)
    p = _freeze_protocol(protocol)
    result = {"status": "PREPARING", "started_at": called_at.isoformat(),
              "completed_epochs": 0, "successful_updates": 0, "amp_skips": 0,
              "latest": None, "best": None, "training_executed": False,
              "physical_instance_metrics": None, "protected_data_files_opened": 0,
              "calibration_or_exposed_dev_files_opened": 0, "production_admission": "NOT_ASSESSED"}
    model = optimizer = scaler = None
    baseline = best_metric = None
    progress = {"completed_epochs": 0, "cursor": 0, "order": None, "seen": [],
                "roof_valid_pixels": 0, "facade_valid_pixels": 0, "all_ignore_batches": 0,
                "pending_evaluation": 0}
    scheduler = {"successful_updates": 0, "warmup_steps": p["warmup_steps"],
                 "total_steps": 1, "plateau_multiplier": 1., "drops": 0,
                 "no_improvement_evaluations": 0, "evaluations_since_drop": 0}
    attempted_batches = amp_skips = 0
    dataset_sha = selection_sha = None
    origin, smoke_passed, last_checkpoint = called_at, False, time.monotonic()

    def save(name):
        nonlocal last_checkpoint
        if model is None or optimizer is None or not smoke_passed:
            return
        _atomic_save(output / name, {"schema": "XIAN_CLOUD_S_TRAIN_V1",
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": _rng(), "scheduler_equivalent": scheduler,
            "epoch": progress["completed_epochs"], "cursor": progress["cursor"],
            "sampler": progress, "baseline": baseline, "best_metric": best_metric,
            "protocol": p, "dataset_sha256": dataset_sha, "selection_sha256": selection_sha,
            "origin_started_at": origin.isoformat(), "smoke_passed": smoke_passed,
            "attempted_batches": attempted_batches, "amp_skips": amp_skips})
        last_checkpoint = time.monotonic()
        result["latest" if name == "latest.pth" else "best"] = str(output / name)

    def checkpoint_tick():
        if time.monotonic() - last_checkpoint >= 1800:
            save("latest.pth")

    def evaluate_pending():
        nonlocal baseline, best_metric
        epoch = progress.get("pending_evaluation")
        if epoch is None:
            return False
        _append(output / "evaluation_events.jsonl", {"epoch": epoch, "status": "STARTED", "role": "TUNE"})
        try:
            metric = _evaluate(model, tune, api, device, budget, epoch,
                checkpoint_tick=checkpoint_tick,
                on_image=lambda row: _append(output / "tune_images.jsonl", {"epoch": epoch, **row}))
        except BudgetStop:
            _append(output / "evaluation_events.jsonl", {"epoch": epoch, "status": "BUDGET_INTERRUPTED",
                                                        "eligible_for_selection": False})
            raise
        _json_write(output / f"tune_epoch{epoch:04d}.json", metric)
        progress["pending_evaluation"] = None
        _append(output / "evaluation_events.jsonl", {"epoch": epoch, "status": "COMPLETE", "role": "TUNE"})
        if baseline is None:
            baseline = best_metric = metric
            save("best.pth")
        else:
            scheduler["evaluations_since_drop"] += 1
            if _candidate(metric, baseline, best_metric, p):
                best_metric = metric
                scheduler["no_improvement_evaluations"] = 0
                save("best.pth")
            else:
                scheduler["no_improvement_evaluations"] += 1
            if progress["completed_epochs"] >= MIN_EPOCHS and scheduler["no_improvement_evaluations"] >= 8:
                if scheduler["drops"] < 2:
                    scheduler["plateau_multiplier"] *= .5
                    scheduler["drops"] += 1
                    scheduler["evaluations_since_drop"] = 0
                    scheduler["no_improvement_evaluations"] = 0
                elif scheduler["evaluations_since_drop"] >= 4:
                    result["status"] = "OBSERVED_TUNE_PLATEAU"
                    save("latest.pth")
                    return True
        save("latest.pth")
        return False

    try:
        resume = None
        if p.get("resume_from"):
            resume = torch.load(str(p["resume_from"]), map_location="cpu", weights_only=False)
            if resume.get("schema") != "XIAN_CLOUD_S_TRAIN_V1" or not resume.get("smoke_passed"):
                raise ValueError("invalid/incomplete S resume checkpoint")
            if resume["protocol"]["training_contract_sha256"] != p["training_contract_sha256"]:
                raise ValueError("resume changed frozen optimization/selection contract")
            origin = _timestamp(resume["origin_started_at"])
        budget = _Budget(root, p, origin)
        result["effective_training_deadline"] = budget.deadline.isoformat()
        budget.check()
        api = _adapter(root)
        manifest_path = _relative(root, "config/segmentation_data.json")
        dataset_sha = _sha(manifest_path)
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        records = manifest["records"]
        counts = _validate_records(records)
        selected, selection = _select_train(records, p["max_train_images_per_parent"])
        selection_sha = _digest(selection)
        result["selection"] = selection
        if resume and (resume["dataset_sha256"] != dataset_sha or resume["selection_sha256"] != selection_sha):
            raise ValueError("resume data/order selection changed")
        _json_write(output / "protocol.json", {**p, "seed": SEED, "native_side": SIDE,
            "dataset_sha256": dataset_sha, "selection_sha256": selection_sha,
            "effective_training_deadline": budget.deadline.isoformat(),
            "origin_started_at": origin.isoformat(), "freeze_before_tune_scores": True,
            "trainer_sha256": _sha(__file__), "adapter_sha256": _sha(api.__file__)})
        _json_write(output / "selection.json", selection)
        _json_write(output / "pool_roles.json", {"counts": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
            "source_manifest_sha256": dataset_sha,
            "records": [{k: r.get(k) for k in ("id", "dataset", "split", "parent_scene", "prior_trained_parent")}
                        for r in records], "EXPOSED_DEV": "DIAGNOSTIC_ONLY_NOT_USED_HERE"})
        if not selected:
            raise ValueError("empty selected TRAIN pool")
        dataset = _NativeCrops(root, selected, api, augment=True)
        tune = _NativeCrops(root, [r for r in records if r["split"] == "TUNE"], api)
        device = torch.device(p["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for original B0 forward")
        scheduler["total_steps"] = math.ceil(len(dataset) / p["batch_size"]) * p["max_epochs"]
        if not resume:
            _smoke(root, dataset, api, device, p, budget, output)
            smoke_passed = True
            # Mandatory fresh original B0: smoke model never becomes formal model.
            _seed()
            budget.check()
            model = make_s_model(root).to(device).set_stage("heads")
            optimizer = _optimizer(model)
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda", init_scale=256.)
            save("latest.pth")
            budget.check()
        else:
            _seed()
            budget.check()
            model = make_s_model(root, p["resume_from"]).to(device)
            optimizer = _optimizer(model)
            optimizer.load_state_dict(resume["optimizer"])
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda", init_scale=256.)
            scaler.load_state_dict(resume["scaler"])
            scheduler, progress = resume["scheduler_equivalent"], resume["sampler"]
            baseline, best_metric = resume["baseline"], resume["best_metric"]
            attempted_batches, amp_skips = resume["attempted_batches"], resume["amp_skips"]
            smoke_passed = True
            _restore_rng(resume["rng"])
            # Preserve an old best as a reference; do not relabel current latest best.
            result["best"] = str(p.get("resume_best")) if p.get("resume_best") else None
            if baseline is None:
                progress["pending_evaluation"] = 0
            save("latest.pth")
        del resume
        stopped_on_resume_eval = evaluate_pending()
        guard = _BufferGuard(model)
        while not stopped_on_resume_eval and progress["completed_epochs"] < p["max_epochs"]:
            budget.check()
            epoch = progress["completed_epochs"]
            desired = "heads" if epoch < 2 else "last"
            if model.stage != desired:
                model.set_stage(desired)
                optimizer = _optimizer(model, optimizer)
            model.train()
            dataset.epoch = epoch
            if progress["order"] is None:
                progress["order"] = _epoch_order(dataset.cells, epoch)
            if (sorted(progress["order"]) != list(range(len(dataset)))
                    or progress["seen"] != progress["order"][:progress["cursor"]]):
                raise ValueError("invalid resume grid permutation/cursor/coverage")
            loader = DataLoader(dataset, batch_size=p["batch_size"],
                sampler=progress["order"][progress["cursor"]:], num_workers=p["num_workers"],
                drop_last=False, pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(SEED + epoch))
            iterator = iter(loader)
            try:
                while progress["cursor"] < len(dataset):
                    budget.check()
                    start_wait = time.perf_counter()
                    batch = next(iterator)
                    load_wait = time.perf_counter() - start_wait
                    budget.check()
                    start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    rgb, roof, facade = (batch[k].to(device, non_blocking=True) for k in ("rgb", "roof", "facade"))
                    _schedule(optimizer, scheduler)
                    with _autocast(device):
                        out = model(api.normalize_rgb(rgb))
                        losses = api.native_segmentation_loss(out, roof, facade_target=facade,
                            facade_source_weight=p["facade_source_weight"], boundary_trusted=True)
                    status, gradients, norm, before_scale = "ALL_IGNORE_SKIP", None, None, scaler.get_scale()
                    if not losses["skip_update"]:
                        scaler.scale(losses["loss"]).backward()
                        scaler.unscale_(optimizer)  # MUST precede clipping
                        gradients = _gradients(model)
                        if gradients["finite"]:
                            norm = float(torch.nn.utils.clip_grad_norm_(
                                [v for v in model.parameters() if v.requires_grad], 1.))
                            if not math.isfinite(norm):
                                raise FloatingPointError("nonfinite gradient norm before clipping")
                        guard.check(model)
                        budget.check()  # no new update may cross stage/global cutoff
                        if not gradients["finite"] and device.type != "cuda":
                            raise FloatingPointError("nonfinite FP32 gradients")
                        scaler.step(optimizer)  # native AMP overflow skip, no fake update count
                        scaler.update()
                        if scaler.get_scale() < before_scale:
                            status, amp_skips = "AMP_SKIP", amp_skips + 1
                        else:
                            status = "SUCCESSFUL_UPDATE"
                            scheduler["successful_updates"] += 1
                    else:
                        progress["all_ignore_batches"] += 1
                        guard.check(model)
                    indices = batch["index"].tolist()
                    progress["cursor"] += len(indices)
                    progress["seen"].extend(indices)
                    progress["roof_valid_pixels"] += losses["roof_valid_pixels"]
                    progress["facade_valid_pixels"] += losses["facade_valid_pixels"]
                    attempted_batches += 1
                    _sync(device)
                    _append(output / "steps.jsonl", {"timestamp": _now().isoformat(), "epoch": epoch + 1,
                        "cursor": progress["cursor"], "attempted_batch": attempted_batches,
                        "successful_updates": scheduler["successful_updates"], "status": status,
                        "crop_ids": batch["crop_id"], "stage": model.stage,
                        "loss_terms": {k: float(v.detach()) if isinstance(v, torch.Tensor) else v
                            for k, v in losses.items()}, "gradients": gradients, "gradient_norm_before_clip": norm,
                        "clip_norm": 1., "unscaled_before_clip": True, "frozen_buffers_unchanged": True,
                        "amp_scale_before": before_scale, "amp_scale_after": scaler.get_scale(),
                        "lr": [g["lr"] for g in optimizer.param_groups], "memory": _memory(device),
                        "load_wait_seconds": load_wait, "step_seconds": time.perf_counter() - start})
                    if time.monotonic() - last_checkpoint >= 1800:
                        save("latest.pth")
                    del rgb, roof, facade, out, losses
            finally:
                if p["num_workers"] and hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()
                del iterator, loader
            guard.check(model)
            if len(set(progress["seen"])) != len(dataset):
                raise RuntimeError("coverage epoch contains missing/duplicate grid cells")
            coverage = {"epoch": epoch + 1, "complete": True, "selected_images": len(selected),
                        "selected_parents": selection["selected_parents"], "planned_grid_crops": len(dataset),
                        "visited_unique_grid_crops": len(set(progress["seen"])),
                        "visited_crop_ids": [dataset.cells[i][3] for i in progress["seen"]],
                        "roof_valid_pixels": progress["roof_valid_pixels"],
                        "facade_valid_pixels": progress["facade_valid_pixels"],
                        "all_ignore_batches": progress["all_ignore_batches"],
                        "pure_background_crops_kept": True, "full_original_train_pool": p["max_train_images_per_parent"] is None}
            _json_write(output / f"coverage_epoch{epoch+1:04d}.json", coverage)
            progress.update({"completed_epochs": epoch + 1, "cursor": 0, "order": None, "seen": [],
                             "roof_valid_pixels": 0, "facade_valid_pixels": 0, "all_ignore_batches": 0,
                             "pending_evaluation": epoch + 1 if (epoch + 1) % 2 == 0 else None})
            save("latest.pth")
            if scheduler["successful_updates"] == 0:
                result["status"] = "NO_SUPERVISED_UPDATES"
                break
            if evaluate_pending():
                break
        else:
            if not stopped_on_resume_eval:
                result["status"] = "MAX_EPOCHS_REACHED_NOT_PROOF_OF_CONVERGENCE"
    except BudgetStop as exc:
        result["status"] = ("BUDGET_STOP_NOT_CONVERGED" if progress["completed_epochs"] < MIN_EPOCHS
                            else "BUDGET_STOP_MINIMUM_COVERAGE_REACHED")
        result["reason"] = str(exc)
    except KeyboardInterrupt:
        result["status"] = "INTERRUPTED_NOT_CONVERGED"
    except Exception as exc:
        result["status"] = "FAILED_NOT_CONVERGED"
        result["reason"] = type(exc).__name__ + ": " + str(exc)
        _json_write(output / "failure.json", {"error": result["reason"], "traceback": traceback.format_exc()})
    finally:
        result.update({"completed_epochs": progress["completed_epochs"],
                       "successful_updates": scheduler["successful_updates"], "amp_skips": amp_skips,
                       "training_executed": scheduler["successful_updates"] > 0,
                       "incomplete_epoch_cursor": progress["cursor"],
                       "minimum8_complete_epochs_reached": progress["completed_epochs"] >= MIN_EPOCHS,
                       "baseline_tune": baseline, "best_tune": best_metric,
                       "model_converged": "NOT_PROVEN", "ended_at": _now().isoformat()})
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            save("latest.pth")
        except Exception as exc:
            result["checkpoint_save_error"] = type(exc).__name__ + ": " + str(exc)
            result["status"] = "CHECKPOINT_SAVE_FAILED"
        result["best_checkpoint"] = result["best"]
        result["latest_checkpoint"] = result["latest"]
        result["checkpoint"] = result["best"] or result["latest"]
        result["best_is_step0"] = best_metric is not None and best_metric.get("epoch") == 0
        result["accepted_roof_improvement"] = best_metric is not None and best_metric.get("epoch", 0) > 0
        result["checkpoint_kind"] = ("BASELINE_STEP0_NO_IMPROVEMENT" if result["best"] and result["best_is_step0"] else
                                     "BEST_TUNE_GUARDED" if result["best"] else
                                     "LATEST_UNSELECTED" if result["latest"] else None)
        _json_write(output / "result.json", result)
        del model, optimizer, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    with open(args.protocol, encoding="utf-8") as handle:
        protocol = json.load(handle)
    print(json.dumps(run_s(args.payload_root, args.output, protocol), ensure_ascii=False))
