"""Synthetic-only trainer checks. All training I/O is intercepted in memory.

No real payload, weight, TUNE, calibration, development or FINAL file is read.
No tmp data/checkpoints are written: atomic saves serialize into BytesIO.
"""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

_repo = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s_train_under_test", _repo / "tools/xian_cloud_s_train.py")
t = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = t
_spec.loader.exec_module(t)
_spec = importlib.util.spec_from_file_location("s_api_under_test", _repo / "src/xian_r5/cloud_joint/segmentation.py")
api = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = api
_spec.loader.exec_module(api)
torch.set_num_threads(2)


def record(i, split="TRAIN", dataset="BANDON", parent=None, size=(8, 8)):
    return {"id": f"{dataset}_{split}_{i}", "dataset": dataset, "split": split,
            "parent_scene": parent or f"{dataset}/{split}/{i}", "prior_trained_parent": False,
            "image": f"data/{split}/{i}.png", "label": f"data/{split}/{i}_label.png",
            "image_sha256": t._digest([dataset, split, i]), "label_sha256": t._digest([dataset, split, i, "label"]),
            "native_size": list(size), "label_kind": "BANDON_ROOF_FACADE_0_1_2_255" if dataset == "BANDON" else "BONAI_ROOF_0_1_255"}


class ToyS(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 3, 1)
        self.roof_head = nn.Conv2d(3, 1, 1)
        self.facade_aux = nn.Conv2d(3, 1, 1)
        self.frozen = nn.BatchNorm2d(3)
        self.stage = "heads"
        self.set_stage("heads")

    def set_stage(self, stage):
        self.stage = stage
        for p in self.parameters():
            p.grad = None
            p.requires_grad_(True)
        self.backbone.requires_grad_(stage == "last")
        self.frozen.requires_grad_(False)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def parameter_groups(self, head_lr, backbone_lr):
        groups = [{"params": list(self.roof_head.parameters()) + list(self.facade_aux.parameters()), "lr": head_lr}]
        if self.stage == "last":
            groups.insert(0, {"params": list(self.backbone.parameters()), "lr": backbone_lr})
        return groups

    def forward(self, image):
        features = self.frozen(self.backbone(image))
        return {"roof_logits": self.roof_head(features), "facade_logits": self.facade_aux(features)}

    def get_extra_state(self):
        return {"stage": self.stage, "initialization_sha256": api.B0_SHA256}

    def set_extra_state(self, state):
        self.set_stage(state["stage"])


@pytest.fixture
def world(monkeypatch):
    """Small synthetic workflow with real AdamW/loss but no filesystem writes."""
    rows = [record(i) for i in range(12)] + [record(100, "TUNE"), record(101, "FINAL_TEST"),
                                           record(102, "CALIBRATION"), record(103, "EXPOSED_DEV")]
    state = types.SimpleNamespace(records=rows, writes={}, logs={}, checkpoints={}, opened=[],
                                  stop_after=None, stop_eval_epoch=None, budgets=[], builds=[], all_ignore=False)
    original_open, original_load = Path.open, torch.load
    root = Path("synthetic_payload").resolve()
    state.root = root
    state.output = Path("synthetic_new_output").resolve()
    monkeypatch.setattr(t, "SIDE", 8)
    monkeypatch.setattr(t, "_adapter", lambda _: api)
    monkeypatch.setattr(t, "_sha", lambda path: t._digest(str(path)))
    validate = t._validate_records
    monkeypatch.setattr(t, "_validate_records", lambda rows: validate(rows, production=False))
    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(t, "_json_write", lambda path, obj: state.writes.__setitem__(str(path), copy.deepcopy(obj)))
    monkeypatch.setattr(t, "_append", lambda path, obj: state.logs.setdefault(str(path), []).append(copy.deepcopy(obj)))

    def opened(path, *args, **kwargs):
        if str(path) == str(root / "config/segmentation_data.json"):
            return io.StringIO(json.dumps({"records": state.records}))
        raise AssertionError("unexpected/protected file I/O: " + str(path))

    monkeypatch.setattr(Path, "open", opened)

    def saved(path, obj):
        stream = io.BytesIO()
        torch.save(obj, stream)
        stream.seek(0)
        state.checkpoints[str(path)] = original_load(stream, map_location="cpu", weights_only=False)

    monkeypatch.setattr(t, "_atomic_save", saved)

    def loaded(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            return copy.deepcopy(state.checkpoints[str(path)])
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", loaded)

    def make(_root, checkpoint=None):
        model = ToyS().eval()
        if checkpoint:
            model.load_state_dict(state.checkpoints[str(checkpoint)]["model"], strict=True)
        state.builds.append(model)
        return model

    monkeypatch.setattr(t, "make_s_model", make)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def image(self, index):
        r = self.records[index]
        assert r["split"] in {"TRAIN", "TUNE"}
        state.opened.append(r["split"])
        rgb = np.full((*r["native_size"], 3), 125, dtype=np.uint8)
        label = np.zeros(r["native_size"], dtype=np.uint8)
        label[2:6, 2:6] = 1
        if r["dataset"] == "BANDON":
            label[5:7, 1:3] = 2
        if state.all_ignore and r["split"] == "TRAIN":
            label[:] = 255
        return rgb, label

    monkeypatch.setattr(t._NativeCrops, "image", image)

    class Budget:
        def __init__(self, _root, _p, origin):
            self.deadline = origin + timedelta(hours=3)
            state.budgets.append(self)

        def check(self):
            steps = sum(len(v) for k, v in state.logs.items() if k.endswith("steps.jsonl"))
            if state.stop_after is not None and steps >= state.stop_after:
                raise t.BudgetStop("synthetic deadline")
            if state.stop_eval_epoch is not None:
                starts = [r for k, v in state.logs.items() if k.endswith("evaluation_events.jsonl") for r in v
                          if r["status"] == "STARTED"]
                if starts and starts[-1]["epoch"] == state.stop_eval_epoch:
                    raise t.BudgetStop("synthetic evaluation deadline")

    monkeypatch.setattr(t, "_Budget", Budget)
    state.protocol = {"device": "cpu", "batch_size": 4, "smoke_batch_sizes": [4],
                      "max_epochs": 8, "num_workers": 0, "warmup_steps": 2}
    return state


def test_complete_production_manifest_roles_and_sha_selection():
    rows = []
    for (dataset, split), n in t.EXPECTED_COUNTS.items():
        for i in range(n):
            parent = f"{dataset}/{split}/{i % 20}" if split in {"TUNE", "CALIBRATION", "FINAL_TEST"} else f"{dataset}/{split}/{i}"
            rows.append(record(i, split, dataset, parent))
    assert t._validate_records(rows) == t.EXPECTED_COUNTS
    selected, info = t._select_train(rows, None)
    assert len(selected) == 1633 and info["unused_frames"] == 0
    train = [record(i, parent=f"parent{i//3}") for i in range(12)]
    a, summary = t._select_train(train, 1)
    b, _ = t._select_train(list(reversed(train)), 1)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 4 and summary["unused_frames"] == 8
    assert not summary["full_1601_bandon_training_claim"]
    bonai = [record(i, dataset="BONAI") for i in range(4)]
    for row in bonai:
        row["source_group"] = "one_bonai_source"
    assert len(t._select_train(bonai, 1)[0]) == 1
    bad = copy.deepcopy(rows)
    next(r for r in bad if r["split"] == "TUNE")["prior_trained_parent"] = True
    with pytest.raises(ValueError):
        t._validate_records(bad)


def test_grid_full512_no1024cap_deterministic_order_and_background(world, monkeypatch):
    monkeypatch.setattr(t, "SIDE", 512)
    rows = [record(i, size=(2048, 2048)) for i in range(70)]
    cells = t._grid(rows)
    assert len(cells) == 1120 > 1024
    assert len({c[3] for c in cells}) == 1120
    assert {c[1:3] for c in cells[:16]} == {(y, x) for y in range(0, 2048, 512) for x in range(0, 2048, 512)}
    assert sorted(t._epoch_order(cells, 0)) == list(range(1120))
    assert t._epoch_order(cells, 0) != t._epoch_order(cells, 1)
    dataset = t._NativeCrops(world.root, [record(1, size=(5, 7), dataset="BONAI")], api)
    item = dataset[0]
    assert item["roof"].shape == (512, 512)
    assert (item["roof"][5:] == 255).all()
    assert (item["facade"] == 255).all()
    # No positive-area criterion: valid pure background is kept as a grid cell.
    monkeypatch.setattr(t._NativeCrops, "image", lambda *_: (np.zeros((5, 7, 3), np.uint8), np.zeros((5, 7), np.uint8)))
    assert (dataset[0]["roof"] == 0).sum() == 35


def test_protected_splits_rejected_at_loader_seam_and_path_escape():
    for split in ("FINAL_TEST", "CALIBRATION", "EXPOSED_DEV"):
        with pytest.raises(ValueError):
            t._NativeCrops(Path("memory"), [record(1, split)], api)
    for path in ("../secret", "/absolute", "C:/Windows/x", "data\\FINAL/x"):
        with pytest.raises(ValueError):
            t._relative(Path("memory"), path)


def test_candidate_uses_roof_guardrails_not_facade_or_area():
    p = t._freeze_protocol({"smoke_batch_sizes": [4]})
    base = {"complete": True, "roof_iou": .6, "roof_recall": .8, "roof_boundary_f1_2px": .7}
    good = {**base, "roof_iou": .62}
    assert t._candidate(good, base, base, p)
    assert not t._candidate({**good, "roof_recall": .7, "facade_iou": .99}, base, base, p)
    assert not t._candidate({**good, "roof_boundary_f1_2px": .69}, base, base, p)
    assert not t._candidate({**good, "complete": False}, base, base, p)
    assert not t._candidate({**base, "roof_iou": None}, base, base, p)
    assert t._metrics(np.zeros((8, 8), bool), np.full((8, 8), 255))["physical_instance_metrics"] is None


def test_facade_roof_precision_area_and_groups_use_same_forward(world):
    calls = []
    class Prediction(nn.Module):
        def forward(self, image):
            calls.append(image.shape)
            facade = torch.full((1, 1, 8, 8), -10.)
            facade[:, :, 5:7, 1:3] = 10.
            return {"roof_logits": torch.full((1, 1, 8, 8), 10.), "facade_logits": facade}
    row = record(100, "TUNE")
    row.update(city="synthetic_city", time_group="t1")
    dataset = t._NativeCrops(world.root, [row], api)
    budget = t._Budget(world.root, {}, t._now())
    result = t._evaluate(Prediction(), dataset, api, torch.device("cpu"), budget, 0)
    assert len(calls) == 1  # both logits measured in the same forward
    assert result["roof_precision"] == result["roof_iou"] == 15 / 64
    assert result["roof_recall"] == 1.
    assert result["counts"]["facade_tp"] == 4
    assert result["counts"]["facade_fp"] == result["counts"]["facade_fn"] == 0
    assert result["facade_precision"] == result["facade_recall"] == result["facade_iou"] == 1.
    assert result["per_image"][0]["signed_roof_area_error_pixels"] == 49
    assert result["mean_absolute_roof_area_error_pixels"] == 49
    assert result["per_parent"][row["parent_scene"]]["roof_precision"] == 15 / 64
    assert result["per_city_time"]["synthetic_city/t1"]["facade_iou"] == 1.
    assert result["physical_instance_metrics"] is None
    # Aggregate absolute per-image errors before averaging; do not cancel signs.
    over = t._metrics(np.ones((2, 2), bool), np.zeros((2, 2), np.uint8))
    under = t._metrics(np.zeros((2, 2), bool), np.ones((2, 2), np.uint8))
    pooled = t._aggregate_metrics([over, under])
    assert pooled["mean_signed_roof_area_error_pixels"] == 0
    assert pooled["mean_absolute_roof_area_error_pixels"] == 4


def test_budget_uses_earliest_hard_cutoff_and_original_start(monkeypatch):
    now = datetime(2026, 10, 20, 8, 0, tzinfo=timezone.utc)  #16:00 Shanghai
    monkeypatch.setattr(t, "_now", lambda: now)
    monkeypatch.setattr(Path, "is_file", lambda _: False)
    first = t._Budget(Path("memory"), {}, now)
    assert first.deadline == now + timedelta(hours=3)
    resumed = t._Budget(Path("memory"), {"max_s_hours": 2}, now-timedelta(hours=1))
    assert resumed.deadline == now + timedelta(hours=1)
    late = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(t, "_now", lambda: late)
    budget = t._Budget(Path("memory"), {"hard_training_deadline": "2026-10-20T23:00:00+08:00"}, late)
    assert budget.deadline == t._timestamp(t.TRAIN_DEADLINE)
    monkeypatch.setattr(t, "_now", lambda: t._timestamp(t.TRAIN_DEADLINE))
    with pytest.raises(t.BudgetStop):
        budget.check()


def test_smoke_rebuild_formal_complete_coverage_and_stage_transition(world):
    result = t.run_s(world.root, world.output, world.protocol)
    assert result["status"] == "MAX_EPOCHS_REACHED_NOT_PROOF_OF_CONVERGENCE"
    assert result["completed_epochs"] == 8 and result["successful_updates"] == 24
    assert result["checkpoint"] == result["best_checkpoint"]
    assert result["latest_checkpoint"] == result["latest"]
    assert len(world.builds) == 2  # discarded smoke, fresh formal B0
    smoke = world.writes[str(world.output / "smoke.json")]
    assert smoke["formal_updates"] == 0 and smoke["batches"][0]["optimizer_updates"] == 0
    assert smoke["batches"][0]["strict_reload_bitwise_equal"]
    steps = world.logs[str(world.output / "steps.jsonl")]
    assert all(r["stage"] == ("heads" if r["epoch"] <= 2 else "last") for r in steps)
    assert all(r["unscaled_before_clip"] and r["frozen_buffers_unchanged"] for r in steps)
    for epoch in range(1, 9):
        coverage = world.writes[str(world.output / f"coverage_epoch{epoch:04d}.json")]
        assert coverage["planned_grid_crops"] == coverage["visited_unique_grid_crops"] == 12
    assert {r["epoch"] for r in world.logs[str(world.output / "evaluation_events.jsonl")]} == {0, 2, 4, 6, 8}
    assert set(world.opened) == {"TRAIN", "TUNE"}
    latest = world.checkpoints[str(world.output / "latest.pth")]
    assert {"optimizer", "scaler", "rng", "scheduler_equivalent", "sampler", "model"} <= latest.keys()
    assert latest["scheduler_equivalent"]["successful_updates"] == 24
    assert latest["model"]["_extra_state"]["stage"] == "last"


def test_budget_partial_epoch_resume_exact_order_and_model(world):
    world.stop_after = 1
    first = t.run_s(world.root, world.output, world.protocol)
    assert first["status"] == "BUDGET_STOP_NOT_CONVERGED"
    assert first["completed_epochs"] == 0 and first["successful_updates"] == 1
    old = str(world.output / "latest.pth")
    checkpoint = world.checkpoints[old]
    assert checkpoint["cursor"] == 4 and len(checkpoint["sampler"]["seen"]) == 4
    world.stop_after = None
    resumed_output = world.output.with_name("synthetic_resumed_output")
    resumed = t.run_s(world.root, resumed_output, {**world.protocol, "resume_from": old})
    assert resumed["completed_epochs"] == 8 and resumed["successful_updates"] == 24
    assert world.checkpoints[str(resumed_output / "latest.pth")]["origin_started_at"] == checkpoint["origin_started_at"]
    before = world.logs[str(world.output / "steps.jsonl")][0]["crop_ids"]
    after = world.logs[str(resumed_output / "steps.jsonl")][0]["crop_ids"]
    assert not set(before) & set(after)
    full_output = world.output.with_name("synthetic_full_output")
    complete = t.run_s(world.root, full_output, world.protocol)
    assert complete["successful_updates"] == 24
    a = world.checkpoints[str(resumed_output / "latest.pth")]["model"]
    b = world.checkpoints[str(full_output / "latest.pth")]["model"]
    for key in a:
        if isinstance(a[key], torch.Tensor):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_resume_repeats_interrupted_required_validation_before_any_update(world):
    world.stop_eval_epoch = 2
    first = t.run_s(world.root, world.output, world.protocol)
    assert first["completed_epochs"] == 2 and first["successful_updates"] == 6
    old = str(world.output / "latest.pth")
    assert world.checkpoints[old]["sampler"]["pending_evaluation"] == 2
    world.stop_eval_epoch = None
    output = world.output.with_name("synthetic_resume_eval_output")
    second = t.run_s(world.root, output, {**world.protocol, "resume_from": old})
    assert second["completed_epochs"] == 8
    events = world.logs[str(output / "evaluation_events.jsonl")]
    assert events[0]["epoch"] == 2 and events[0]["status"] == "STARTED"


def test_all_ignore_never_optimizer_or_scheduler_update(world, monkeypatch):
    world.all_ignore = True
    monkeypatch.setattr(t, "_smoke", lambda *_: None)  # independently smoke tested above
    result = t.run_s(world.root, world.output, world.protocol)
    assert result["status"] == "NO_SUPERVISED_UPDATES"
    assert result["successful_updates"] == 0
    assert all(r["status"] == "ALL_IGNORE_SKIP" for r in world.logs[str(world.output / "steps.jsonl")])
    checkpoint = world.checkpoints[str(world.output / "latest.pth")]
    assert checkpoint["scheduler_equivalent"]["successful_updates"] == 0
    assert not checkpoint["optimizer"]["state"]


def test_amp_skip_not_success_and_unscale_precedes_clip(world, monkeypatch):
    events = []

    class Scaler:
        def __init__(self, **_):
            self.value, self.steps, self.did_skip = 256., 0, False
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            events.append("unscale")
        def step(self, optimizer):
            if self.steps:
                optimizer.step()
            else:
                self.did_skip = True
            self.steps += 1
        def update(self):
            if self.did_skip:
                self.value /= 2
                self.did_skip = False
        def get_scale(self):
            return self.value
        def state_dict(self):
            return {"value": self.value, "steps": self.steps}
        def load_state_dict(self, state):
            self.value, self.steps = state["value"], state["steps"]

    original = torch.nn.utils.clip_grad_norm_
    def clip(*args, **kwargs):
        assert events[-1] == "unscale"
        events.append("clip")
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.cuda.amp, "GradScaler", Scaler)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    result = t.run_s(world.root, world.output, world.protocol)
    assert result["amp_skips"] == 1 and result["successful_updates"] == 23
    assert world.logs[str(world.output / "steps.jsonl")][0]["status"] == "AMP_SKIP"


def test_frozen_gradient_and_buffer_checks_are_enforced():
    model = ToyS().train()
    guard = t._BufferGuard(model)
    with torch.no_grad():
        model.frozen.running_mean.add_(1)
    with pytest.raises(RuntimeError, match="buffer changed"):
        guard.check(model)
    model.frozen.weight.grad = torch.zeros_like(model.frozen.weight)
    with pytest.raises(RuntimeError, match="frozen parameters"):
        t._gradients(model)


def test_api_make_s_model_builds_only_cfg_model_and_strict_restore(monkeypatch):
    calls = []
    sentinel = types.SimpleNamespace(eval=lambda: sentinel,
        load_state_dict=lambda state, strict: calls.append(("restore", strict)))
    class Wrapped:
        @staticmethod
        def from_b0(factory, path):
            calls.append(("b0", str(path)))
            factory()
            return sentinel
    monkeypatch.setattr(t, "_adapter", lambda _: types.SimpleNamespace(RoofFacadeSegmenter=Wrapped, B0_SHA256=api.B0_SHA256))
    modules = {name: types.ModuleType(name) for name in ("opencd", "opencd.models", "opencd.registry", "mmengine", "mmengine.config", "mmengine.registry")}
    cfg = types.SimpleNamespace(model={"only": "model config"})
    modules["mmengine.config"].Config = types.SimpleNamespace(fromfile=lambda path: cfg)
    modules["mmengine.registry"].init_default_scope = lambda scope: calls.append(("scope", scope))
    modules["opencd.registry"].MODELS = types.SimpleNamespace(build=lambda model: calls.append(("build", model)))
    for key, value in modules.items():
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {"model": {"_extra_state": {"initialization_sha256": api.B0_SHA256}}})
    assert t.make_s_model(Path("synthetic_payload"), "synthetic_checkpoint") is sentinel
    assert ("build", cfg.model) in calls and ("restore", True) in calls
    assert any(name == "b0" and path.endswith("weights\\B0.pth") or name == "b0" and path.endswith("weights/B0.pth") for name, path in calls if name == "b0")


def test_atomic_save_and_json_replacement_are_local_and_serializable(monkeypatch):
    files, replacements = {}, []
    class Buffer(io.BytesIO):
        def close(self):
            files[self.name] = self.getvalue()
        def fileno(self):
            return 10
    class Text(io.StringIO):
        def close(self):
            files[self.name] = self.getvalue()
        def fileno(self):
            return 10
    def opened(path, mode, **_kwargs):
        buf = Buffer() if "b" in mode else Text()
        buf.name = str(path)
        return buf
    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(t.os, "fsync", lambda _: None)
    monkeypatch.setattr(t.os, "replace", lambda old, new: replacements.append((str(old), str(new))))
    t._atomic_save(Path("synthetic_output/latest.pth"), {"x": torch.ones(2)})
    t._json_write(Path("synthetic_output/protocol.json"), {"ok": True})
    assert len(replacements) == 2
    assert all(old == new + ".tmp" for old, new in replacements)
    loaded = torch.load(io.BytesIO(files[str(Path("synthetic_output/latest.pth.tmp"))]), weights_only=True)
    assert loaded["x"].tolist() == [1., 1.]
