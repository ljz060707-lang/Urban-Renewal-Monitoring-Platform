from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FusionResult:
    outputs: dict[str, np.ndarray]
    pending_outputs: dict[str, str]
    warnings: list[str]


def fuse_teacher_outputs(
    *,
    any_change: np.ndarray | None,
    new_building: np.ndarray | None,
    demolition: np.ndarray | None,
    semantic_transition: np.ndarray | None = None,
    building_t1: np.ndarray | None = None,
    building_t2: np.ndarray | None = None,
) -> FusionResult:
    arrays = [value for value in (any_change, new_building, demolition) if value is not None]
    if arrays and any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("Teacher probability shapes differ; geospatial alignment is required")
    outputs: dict[str, np.ndarray] = {}
    pending: dict[str, str] = {}
    warnings: list[str] = []
    if any_change is None:
        pending["any_change_probability"] = "pending_teacher: dsifn"
    else:
        outputs["any_change_probability"] = np.asarray(any_change, dtype=np.float32)
    for name, value in (("new_building", new_building), ("demolition", demolition)):
        if value is None:
            pending[f"{name}_evidence"] = "pending_teacher: s2looking"
            pending[f"{name}_candidate"] = "pending_teacher"
            continue
        evidence = np.asarray(value, dtype=np.float32)
        outputs[f"{name}_evidence"] = evidence
        outputs[f"{name}_candidate"] = evidence if any_change is None else evidence * np.asarray(any_change)
    if semantic_transition is None:
        pending["semantic_transition"] = "pending_teacher: second"
        pending["construction_candidate"] = "pending_teacher: second"
    else:
        outputs["semantic_transition"] = np.asarray(semantic_transition)
    if building_t1 is None or building_t2 is None:
        pending["building_geometry"] = "pending_teacher: rsbuilding"
    else:
        outputs["building_t1_probability"] = np.asarray(building_t1, dtype=np.float32)
        outputs["building_t2_probability"] = np.asarray(building_t2, dtype=np.float32)
        if "new_building_candidate" in outputs:
            outputs["new_building_candidate"] *= outputs["building_t2_probability"]
        if "demolition_candidate" in outputs:
            outputs["demolition_candidate"] *= outputs["building_t1_probability"]
    return FusionResult(outputs, pending, warnings)

