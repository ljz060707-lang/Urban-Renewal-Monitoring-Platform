from __future__ import annotations

import numpy as np


def surface_state_metrics(prediction: np.ndarray, target: np.ndarray, *, class_names: tuple[str, ...], ignore_index: int = 255) -> dict:
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = target != ignore_index
    prediction, target = prediction[valid], target[valid]
    if target.size == 0:
        return {"status": "skipped_no_supervision", "per_class": {}, "macro_f1": None, "miou": None, "confusion_matrix": None}
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(matrix, (target, prediction), 1)
    per_class = {}
    f1_values, ious = [], []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        iou = tp / max(1, tp + fp + fn)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "support": int(matrix[index].sum())}
        f1_values.append(f1); ious.append(iou)
    return {"status": "available", "per_class": per_class, "macro_f1": float(np.mean(f1_values)), "miou": float(np.mean(ious)), "confusion_matrix": matrix.tolist()}
