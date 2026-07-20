from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def _scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    return {
        "precision": precision, "recall": recall, "f1": f1, "iou": iou, "dice": f1,
        "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2,
    }


def _pr_auc(probabilities: list[np.ndarray], labels: list[np.ndarray]) -> float | None:
    if not probabilities:
        return None
    probability = np.concatenate(probabilities)
    truth = np.concatenate(labels).astype(bool)
    if not truth.any():
        return None
    order = np.argsort(-probability)
    truth = truth[order]
    tp = np.cumsum(truth); fp = np.cumsum(~truth)
    recall = tp / truth.sum(); precision = tp / np.maximum(1, tp + fp)
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


@dataclass
class ChangeMetricAccumulator:
    semantic_classes: int = 7
    building_class_id: int = 1
    binary: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    new: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    demolition: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    semantic_confusion: np.ndarray | None = None
    semantic_change_correct: int = 0
    semantic_change_total: int = 0
    probabilities: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    sample_count: int = 0
    binary_sample_count: int = 0
    directional_sample_count: int = 0
    semantic_sample_count: int = 0

    def __post_init__(self):
        if self.semantic_confusion is None:
            self.semantic_confusion = np.zeros((self.semantic_classes, self.semantic_classes), dtype=np.int64)

    @staticmethod
    def _confusion(prediction, target) -> np.ndarray:
        prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
        return np.array([(prediction & target).sum(), (prediction & ~target).sum(), (~prediction & target).sum(), (~prediction & ~target).sum()])

    def update(self, *, binary_probability=None, binary_target=None, new_probability=None, new_target=None,
               demolition_probability=None, demolition_target=None, semantic_t1=None, semantic_t2=None,
               semantic_target_t1=None, semantic_target_t2=None, threshold: float = 0.5):
        self.sample_count += 1
        if binary_target is not None:
            self.binary_sample_count += 1
            self.binary += self._confusion(binary_probability >= threshold, binary_target)
            flat_probability = np.asarray(binary_probability).ravel()
            flat_target = np.asarray(binary_target).ravel()
            stride = max(1, math.ceil(len(flat_probability) / 200_000))
            self.probabilities.append(flat_probability[::stride]); self.labels.append(flat_target[::stride])
        if new_target is not None:
            self.directional_sample_count += 1
            self.new += self._confusion(new_probability >= threshold, new_target)
        if demolition_target is not None:
            self.demolition += self._confusion(demolition_probability >= threshold, demolition_target)
        if semantic_target_t1 is not None and semantic_target_t2 is not None:
            self.semantic_sample_count += 1
            for predicted, target in ((semantic_t1, semantic_target_t1), (semantic_t2, semantic_target_t2)):
                valid = (target >= 0) & (target < self.semantic_classes)
                values = self.semantic_classes * target[valid].astype(np.int64) + predicted[valid].astype(np.int64)
                self.semantic_confusion += np.bincount(values, minlength=self.semantic_classes ** 2).reshape(self.semantic_classes, self.semantic_classes)
            predicted_change = semantic_t1 != semantic_t2
            target_change = semantic_target_t1 != semantic_target_t2
            self.semantic_change_correct += int((predicted_change == target_change).sum())
            self.semantic_change_total += int(target_change.size)

    def result(self) -> dict:
        binary = _scores(*self.binary.tolist()) if self.binary_sample_count else {key: None for key in ("precision","recall","f1","iou","dice","specificity","balanced_accuracy")}
        binary.update(available=bool(self.binary_sample_count), supervised_samples=self.binary_sample_count, pr_auc=_pr_auc(self.probabilities, self.labels))
        new = _scores(*self.new.tolist()) if self.directional_sample_count else {key: None for key in ("precision","recall","f1","iou","dice","specificity","balanced_accuracy")}
        demolition = _scores(*self.demolition.tolist()) if self.directional_sample_count else {key: None for key in ("precision","recall","f1","iou","dice","specificity","balanced_accuracy")}
        new.update(available=bool(self.directional_sample_count), supervised_samples=self.directional_sample_count)
        demolition.update(available=bool(self.directional_sample_count), supervised_samples=self.directional_sample_count)
        matrix = self.semantic_confusion
        ious = []
        for class_id in range(self.semantic_classes):
            tp = matrix[class_id, class_id]
            denominator = matrix[class_id, :].sum() + matrix[:, class_id].sum() - tp
            ious.append(float(tp / denominator) if denominator else None)
        valid = [value for value in ious if value is not None]
        return {
            "sample_count": self.sample_count,
            "binary": binary,
            "directional": {"new": new, "demolition": demolition},
            "semantic": {
                "available": bool(self.semantic_sample_count), "supervised_samples": self.semantic_sample_count,
                "per_class_iou": {str(index): value for index, value in enumerate(ious)},
                "mean_iou": float(np.mean(valid)) if valid and self.semantic_sample_count else None,
                "building_iou": ious[self.building_class_id] if self.semantic_sample_count and self.building_class_id < len(ious) else None,
                "semantic_change_accuracy": self.semantic_change_correct / self.semantic_change_total if self.semantic_change_total else None,
                "confusion_matrix": matrix.tolist(),
            },
        }
