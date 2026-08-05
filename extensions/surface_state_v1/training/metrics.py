from __future__ import annotations

import numpy as np


class Confusion:
    def __init__(self, classes: int):
        self.matrix = np.zeros((classes, classes), dtype=np.int64)

    def update(self, prediction, target, ignore_index: int = 255):
        prediction = np.asarray(prediction).reshape(-1)
        target = np.asarray(target).reshape(-1)
        valid = target != ignore_index
        np.add.at(self.matrix, (target[valid], prediction[valid]), 1)

    def report(self, class_names: list[str]) -> dict:
        per_class, ious, f1s = {}, [], []
        for index, name in enumerate(class_names):
            tp = int(self.matrix[index, index])
            fp = int(self.matrix[:, index].sum() - tp)
            fn = int(self.matrix[index].sum() - tp)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            iou = tp / max(1, tp + fp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            per_class[name] = {"iou": iou, "f1": f1, "precision": precision, "recall": recall, "support": int(self.matrix[index].sum())}
            ious.append(iou)
            f1s.append(f1)
        return {
            "pixel_accuracy": float(np.trace(self.matrix) / max(1, self.matrix.sum())),
            "macro_mIoU": float(np.mean(ious)),
            "macro_F1": float(np.mean(f1s)),
            "per_class": per_class,
            "confusion_matrix": self.matrix.tolist(),
        }


class BinaryCounts:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def update(self, prediction, target):
        prediction = np.asarray(prediction, dtype=bool)
        target = np.asarray(target, dtype=bool)
        self.tp += int((prediction & target).sum())
        self.fp += int((prediction & ~target).sum())
        self.fn += int((~prediction & target).sum())

    def report(self):
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1, "tp": self.tp, "fp": self.fp, "fn": self.fn}
