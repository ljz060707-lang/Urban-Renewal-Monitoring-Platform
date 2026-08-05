"""URRS Building Foundation v2 static building candidate components."""

from .model import BuildingFoundationNet
from .data import WHUBuildingDataset, build_whu_manifest, load_label_contract
from .metrics import evaluate_batch_outputs, prediction_distribution_gate, summarize_metrics
from .object_metrics import area_bias_metrics, match_objects, object_detection_metrics, size_stratified_recall

__all__ = [
    "BuildingFoundationNet",
    "WHUBuildingDataset",
    "build_whu_manifest",
    "load_label_contract",
    "evaluate_batch_outputs",
    "prediction_distribution_gate",
    "summarize_metrics",
    "match_objects",
    "object_detection_metrics",
    "size_stratified_recall",
    "area_bias_metrics",
]
