"""Track-guided merged-mask resolver contract.

The first implementation is a safe resolver shell: it can flag 1:N/N:1
relations and accepts an injected RBF-SAM segmenter, but it never pretends a
missing pretrained model produced a footprint.  This makes the RBF-SAM stage
reproducible and prevents a diagnostic mask from becoming final geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class MergeResolution:
    current_id: str
    prior_ids: list[str]
    status: str
    resolver_stage: str
    evidence_source: str
    forced_resolution: bool = False
    child_masks: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def resolve_merged_mask(
    current_id: str,
    prior_ids: Iterable[str],
    *,
    current_area_m2: float | None = None,
    prior_area_m2: Iterable[float] | None = None,
    segmenter: Callable[..., list[Any]] | None = None,
    image: Any = None,
    prompts: list[Any] | None = None,
) -> MergeResolution:
    prior = list(prior_ids)
    areas = list(prior_area_m2 or [])
    suspect_reason = len(prior) > 1
    if current_area_m2 is not None and areas and current_area_m2 > 1.6 * sum(areas):
        suspect_reason = True
    if not suspect_reason:
        return MergeResolution(current_id, prior, "NOT_TRIGGERED", "R0_GEOMETRY", "no_merge_evidence")
    if segmenter is None:
        return MergeResolution(
            current_id,
            prior,
            "SEGMENTATION_MERGE_SUSPECT",
            "R3_MERGED_MASK_RBF_SAM_BLOCKED",
            "TBIA_RELATION_GRAPH;NO_PRETRAINED_RBF_SAM_RUNTIME",
            notes=["No mask was fabricated; current polygon remains B0-owned geometry."],
        )
    try:
        masks = list(segmenter(image=image, prompts=prompts or [], prior_ids=prior))
    except Exception as exc:  # pragma: no cover - model-runtime dependent
        return MergeResolution(current_id, prior, "SEGMENTER_FAILED", "R3_MERGED_MASK_RBF_SAM_FAILED", "RBF_SAM_RUNTIME", notes=[repr(exc)])
    if len(masks) == len(prior):
        return MergeResolution(current_id, prior, "INSTANCE_MERGE_ERROR_CANDIDATE", "R3_MERGED_MASK_RBF_SAM", "RBF_SAM_TRACK_GUIDED_LOCAL_RESEGMENTATION", child_masks=masks)
    return MergeResolution(current_id, prior, "SEGMENTER_OUTPUT_INCOMPLETE", "R3_MERGED_MASK_RBF_SAM", "RBF_SAM_TRACK_GUIDED_LOCAL_RESEGMENTATION", child_masks=masks)
