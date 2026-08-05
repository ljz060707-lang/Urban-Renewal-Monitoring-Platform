"""Checkpoint eligibility and deterministic selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    eligible: bool
    selection_score: float | None
    reason: str


def candidate_eligibility(*, metrics: dict[str, Any], leakage_count: int, test_pixel_reads: int, source_migration_ok: bool) -> CandidateResult:
    finite = all(value is not None for value in (metrics.get("f1"), metrics.get("iou")))
    collapse = float(metrics.get("predicted_positive_fraction", 0.0)) in {0.0, 1.0}
    eligible = bool(finite and not collapse and leakage_count == 0 and test_pixel_reads == 0 and source_migration_ok)
    reason = "eligible" if eligible else ";".join(filter(None, ["nonfinite" if not finite else "", "class_collapse" if collapse else "", "leakage" if leakage_count else "", "test_read" if test_pixel_reads else "", "source_migration" if not source_migration_ok else ""]))
    return CandidateResult("candidate", eligible, float(metrics.get("iou")) if finite else None, reason)


def select_best_checkpoint(results: list[CandidateResult], *, baseline: CandidateResult) -> CandidateResult:
    eligible = [r for r in results if r.eligible and r.selection_score is not None]
    if not eligible: return baseline
    best = max(eligible, key=lambda r: (float(r.selection_score), r.candidate_id))
    if baseline.selection_score is not None and float(best.selection_score) <= float(baseline.selection_score): return baseline
    return best
