from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import fiona  # noqa: F401
except ModuleNotFoundError:
    # These tests exercise deterministic sidecar logic, not GeoPackage I/O.
    sys.modules["fiona"] = types.ModuleType("fiona")

from shapely.geometry import Point, box

from xian_r5.alignment.review_packet import (
    R5_RELATIVE_REVIEW_SCOPE,
    R5_RELATIVE_REVIEW_SCOPE_ID,
    _apply_review_scope_allowlist,
    _scope_review_context_sources,
    _write_tie_pair_review_template,
)
from xian_r5.alignment.reviewed_field import _recompute_relative_ties


RULE = "R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1"
SIDECAR_SCHEMA = "xian_r5.relative_alignment_tie_sidecar.v2"
FIELD_ID = "unit-field"
CELL_ID = "unit-cell"
MOVING_DATE = "2018-04-29"
SOURCE_REF = "unit:candidate/source"


class RelativeAlignmentContractTests(unittest.TestCase):
    def _fixture(self, root: Path, *, fit_count: int = 3,
                 holdout_count: int = 3) -> tuple[Path, dict, dict]:
        evidence = root / "ground_feature_evidence.txt"
        evidence.write_text(
            "Evidence bundle supports curb-corner identity and stated localization bounds.",
            encoding="utf-8")
        reference = root / "reference.tif"
        reference.write_bytes(b"fixed relative-image-grid reference")
        reference_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
        dx, dy = 2.0, -3.0
        support = box(90, 90, 120, 120)
        ties = []
        roles = ["FIT"] * fit_count + ["HOLDOUT"] * holdout_count
        for index, role in enumerate(roles):
            ge_x, ge_y = 101.0 + index, 105.0
            target_residual = 0.6 if role == "FIT" else 0.9
            tie = {
                "tie_id": f"unit-tie-{index + 1}",
                "field_id": FIELD_ID,
                "cell_id": CELL_ID,
                "moving_date": MOVING_DATE,
                "FIT_or_HOLDOUT": role,
                "independence_group": f"independent-feature-{index + 1}",
                "feature_type": "PAVED_CURB_JUNCTION",
                "ge_x": ge_x,
                "ge_y": ge_y,
                "reference_x": ge_x + dx - target_residual,
                "reference_y": ge_y + dy,
                "candidate_dx_m": dx,
                "candidate_dy_m": dy,
                "residual_dx_m": 0.0,
                "residual_dy_m": 0.0,
                "residual_m": 0.0,
                "ge_localization_bound_m": 0.1,
                "reference_localization_bound_m": 0.1,
                "adjusted_bound_m": 0.0,
                "localization_basis": (
                    "Curb-junction edge intersections bracket the GE and reference points; "
                    "each is localized to within 0.1 m by the visible edge geometry."),
                "evidence_path": str(evidence.resolve()),
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "within_exact_support": True,
                "precision_eligible": True,
                "precision_ineligibility_reason": None,
                "feature_class": "STABLE_NONBUILDING_GROUND_FEATURE",
                "stable_nonbuilding_ground_evidence": True,
                "stability_basis": (
                    "Non-building paved curb junction persists across both dated images."),
            }
            self._refresh_tie(tie)
            ties.append(tie)

        sidecar = {
            "schema_version": SIDECAR_SCHEMA,
            "review_contract_amendment": "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_AMENDMENT_V2",
            "uncertainty_rule": RULE,
            "coordinates_crs": "EPSG:32649",
            "field_id": FIELD_ID,
            "cell_id": CELL_ID,
            "moving_date": MOVING_DATE,
            "candidate_source_ref": SOURCE_REF,
            "candidate_dx_m": dx,
            "candidate_dy_m": dy,
            "reference_source": str(reference.resolve()),
            "reference_source_sha256": reference_sha,
            "support_geometry_sha256": hashlib.sha256(support.wkb).hexdigest(),
            "fit_max_adjusted_bound_m": 0.0,
            "holdout_max_adjusted_bound_m": 0.0,
            "relative_registration_uncertainty_m": 0.0,
            "positional_uncertainty_m": 0.0,
            "uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
            "absolute_position_accuracy": "UNKNOWN",
            "ties": ties,
        }
        self._refresh_aggregates(sidecar)
        return evidence, sidecar, {
            "dx": dx, "dy": dy, "support": support, "reference": reference,
        }

    @staticmethod
    def _refresh_tie(tie: dict) -> None:
        dx = tie["candidate_dx_m"]
        dy = tie["candidate_dy_m"]
        tie["residual_dx_m"] = (tie["ge_x"] + dx) - tie["reference_x"]
        tie["residual_dy_m"] = (tie["ge_y"] + dy) - tie["reference_y"]
        tie["residual_m"] = (tie["residual_dx_m"] ** 2 + tie["residual_dy_m"] ** 2) ** 0.5
        ge_bound = tie["ge_localization_bound_m"]
        reference_bound = tie["reference_localization_bound_m"]
        if ge_bound is None or reference_bound is None:
            tie["adjusted_bound_m"] = None
        else:
            tie["adjusted_bound_m"] = tie["residual_m"] + ge_bound + reference_bound

    @classmethod
    def _refresh_aggregates(cls, sidecar: dict) -> None:
        eligible = [tie for tie in sidecar["ties"] if tie["precision_eligible"]]
        fit = [tie["adjusted_bound_m"] for tie in eligible
               if tie["FIT_or_HOLDOUT"] == "FIT"]
        holdout = [tie["adjusted_bound_m"] for tie in eligible
                   if tie["FIT_or_HOLDOUT"] == "HOLDOUT"]
        fit_max = max(fit, default=0.0)
        holdout_max = max(holdout, default=0.0)
        sidecar["fit_max_adjusted_bound_m"] = fit_max
        sidecar["holdout_max_adjusted_bound_m"] = holdout_max
        sidecar["relative_registration_uncertainty_m"] = holdout_max
        sidecar["positional_uncertainty_m"] = holdout_max

    def _validate(self, root: Path, sidecar: dict, values: dict,
                  *, support_geometry=None) -> dict:
        sidecar_path = root / "ties.json"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        evidence_hashes = {str(sidecar_path): sidecar_sha}
        return _recompute_relative_ties(
            sidecar_path,
            field_id=FIELD_ID,
            cell_id=CELL_ID,
            moving_date=MOVING_DATE,
            candidate_source_ref=SOURCE_REF,
            candidate_dx_m=values["dx"],
            candidate_dy_m=values["dy"],
            support_geometry=support_geometry or values["support"],
            reference_path=values["reference"],
            review_layer_path=root / "review.gpkg",
            evidence_hashes=evidence_hashes,
            review_sidecar_digest=sidecar_sha,
        )

    def test_three_fit_three_holdout_recompute_additive_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            result = self._validate(root, sidecar, values)
            self.assertAlmostEqual(result["fit_max_adjusted_bound_m"], 0.8)
            self.assertAlmostEqual(result["holdout_max_adjusted_bound_m"], 1.1)
            self.assertAlmostEqual(result["relative_registration_uncertainty_m"], 1.1)

    def test_fit_maximum_adjusted_bound_over_two_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            tie = sidecar["ties"][0]
            tie["reference_x"] = tie["ge_x"] + values["dx"] - 1.95
            self._refresh_tie(tie)
            self._refresh_aggregates(sidecar)
            with self.assertRaisesRegex(ValueError, "exceeds 2"):
                self._validate(root, sidecar, values)

    def test_holdout_maximum_adjusted_bound_over_two_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            tie = sidecar["ties"][3]
            tie["reference_x"] = tie["ge_x"] + values["dx"] - 1.95
            self._refresh_tie(tie)
            self._refresh_aggregates(sidecar)
            with self.assertRaisesRegex(ValueError, "exceeds 2"):
                self._validate(root, sidecar, values)

    def test_submitted_uncertainty_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            sidecar["relative_registration_uncertainty_m"] += 0.1
            sidecar["positional_uncertainty_m"] += 0.1
            with self.assertRaisesRegex(ValueError, "recomputation mismatch"):
                self._validate(root, sidecar, values)

    def test_candidate_vector_change_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            sidecar["candidate_dx_m"] += 0.25
            with self.assertRaisesRegex(ValueError, "candidate_dx_m recomputation mismatch"):
                self._validate(root, sidecar, values)

    def test_support_geometry_change_fails_by_frozen_geometry_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "support_geometry_sha256 mismatch"):
                self._validate(root, sidecar, values,
                               support_geometry=box(89, 90, 120, 120))

    def test_tie_outside_exact_support_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            tie = sidecar["ties"][0]
            tie["ge_x"] = 200.0
            tie["within_exact_support"] = False
            self._refresh_tie(tie)
            self._refresh_aggregates(sidecar)
            with self.assertRaisesRegex(ValueError, "exact support"):
                self._validate(root, sidecar, values)

    def test_insufficient_fit_ties_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root, fit_count=2, holdout_count=3)
            self._refresh_aggregates(sidecar)
            with self.assertRaisesRegex(ValueError, "at least 3 independent"):
                self._validate(root, sidecar, values)

    def test_insufficient_holdout_ties_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root, fit_count=3, holdout_count=2)
            self._refresh_aggregates(sidecar)
            with self.assertRaisesRegex(ValueError, "at least 3 independent"):
                self._validate(root, sidecar, values)

    def test_non_precision_eligible_tie_does_not_count_toward_three(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            tie = sidecar["ties"][0]
            tie.update({
                "precision_eligible": False,
                "precision_ineligibility_reason": "TIE_NOT_PRECISION_ELIGIBLE",
                "ge_localization_bound_m": None,
                "reference_localization_bound_m": None,
                "adjusted_bound_m": None,
                "localization_basis": (
                    "Native 10 m reference pixels do not resolve distinct curb corners; "
                    "no defensible metre bound can be assigned."),
            })
            self._refresh_aggregates(sidecar)
            with self.assertRaisesRegex(ValueError, "at least 3 independent"):
                self._validate(root, sidecar, values)

    def test_duplicate_independence_groups_do_not_count_as_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            sidecar["ties"][3]["independence_group"] = sidecar["ties"][0]["independence_group"]
            with self.assertRaisesRegex(ValueError, "not independent"):
                self._validate(root, sidecar, values)

    def test_reference_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            sidecar["reference_source_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "reference_source_sha256 mismatch"):
                self._validate(root, sidecar, values)

    def test_tie_evidence_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            sidecar["ties"][0]["evidence_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "evidence SHA-256 mismatch"):
                self._validate(root, sidecar, values)

    def test_localization_basis_requires_explicit_distance_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            sidecar["ties"][0]["localization_basis"] = (
                "Curb-junction edge intersections are crisp and stable in both images.")
            with self.assertRaisesRegex(ValueError, "explicit numeric metre range"):
                self._validate(root, sidecar, values)

    def test_eight_of_nine_scope_fails_closed(self):
        rows = self._scope_rows()
        with self.assertRaisesRegex(ValueError, "matched 0"):
            _apply_review_scope_allowlist(
                rows[:-1], [], R5_RELATIVE_REVIEW_SCOPE_ID)

    def test_ten_matching_rows_for_nine_scope_fails_closed(self):
        rows = self._scope_rows()
        rows.append({"properties": dict(rows[0]["properties"])})
        with self.assertRaisesRegex(ValueError, "matched 2"):
            _apply_review_scope_allowlist(
                rows, [], R5_RELATIVE_REVIEW_SCOPE_ID)

    def test_exact_nine_scope_selects_only_allowlisted_rows(self):
        rows = self._scope_rows()
        rows.extend({"properties": {
            "field_id": f"outside-{index}",
            "cell_id": f"outside-cell-{index}",
            "moving_date": "2018-04-29",
        }} for index in range(9))
        selected, relative, manifest = _apply_review_scope_allowlist(
            rows, [{"properties": {"cell_id": "relative-diagnostic"}}],
            R5_RELATIVE_REVIEW_SCOPE_ID)
        selected_keys = [(row["properties"]["cell_id"],
                          row["properties"]["moving_date"]) for row in selected]
        self.assertEqual(len(selected), 9)
        self.assertEqual(selected_keys, list(R5_RELATIVE_REVIEW_SCOPE))
        self.assertEqual(relative, [])
        self.assertEqual(manifest["expected_date_cell_count"], 9)
        self.assertTrue(manifest["exact_set_match"])

    def test_relative_semantics_keep_absolute_accuracy_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sidecar, values = self._fixture(root)
            result = self._validate(root, sidecar, values)
            self.assertEqual(result["positional_uncertainty_m"],
                             result["relative_registration_uncertainty_m"])
            self.assertEqual(result["uncertainty_semantics"],
                             "RELATIVE_IMAGE_GRID_ONLY")
            self.assertEqual(result["absolute_position_accuracy"], "UNKNOWN")

    def test_scoped_context_keeps_only_allowed_date_level_base_grid(self):
        contexts = _scope_review_context_sources([
            {"kind": "BASE_GRID_CONTEXT", "dates": "2018-04-29"},
            {"kind": "BASE_GRID_CONTEXT", "dates": "2020-01-01"},
            {"kind": "RELATIVE_MATCH_CONTEXT", "dates": "2018-04-29;2018-11-01"},
        ], R5_RELATIVE_REVIEW_SCOPE_ID)
        self.assertEqual(contexts, [
            {"kind": "BASE_GRID_CONTEXT", "dates": "2018-04-29"},
        ])

    def test_blank_tie_template_has_no_synthetic_review_evidence(self):
        rows = [{"properties": {
            "field_id": f"field-{index}",
            "cell_id": cell_id,
            "moving_date": moving_date,
            "reference_source": "fixed-reference.tif",
            "reference_source_sha256": "a" * 64,
            "correction_dx_m": 1.0,
            "correction_dy_m": -1.0,
        }} for index, (cell_id, moving_date) in enumerate(R5_RELATIVE_REVIEW_SCOPE)]
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "blank_ties.csv"
            self.assertEqual(_write_tie_pair_review_template(template_path, rows), 54)
            with template_path.open(encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 54)
            self.assertNotIn("reviewer_id", records[0])
            self.assertNotIn("reviewed_at", records[0])
            self.assertTrue(all(
                not row["tie_id"] and not row["feature_type"]
                and not row["precision_eligible"] and not row["ge_x"]
                and not row["evidence_sha256"]
                for row in records
            ))

    @staticmethod
    def _scope_rows() -> list[dict]:
        return [{"properties": {
            "field_id": f"field-{index}",
            "cell_id": cell_id,
            "moving_date": moving_date,
        }} for index, (cell_id, moving_date) in enumerate(R5_RELATIVE_REVIEW_SCOPE)]


if __name__ == "__main__":
    unittest.main()
