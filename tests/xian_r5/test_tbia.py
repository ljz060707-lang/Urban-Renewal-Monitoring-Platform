import sys
from pathlib import Path
import unittest

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.reconstruction.core import Observation, _tbia_observation, reconstruct
from xian_r5.tracking.tbia import BuildingObservation, TBIAConfig, TBIAPrototype, TrackQueryEncoder


class TBIATests(unittest.TestCase):
    def test_missing_groups_are_not_zero_imputed(self) -> None:
        observation = BuildingObservation("a", "2018", 0.0, 0.0, 100.0)
        encoded = TrackQueryEncoder().encode(observation, [observation])
        self.assertFalse(encoded["groups_available"]["semantic"])
        self.assertFalse(encoded["groups_available"]["boundary"])
        self.assertFalse(encoded["groups_available"]["offset"])
        self.assertFalse(encoded["groups_available"]["temporal"])
        self.assertTrue(encoded["groups_available"]["shape"])

    def test_explicit_zero_feature_is_available(self) -> None:
        observation = BuildingObservation(
            "a",
            "2018",
            0.0,
            0.0,
            100.0,
            metadata={"alignment_dx": 0.0, "alignment_dy": 0.0},
        )
        encoded = TrackQueryEncoder().encode(observation, [observation])
        self.assertTrue(encoded["groups_available"]["offset"])

    def test_unmeasured_alignment_uncertainty_stays_unknown_in_reconstruction_adapter(self) -> None:
        source = Observation("a", "2018", "2018-04-29", None, None,
                             box(0, 0, 10, 10), "source")

        adapted = _tbia_observation(source)
        encoded = TrackQueryEncoder().encode(adapted, [adapted])

        self.assertIsNone(adapted.alignment_uncertainty)
        self.assertEqual(adapted.metadata["alignment_uncertainty_source"], "UNKNOWN")
        self.assertFalse(encoded["groups_available"]["offset"])

    def test_unknown_sigma_uses_search_buffer_without_becoming_a_measurement(self) -> None:
        matcher = TBIAPrototype(TBIAConfig(
            base_radius_m=10.0, alpha_uncertainty=2.0,
            unknown_alignment_search_buffer_m=15.0,
            beta_offset=0.0, gamma_object_scale=0.0))
        unknown = BuildingObservation("unknown", "2018", 0, 0, 100)
        measured = BuildingObservation("measured", "2018", 0, 0, 100,
                                       alignment_uncertainty=8.0)

        self.assertIsNone(unknown.alignment_uncertainty)
        self.assertEqual(matcher.search_radius(unknown), 25.0)
        self.assertEqual(matcher.search_radius(measured), 26.0)

    def test_pair_search_radius_uses_both_dates_uncertainty(self) -> None:
        matcher = TBIAPrototype(TBIAConfig(
            base_radius_m=10.0, alpha_uncertainty=2.0,
            unknown_alignment_search_buffer_m=15.0,
            beta_offset=0.0, gamma_object_scale=0.0,
            min_affinity=0.30))
        previous = BuildingObservation("a", "2018", 0, 0, 100,
                                       alignment_uncertainty=0.0)
        current = BuildingObservation("b", "2019", 20, 0, 100,
                                      alignment_uncertainty=None)

        decision = matcher.match([previous], [current])[0]

        self.assertEqual(matcher.search_radius(previous), 10.0)
        self.assertEqual(decision.search_radius_m, 25.0)
        self.assertEqual(decision.candidate_search_radii_m, {"b": 25.0})
        self.assertEqual(decision.candidate_matches[0][0], "b")

    def test_identity_edge_records_candidate_pair_search_radius(self) -> None:
        previous = Observation("a", "2018", "2018-04-29", None, None,
                               box(0, 0, 10, 10), "source",
                               registration_sigma_m=0.0)
        current = Observation("b", "2019", "2018-11-01", None, None,
                              box(30, 0, 40, 10), "source")

        result = reconstruct([previous, current])

        self.assertEqual(len(result.candidate_edges), 1)
        self.assertEqual(result.candidate_edges[0]["to_uid"], "b")
        self.assertEqual(result.candidate_edges[0]["search_radius_m"], 45.0)

    def test_mapping_preserves_missing_or_invalid_alignment_uncertainty(self) -> None:
        base = {"object_id": "a", "epoch": "2018", "x": 0, "y": 0, "area_m2": 100}

        self.assertIsNone(BuildingObservation.from_mapping(base).alignment_uncertainty)
        invalid = BuildingObservation.from_mapping(
            {**base, "alignment_uncertainty": float("nan")})
        self.assertIsNone(invalid.alignment_uncertainty)
        self.assertFalse(TrackQueryEncoder().encode(invalid, [invalid])["groups_available"]["offset"])

    def test_prototype_returns_identity_decision_and_hard_negative(self) -> None:
        previous = BuildingObservation("a", "2018", 0.0, 0.0, 100.0, semantic=(1.0, 0.0))
        current = BuildingObservation("b", "2019", 2.0, 1.0, 102.0, semantic=(1.0, 0.0))
        decision = TBIAPrototype().match([previous], [current])[0]
        self.assertEqual(decision.current_id, "b")
        self.assertIn("b", decision.hard_negative_ids)
        self.assertGreaterEqual(decision.affinity, 0.45)
        self.assertEqual(decision.candidate_matches, [("b", decision.affinity)])

    def test_prototype_retains_all_above_threshold_candidates(self) -> None:
        previous = BuildingObservation("a", "2018", 0.0, 0.0, 100.0)
        current = [BuildingObservation("b", "2019", 0.0, 0.0, 100.0),
                   BuildingObservation("c", "2019", 0.0, 0.0, 100.0)]

        decision = TBIAPrototype().match([previous], current)[0]

        self.assertEqual(decision.current_id, "b")
        self.assertEqual([uid for uid, _ in decision.candidate_matches], ["b", "c"])
        self.assertTrue(all(score >= 0.45 for _, score in decision.candidate_matches))


if __name__ == "__main__":
    unittest.main()
