import unittest

from xian_r5.training.temporal_consistency import cycle_consistency, resolve_unknown_cascade


class TemporalConsistencyTests(unittest.TestCase):
    def test_cycle_keeps_formal_transition_ineligible(self) -> None:
        rows = cycle_consistency(
            [{"source_id": "a", "target_id": "b"}],
            [{"source_id": "b", "target_id": "c"}],
            [{"source_id": "a", "target_id": "c"}],
        )
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["formal_transition_eligible"])
        self.assertEqual(rows[0]["status"], "CYCLE_SUPPORT")

    def test_evidence_cascade_does_not_force_unknown(self) -> None:
        resolved = resolve_unknown_cascade({"R2_TBIA": {"scientific_state": "STABLE", "confidence": 0.95}})
        unresolved = resolve_unknown_cascade({"R2_TBIA": {"scientific_state": "STABLE", "confidence": 0.20}})
        self.assertEqual(resolved["scientific_state"], "STABLE")
        self.assertEqual(unresolved["scientific_state"], "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
