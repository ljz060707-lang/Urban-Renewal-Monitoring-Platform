from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.run_xian_base_mosaic_pair_diagnostic import _validate_fixed_case_diagnostic


class BaseMosaicPairDiagnosticTests(unittest.TestCase):
    def test_accepts_named_bounded_b03_case_receipt(self) -> None:
        self.assertEqual(_validate_fixed_case_diagnostic({
            "scope": "FIXED_CASE_LOCAL_ONLY",
            "case_id": "B03-TEX04",
        }), "B03-TEX04")

    def test_rejects_citywide_or_anonymous_receipts(self) -> None:
        for receipt in (
            {"scope": "CITYWIDE", "case_id": "B03-TEX04"},
            {"scope": "FIXED_CASE_LOCAL_ONLY", "case_id": ""},
            {"scope": "FIXED_CASE_LOCAL_ONLY", "case_id": "F:/all-cities"},
        ):
            with self.subTest(receipt=receipt), self.assertRaisesRegex(ValueError, "fixed-case"):
                _validate_fixed_case_diagnostic(receipt)


if __name__ == "__main__":
    unittest.main()
