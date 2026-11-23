"""Persist bounded closure tests, including environment errors without hiding them."""
import io
import json
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "src"), str(root / "tests" / "xian_r5")]
output = Path(sys.argv[1])
names = ["test_reviewed_image_alignment", "test_grounded_reconstruction",
         "test_reviewed_absences", "test_reconstruction_qa"]
stream = io.StringIO()
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
receipt = {"test_modules": names, "tests_run": result.testsRun,
           "failures": [(str(t), e) for t, e in result.failures],
           "errors": [(str(t), e) for t, e in result.errors],
           "skipped": [(str(t), reason) for t, reason in result.skipped],
           "all_passed": result.wasSuccessful(),
           "passed_count": result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped),
           "python": sys.executable}
with (output / "CLOSURE_TEST_RECEIPT.json").open("x", encoding="utf-8") as sink:
    json.dump(receipt, sink, ensure_ascii=False, indent=2)
with (output / "CLOSURE_TEST_LOG.txt").open("x", encoding="utf-8") as sink:
    sink.write(stream.getvalue())
print(json.dumps(receipt, ensure_ascii=False))
sys.exit(0 if result.wasSuccessful() else 1)
