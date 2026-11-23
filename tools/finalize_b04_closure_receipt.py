"""Add final test limitations and delivery inventory to this new closure only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction\B04_TQA10_REVIEWED_REGISTRATION_CLOSURE_20260929_02")
tests = json.loads((out / "CLOSURE_TEST_RECEIPT.json").read_text(encoding="utf-8"))
qa = json.loads((out / "ENGINEERING_QA.json").read_text(encoding="utf-8"))
assert qa["structural_qa"] == "PASS"
note = ("\n\n## 最终检查与限制\n\n"
        f"工程QA：PASS，{qa['raster_count']}个TIF，错误清单为空。1m 240×320、10m 24×32，同CRS/原点/边界，精确10×10嵌套。\n\n"
        f"必要测试：运行{tests['tests_run']}项，通过{tests['passed_count']}项，跳过{len(tests['skipped'])}项，失败{len(tests['failures'])}项，错误{len(tests['errors'])}项；详见CLOSURE_TEST_RECEIPT.json和CLOSURE_TEST_LOG.txt。不把环境错误记为通过。\n\n"
        "_01为本轮首次QA失败的保留产物（status产品清单未同步、未解决变化背景遗漏），最终交付为_02；未覆盖历史目录。\n\n"
        "仅完成文档资格审核，未完成任何人工FIT/HOLDOUT点对审核，不能宣称9个P1已人工验真。此次不生成虚构tie/evidence PNG。\n\n"
        "RBF-SAM未运行；无checkpoint。无新候选搜索、无全市运行，停止于GE_TO_SENTINEL_REGISTRATION_EVIDENCE_INSUFFICIENT。\n")
for test, error in tests["errors"]:
    note += f"\n测试环境错误：{test}\n\n```text\n{error}\n```\n"
report = out / "B04_REVIEWED_REGISTRATION_CLOSURE_REPORT.md"
if "## 最终检查与限制" in report.read_text(encoding="utf-8"):
    raise FileExistsError("Final notes already added")
with report.open("a", encoding="utf-8") as sink:
    sink.write(note)
paths = [root / path for path in [
    "src/xian_r5/alignment/reviewed_field.py", "src/xian_r5/reconstruction/products.py",
    "tools/run_xian_r5_grounded_reconstruction.py", "tools/qa_xian_r5_grounded_reconstruction.py",
    "tools/close_b04_reviewed_registration.py", "tools/check_b04_closure_tests.py",
    "tests/xian_r5/test_reviewed_image_alignment.py"]]
inventory = {"code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
             "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
             "historical_outputs_overwritten": False, "citywide_run": False}
with (out / "DELIVERY_INVENTORY.json").open("x", encoding="utf-8") as sink:
    json.dump(inventory, sink, ensure_ascii=False, indent=2)
print(json.dumps({"qa": qa["structural_qa"], "tests": tests, "report": str(report)}, ensure_ascii=False))
