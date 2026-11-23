"""Append-only independent checks of S4.14 serialization and V2 regressions."""
import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import unittest

from xian_r5.acceptance import OBJECT_STATUSES, sha256

REPO = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    run=parser.parse_args().run.resolve()
    # Do not re-run or overwrite an existing verification receipt.
    if (run/"POST_RUN_ACCEPTANCE_VERIFICATION.json").exists() or (run/"SUPPLEMENTAL_RECEIPT.json").exists():
        raise ValueError("Verification is append-only; already executed")
    def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
    checks=[]
    def check(label,result):
        if not result:raise ValueError("Acceptance verification failed: "+label)
        checks.append(label)
    receipt=read(run/"FINAL_RECEIPT.json")
    manifest=read(run/"RUN_MANIFEST.json")
    mh=sha256(run/"RUN_MANIFEST.json")
    check("final_receipt_hash",sha256(run/"FINAL_RECEIPT.json")== (run/"FINAL_RECEIPT.sha256").read_text().strip())
    check("manifest_bound",receipt["input_manifest_sha256"]==mh)
    for name,meta in receipt["artifacts"].items():
        check("artifact_"+name,sha256(run/name)==meta["sha256"])
    for name,pin in manifest["inputs"].items():
        check("immutable_input_"+name,sha256(pin["path"])==pin["sha256"])
    shadow=Path(manifest["shadow_run"])
    original={r["observation_id"]:r for r in read(shadow/"FRESH_OBSERVATIONS_RAW.json")["records"]}
    with (run/"BUILDING_APPLICATION_AUDIT.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    check("177_original_IDs",len(rows)==177 and {r["observation_id"] for r in rows}==set(original))
    for r in rows:
        uid=r["observation_id"];o=original[uid]
        check("row_provenance_"+uid,r["input_manifest_sha256"]==mh and r["original_id"]==uid and
              r["original_geometry_sha256"]==o["geometry_sha256"] and
              json.loads(r["source_hashes"])==o["source_hashes"] and
              json.loads(r["source_tile_ids"])==o["source_tile_ids"])
        check("row_unknown_not_zero_"+uid,all(json.loads(r[k]) is None for k in (
            "candidate_vector_m","validated_vector_m","applied_vector_m","uncertainty_m","physical_id")))
        check("row_failure_closed_"+uid,r["application_status"] in OBJECT_STATUSES and
              r["application_status"]!="ACCEPTED_FOR_SHADOW_ALIGNMENT" and
              "UNRESOLVED_BUILDING_PARALLAX" in json.loads(r["unresolved_reasons"]))
    check("zero_qualified_not_vacuous_pass",receipt["QUALIFIED_10M_PIXEL_COUNT"]==0 and
          receipt["QUALIFIED_10M_SUPERVISION_PASS"]=="UNRESOLVED" and receipt["positive_scientific_authorization"] is False)
    check("tasks_not_promoted",receipt["LOCAL_REGISTRATION_PASS"]=="UNRESOLVED" and
          receipt["BUILDING_ALIGNMENT_PASS"]=="UNRESOLVED" and receipt["MOSAIC_CONTINUITY_PASS"]=="NOT_RUN")
    check("no_fake_CHECK_uncertainty",read(run/"LOCAL_FIT_CHECK_REPORT.json")["statistics"]["relative_registration_uncertainty_m"] is None)
    # These modules create synthetic fixtures in temporary directories only.
    sys.path.insert(0,str(REPO/"tests/xian_r5"))
    names=["test_r5_relative_alignment_contract","test_reviewed_image_alignment","test_roof_candidate_alignment"]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
    log=io.StringIO();result=unittest.TextTestRunner(stream=log).run(suite)
    check("original_V2_and_whole_roof_synthetic_regressions",result.wasSuccessful())
    data=dict(status="PASS",completed_at=datetime.now().astimezone().isoformat(),
              parent_receipt_sha256=sha256(run/"FINAL_RECEIPT.json"),checks_passed=len(checks),checks=checks,
              regression_tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
              synthetic_test_modules=names,log=log.getvalue(),protected_test_records_read=False,
              network_replay="NOT_RUN",training="NOT_RUN",fit="NOT_RUN",displacements_applied=0,
              verifier_path=str(Path(__file__).resolve()),verifier_sha256=sha256(__file__))
    def save(name,value):
        with (run/name).open("x",encoding="utf-8") as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
    save("POST_RUN_ACCEPTANCE_VERIFICATION.json",data)
    save("SUPPLEMENTAL_RECEIPT.json",dict(parent_FINAL_RECEIPT_sha256=sha256(run/"FINAL_RECEIPT.json"),
         verifier_sha256=sha256(__file__),artifacts={"POST_RUN_ACCEPTANCE_VERIFICATION.json":sha256(run/"POST_RUN_ACCEPTANCE_VERIFICATION.json")},
         original_receipt_unchanged=True,append_only=True))
    with (run/"SUPPLEMENTAL_RECEIPT.sha256").open("x",encoding="ascii") as stream:stream.write(sha256(run/"SUPPLEMENTAL_RECEIPT.json"))
    print(json.dumps(dict(status="PASS",checks=len(checks),additional_synthetic_tests=result.testsRun)),flush=True)


if __name__=="__main__":main()
