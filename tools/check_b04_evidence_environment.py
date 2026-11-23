"""Audit base and isolated imports without mutating the base runtime."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\19_ground_registration_evidence\B04_20260929_01\14_receipts")
BASE = Path(r"C:\Users\ljz52\.conda\envs\xian-r5-fixedcase-20260925")
ISOLATED = Path(r"C:\Users\ljz52\.conda\envs\xian-r5-evidence-test-20260929")
env = os.environ.copy()
env["PROJ_DATA"] = str(BASE / "Library/share/proj")
env["GDAL_DATA"] = str(BASE / "Library/share/gdal")
env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), str(ROOT / "tests/xian_r5")])
receipt = {"base_environment_modified": False, "system_python_modified": False,
           "isolation": "NEW_VENV_SYSTEM_SITE_PACKAGES_READ_ONLY_BASE_WITH_LOCAL_OPENCV_OVERRIDE",
           "base": str(BASE), "isolated": str(ISOLATED),
           "opencv_override": "opencv-python-headless==4.10.0.84",
           "process_only_data_paths": {k:env[k] for k in ["PROJ_DATA", "GDAL_DATA"]},
           "initial_isolated_test_error": "PROJ_DATA_NOT_DISCOVERED_UNTIL_PROCESS_PATH_SET",
           "environments": {}}
for name, executable in [("base", BASE / "python.exe"), ("isolated", ISOLATED / "Scripts/python.exe")]:
    results = []
    for module in ["sys", "numpy", "cv2", "rasterio", "fiona", "shapely", "pyproj"]:
        code = (f"import {module} as m; import json,sys; print(json.dumps({{'module':'{module}', 'version':getattr(m,'__version__',sys.version if '{module}'=='sys' else None),'path':getattr(m,'__file__',sys.executable)}}))")
        p = subprocess.run([str(executable), "-c", code], env=env, capture_output=True, timeout=30)
        results.append({"module": module, "returncode": p.returncode,
                        "stdout": p.stdout.decode(errors="replace"), "stderr": p.stderr.decode(errors="replace")})
    receipt["environments"][name] = {"imports": results, "all_imports_pass": all(r["returncode"] == 0 for r in results)}
command = [str(ISOLATED / "Scripts/python.exe"), "-m", "unittest", "test_grounded_reconstruction.GroundedReconstructionTests.test_masked_registration_is_candidate_without_ground_review", "-v"]
p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=120)
receipt["previously_failed_test"] = {"command": command, "returncode": p.returncode, "stdout": p.stdout.decode(errors="replace"), "stderr": p.stderr.decode(errors="replace")}
receipt["resolved_in_isolated_environment"] = receipt["environments"]["isolated"]["all_imports_pass"] and p.returncode == 0
with (OUT / "ENVIRONMENT_IMPORT_RECEIPT.json").open("x", encoding="utf-8") as sink:
    json.dump(receipt, sink, indent=2, ensure_ascii=False)
print(json.dumps(receipt, ensure_ascii=True))
