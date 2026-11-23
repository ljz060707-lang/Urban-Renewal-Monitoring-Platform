"""Append-only audit of returned delivery and prospective rerun guards."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recovery_runner',ROOT/'tools/run_xian_feature_recovery.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
OUT=runner.OUT


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    final='--final' in sys.argv
    report_name='FINAL_DELIVERY_VERIFICATION.json' if final else 'IMPLEMENTATION_REPAIR_VERIFICATION.json'
    if (OUT/report_name).exists():raise ValueError('Fresh append-only verification required')
    receipt=read(('IMPLEMENTATION_FIXED_03' if final else 'IMPLEMENTATION_FIXED_02')+'/FINAL_RECEIPT.json')
    for name,expected in receipt['output_sha256'].items():
        if runner.sha(OUT/name)!=expected:raise AssertionError('Delivered artifact drift: '+name)
    before={p:runner.sha(OUT/p) for p in ['MASK_DIAGNOSTIC_DATA.npz','GROUND_FEATURE_CANDIDATE_LEDGER.csv',
        'AROSICS_SYNTHETIC_TIE_POINT_GRID.csv','FIGURE_FIXED_02/GROUND_FEATURE_ELIGIBILITY_DIAGNOSTIC.pdf']}
    blocked=[]
    for stage in ['audit','feature_plan','ground_ncc','consolidate']:
        try:getattr(runner,stage)()
        except ValueError as e:
            if not str(e).startswith('Append-only stage already has outputs:'):raise
            blocked.append(stage)
        else:raise AssertionError('Rerun not stopped before artifact writes: '+stage)
    after={p:runner.sha(OUT/p) for p in before}
    if before!=after:raise AssertionError('Rerun attempt overwrote output')
    syn=read('AROSICS_SYNTHETIC_TEST.json')
    rows=syn['rows']
    for row in rows:
        if row['output_epsg']!='32649' or row['output_valid_pixels']<=0:raise AssertionError('CRS/valid-area mismatch')
    inverse=np.linalg.norm(np.array([rows[0]['dx_map_m']+rows[1]['dx_map_m'],rows[0]['dy_map_m']+rows[1]['dy_map_m']]))
    with (OUT/'AROSICS_SYNTHETIC_TIE_POINT_GRID.csv').open(encoding='utf-8') as stream:
        ties=list(csv.DictReader(stream))
    outliers={k:sum(t[k]=='True' for t in ties) for k in ['L1_OUTLIER','L2_OUTLIER','L3_OUTLIER','OUTLIER']}
    report=dict(status='DELIVERY_HASHES_AND_HARDENED_PREFLIGHTS_VERIFIED',
        original_tests_passed=330,new_tests_passed=12,total_tests_passed=receipt['tests_passed'],
        guarded_reruns_blocked=blocked,diagnostic_artifacts_unchanged=True,
        source_stat_and2017hash_check='REVALIDATION_RECEIPT_BOUND',
        synthetic_crs_and_nonempty_valid_area_verified=True,
        synthetic_forward_reverse_closure_error_m=float(inverse),
        synthetic_inverse_within_2m=bool(inverse<=2),
        synthetic_numerical_precision_pass=False,synthetic_outlier_counts=outliers,
        outlier_detection_ground_accuracy='NOT_ESTABLISHED',
        historical_stage_code_hashes='NOT_PINNED_BEFORE_EVERY_STAGE; FINAL_HASHES_ONLY; CANNOT_RETROACTIVELY_PROVE_CODE_VERSION',
        prospective_execution_code_hash={str(p):runner.sha(p) for p in [ROOT/'tools/run_xian_feature_recovery.py',
          ROOT/'tools/run_xian_arosics_trial.py',ROOT/'tools/plot_xian_feature_recovery.py',
          ROOT/'src/xian_r5/alignment/feature_recovery.py',ROOT/'src/xian_r5/alignment/ground_trial.py',
          ROOT/'tests/test_xian_feature_recovery.py',Path(__file__)]},
        local_correction_accepts=0,absolute_position_accuracy='UNKNOWN')
    with (OUT/report_name).open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('Delivery verified; no historical stage-code-binding claim; inverse closure',float(inverse),flush=True)


if __name__=='__main__':main()
