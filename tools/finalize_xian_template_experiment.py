"""Seal an append-only fixed-template experiment, never promote it to GT."""
from pathlib import Path
import argparse, hashlib, json, sys, zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_xian_source_partition_alignment import sha, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--tests-passed', type=int, required=True)
    args = p.parse_args(); root = args.run.resolve()
    parent = Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if root.parent != parent.resolve() or (root / 'FIXED_TEMPLATE_RECEIPT.json').exists():
        raise ValueError('Fresh receipt in the dedicated run only')
    plan = json.loads((root / 'FROZEN_SOURCE_TEMPLATE_PLAN.json').read_text(encoding='utf-8'))
    replay = json.loads((root / 'SOURCE_TEMPLATE_REPLAY.json').read_text(encoding='utf-8'))
    qa = json.loads((root / 'TEMPLATE_ENGINEERING_READBACK_QA.json').read_text(encoding='utf-8'))
    for path, digest in plan['protected_sha256'].items():
        if sha(path) != digest: raise RuntimeError('Frozen artifact drift: ' + path)
    selected = [r for r in replay['records'] if r['recommended_experiment_vector_m'] is not None]
    manifest = {'branch': 'ROOF_TEMPLATE_RELATIVE_IMAGE_EXPERIMENT_ONLY',
        'experiment_year': 2019, 'source_observed_date': '2018-11-01',
        'alignment_scope': plan['alignment_scope'],
        'selected_cells': [{'region_id': r['region_id'], 'correction_m': r['recommended_experiment_vector_m']} for r in selected],
        'experimental_samples': qa['experimental_samples'],
        'label': '2019_TEMPLATE_EXPERIMENTAL_LABEL.tif', 'weight': '2019_TEMPLATE_EXPERIMENTAL_WEIGHT.tif',
        'fraction': '2019_TEMPLATE_ALIGNED_FRACTION.tif', 'quality': '2019_TEMPLATE_ALIGNMENT_QUALITY.tif',
        'inputs': '2019_S1S2_INPUT_WINDOW.tif', 'model_preprocessing': 'RAW_INPUT_NORMALIZATION_NOT_RUN',
        'weight_semantics': 'SELECTION_NOT_CONFIDENCE', 'background_semantics': 'MODEL_BELOW_THRESHOLD_ASSUMPTION_NOT_MEASURED_ABSENCE',
        'unselected_cells': 'UNKNOWN_IGNORE_NO_EXTRAPOLATION',
        'development_exposed_roi': True, 'external_independent_validation': False,
        'measured_gt_pixels': 0, 'model_training_runs': 0, 'formal_alignment_accepted': False,
        'ground_footprints': 0, 'absolute_position_accuracy': 'UNKNOWN', 'ready_for_citywide': False}
    files = [root / manifest[k] for k in ('label', 'weight', 'fraction', 'quality')]
    files += [root / 'FIXED_TEMPLATE_REPORT.md', root / 'TEMPLATE_COMPONENT_LINEAGE.json',
              root / 'FROZEN_SOURCE_TEMPLATE_PLAN.json', root / 'SOURCE_TEMPLATE_REPLAY.json',
              root / 'TEMPLATE_ENGINEERING_READBACK_QA.json']
    input_file = parent / 'SINGLE2019_20261012_06/2019_S1S2_INPUT_WINDOW.tif'
    checksums = {f.name: sha(f) for f in files + [input_file]}
    package = root / 'SINGLE2019_ROOF_TEMPLATE_EXPERIMENT.zip'
    with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files + [input_file]: z.write(f, f.name)
        z.writestr('DATASET_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr('FILES_SHA256.json', json.dumps(checksums, indent=2))
    with zipfile.ZipFile(package) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP CRC mismatch')
        for name, digest in checksums.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest: raise RuntimeError('Package hash mismatch: ' + name)
    code_paths = [ROOT / 'src/xian_r5/alignment/source_template_alignment.py',
                  ROOT / 'tests/test_xian_source_template_alignment.py',
                  ROOT / 'tools/run_xian_source_template_alignment.py', Path(__file__)]
    receipt = {'run': root.name, 'tests_passed': args.tests_passed,
        'protected_inputs_checked': len(plan['protected_sha256']), 'protected_inputs_changed': 0,
        'new_code_sha256': {str(f.resolve()): sha(f) for f in code_paths},
        'output_sha256': {f.name: sha(f) for f in root.iterdir() if f.is_file()},
        'package_crc_and_hash_verified': True, 'manifest': manifest,
        'whole_system_governance_completed': False,
        'formal_state': {'FORMAL_FIT_HOLDOUT_REVIEWS_COMPLETED': 0, 'ACCEPTED_ALIGNMENT_CELLS': 0,
            'REJECTED_ALIGNMENT_CELLS': 0, 'UNRESOLVED_ALIGNMENT_CELLS': 9, 'REGISTERED_ROOFS': 0,
            'PHYSICAL_IDS': 0, 'CANONICAL_GEOMETRIES': 0, 'GROUND_FOOTPRINTS': 0,
            'ABSOLUTE_POSITION_ACCURACY': 'UNKNOWN', 'READY_FOR_CITYWIDE': 'NO'}}
    save_json(root / 'FIXED_TEMPLATE_RECEIPT.json', receipt)
    print(json.dumps({'package': str(package), 'experimental_samples': qa['experimental_samples'],
        'protected_changed': 0, 'receipt_sha256': sha(root / 'FIXED_TEMPLATE_RECEIPT.json')}))


if __name__ == '__main__': main()
