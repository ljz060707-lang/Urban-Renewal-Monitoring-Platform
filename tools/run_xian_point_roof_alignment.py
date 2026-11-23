"""Purposeful raw-roof RGB replay with a frozen cross-descriptor conflict gate."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import rasterio, fiona
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import shape
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from run_xian_source_partition_alignment import sha, save_json
from run_xian_single2019_experiment import write
from xian_r5.alignment.point_roof_alignment import point_roof_check, reconcile_roof_descriptors
from xian_r5.fraction_shadow import build_fraction_shadow, QUALITY_BANDS, NODATA
from xian_r5.label_transfer import Grid
from xian_r5.source_partition import SourceRegion
from xian_r5.single_year_experiment import experimental_pseudolabel


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output.resolve()
    parent = Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    prior = parent / 'SINGLE2019_FIXED_TEMPLATE_20261012_10'; base = parent / 'SINGLE2019_20261012_06'
    if out.parent != parent.resolve() or out.exists(): raise ValueError('Fresh dedicated run required')
    prior_plan = json.loads((prior / 'FROZEN_SOURCE_TEMPLATE_PLAN.json').read_text(encoding='utf-8'))
    receipt = json.loads((prior / 'FIXED_TEMPLATE_RECEIPT.json').read_text(encoding='utf-8'))
    protected = dict(prior_plan['protected_sha256']); protected.update(receipt['new_code_sha256'])
    protected.update({str(prior / name): digest for name, digest in receipt['output_sha256'].items()})
    protected[str(prior / 'FIXED_TEMPLATE_RECEIPT.json')] = sha(prior / 'FIXED_TEMPLATE_RECEIPT.json')
    for path, digest in protected.items():
        if sha(path) != digest: raise ValueError('Frozen input drift: ' + path)
    plan = {'run': out.name, 'scope': 'SAME_256x256_NOMINAL_2019',
        'alignment_scope': 'SOURCE_ROOF_POINT_RGB_TO_REFERENCE_IMAGE_CONTEXT_NOT_GROUND',
        'feature_operator': 'POINT_RGB_SEPARATE_BAND_CENTERING_NO_BLUR_NO_TARGET_ANNOTATIONS',
        'experiment_year': 2019, 'source_observed_date': '2018-11-01',
        'source_roof_mask_threshold': .8, 'search_radius_px': 3, 'fit_and_check_minimum_distinct_pixels': 100,
        'fold_block_size_px': 16, 'informative_block_minimum_distinct_pixels': 20,
        'minimum_positive_check_blocks': 3, 'materially_negative_blocks_allowed': 0,
        'minimum_fit_peak_gap': .02, 'minimum_global_check_gain': .01,
        'minimum_improving_check_rgb_bands': 2, 'materially_negative_rgb_bands_allowed': 0,
        'same_source_population_for_every_offset': True, 'no_feature_radius_threshold_sweep': True,
        'conflict_rule': 'IF_DERIVATIVE_FIT_AND_CHECK_COMPUTABLE_REQUIRE_SAME_DIRECTION',
        'descriptor_choice_by_coverage_or_best_check_score': False,
        'development_exposed_roi': True, 'external_independent_validation': False,
        'formal_alignment_acceptance': False, 'protected_sha256': protected,
        'new_code_sha256_at_freeze': {str(f): sha(f) for f in (
            ROOT / 'src/xian_r5/alignment/point_roof_alignment.py',
            ROOT / 'tests/test_xian_point_roof_alignment.py', Path(__file__).resolve())}}
    out.mkdir(); save_json(out / 'FROZEN_POINT_RGB_PLAN.json', plan)
    # Read the data only after freezing the algorithm and admission policy.
    derivative = {r['region_id']: r for r in json.loads((prior / 'SOURCE_TEMPLATE_REPLAY.json').read_text(encoding='utf-8'))['records']}
    grid = Grid(Affine(10, 0, 308460, 0, -10, 3794750), 256, 256, 'EPSG:32649')
    with rasterio.open(base / '2019_ROOF_FRACTION_RAW.tif') as s: raw = s.read(1)
    with rasterio.open(parent / 'SINGLE2019_SUPPORT_REPAIR_20261012_07/2019_BOUNDED_GE_RGB_CACHE.tif') as s:
        source = np.moveaxis(s.read(), 0, -1)
    with rasterio.open('F:/xian/10m底图/Xian_2019_TrueColor.tif') as s:
        window = Window(1516, 1765, 256, 256); target = np.moveaxis(s.read([1, 2, 3], window=window), 0, -1)
        valid = np.all(s.read_masks([1, 2, 3], window=window) > 0, axis=0)
    mask = (raw >= .8) & np.any(source > 0, axis=2)
    cells = json.loads((parent / 'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    records = []; regions = []
    for cell in cells:
        rr, cc = (int(v[1:]) for v in cell['region_id'].split('_')[1:])
        support = np.zeros((256, 256), bool); support[rr*64:(rr+1)*64, cc*64:(cc+1)*64] = True
        result = point_roof_check(source, target, mask, valid, support)
        agreement = reconcile_roof_descriptors(result, derivative[cell['region_id']]['diagnosis'])
        best = result['best']; vector = None
        if best is not None:
            dx, dy = best['correction_dx_px'], best['correction_dy_px']
            vector = [grid.affine.a*dx + grid.affine.b*dy, grid.affine.d*dx + grid.affine.e*dy]
        recommended = vector if agreement['experimental_recommendation'] else None
        rec = {'region_id': cell['region_id'], 'diagnostic_vector_m': vector,
            'recommended_experiment_vector_m': recommended, 'diagnosis': result, 'descriptor_reconciliation': agreement,
            'derivative_diagnostic_vector_m': derivative[cell['region_id']]['diagnostic_vector_m'],
            'source_observed_date': '2018-11-01', 'experiment_year': 2019, 'formal_acceptance': False}
        records.append(rec); digest = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        regions.append(SourceRegion(cell['region_id'], '2018-11-01', shape(cell['support_geometry']),
            None if recommended is None else tuple(recommended), None if recommended is None else digest))
    summary = {'run': out.name, 'alignment_scope': plan['alignment_scope'], 'records': records,
        'numeric_fit_cells': sum(r['diagnostic_vector_m'] is not None for r in records),
        'point_only_internal_pass_cells': sum(r['diagnosis']['engineering_experiment_supported'] for r in records),
        'cross_descriptor_supported_cells': sum(r.correction_m is not None for r in regions),
        'descriptor_conflict_cells': sum('ROOF_DESCRIPTOR_DIRECTION_CONFLICT' in r['descriptor_reconciliation']['reasons'] for r in records),
        'prior_recommended_cells_now_conflicted': [r['region_id'] for r in records if
            derivative[r['region_id']]['recommended_experiment_vector_m'] is not None and
            'ROOF_DESCRIPTOR_DIRECTION_CONFLICT' in r['descriptor_reconciliation']['reasons']],
        'experimental_samples': 0, 'measured_gt_pixels': 0, 'formal_ground_accepts': 0}
    if summary['cross_descriptor_supported_cells']:
        with fiona.open(parent / 'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
            objects = {f.properties['component_key']: shape(f.geometry) for f in s}
        result = build_fraction_shadow(objects, regions, grid)
        pseudo = experimental_pseudolabel(result['candidate_fraction'].astype(np.float32).astype(np.float64),
            result['scope_fraction'][1].astype(np.float32).astype(np.float64), result['quality'], np.ones((256,256), bool),
            allow_unverified_geometry=True, allow_assumed_background=True)
        write(out, '2019_POINT_RGB_FRACTION.tif', result['candidate_fraction'], grid, ['POINT_RGB_FRACTION'], nodata=NODATA)
        write(out, '2019_POINT_RGB_EXPERIMENT_LABEL.tif', pseudo['label'], grid, ['EXPERIMENT_LABEL'], nodata=255)
        write(out, '2019_POINT_RGB_EXPERIMENT_WEIGHT.tif', pseudo['experiment_weight'], grid, ['EXPERIMENT_SELECTION'])
        write(out, '2019_POINT_RGB_QUALITY.tif', result['quality'], grid, QUALITY_BANDS)
        save_json(out / 'POINT_RGB_COMPONENT_LINEAGE.json', {'records': result['component_ledger'], 'role': plan['alignment_scope']})
        summary['experimental_samples'] = pseudo['counts']['experimental_samples']
        summary['label_counts'] = {str(v): int((pseudo['label'] == v).sum()) for v in (0, 1, 255)}
        summary['area_conservation_error_m2'] = result['area_conservation_error_m2']
    save_json(out / 'POINT_RGB_REPLAY.json', summary)
    for path, digest in protected.items():
        if sha(path) != digest: raise RuntimeError('Frozen input changed: ' + path)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}))


if __name__ == '__main__': main()
