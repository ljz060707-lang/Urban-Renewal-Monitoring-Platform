"""Append-only user reference binding and exact 10m annotation fractions."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import fiona, rasterio
from affine import Affine
from pyproj import Transformer
from shapely.geometry import shape, Polygon, mapping
from shapely.ops import transform
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from run_xian_source_partition_alignment import sha, save_json, write_new
from xian_r5.label_transfer import Grid, export_raster
from xian_r5.manual_roof_reference import manual_reference_binding, manual_roof_fractions


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output.resolve()
    parent = Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent != parent.resolve() or out.exists():
        raise ValueError('Fresh append-only reference run required')
    batch = parent/'SINGLE2019_FULLROI_ROOF_PROPOSALS_20261012_20'
    prior = json.loads((batch/'FROZEN_FULLROI_SAM_PLAN.json').read_text(encoding='utf-8'))
    receipt = json.loads((batch/'FULLROI_INFERENCE_RECEIPT.json').read_text(encoding='utf-8'))
    if (receipt['summary']['source_groups_completed'] != 1396 or
            receipt['summary']['source_units_completed'] != 16):
        raise ValueError('Full source proposal batch not complete')
    protected = dict(prior['protected_sha256']); protected.update(prior['new_code_sha256'])
    for n in ('FROZEN_FULLROI_SAM_PLAN.json', 'FULLROI_INFERENCE_RECEIPT.json', 'FULLROI_INFERENCE_SUMMARY.json'):
        protected[str(batch/n)] = sha(batch/n)
    protected.update({str(batch/n): h for n, h in receipt['group_receipts_sha256'].items()})
    gt = Path('F:/xian/西安建筑轮廓数据/西安.shp')
    for f in gt.parent.glob('西安.*'):
        if f.is_file(): protected[str(f)] = sha(f)
    census_path = parent/'GT10M_VALIDATION_PREFLIGHT_20261012_05/REFERENCE_EVIDENCE_CENSUS.json'
    census = json.loads(census_path.read_text(encoding='utf-8'))
    expected = {r['source_fid']: r['geometry_sha256_metric'] for r in census['records']}
    if len(expected) != 4821: raise ValueError('Frozen reference ROI census changed')
    raw_path = parent/'SINGLE2019_20261012_06/2019_ROOF_FRACTION_RAW.tif'
    active = parent/'SINGLE2019_COHERENT_TRANSFER_20261012_12'
    inputs = [raw_path, active/'2019_ROOF_FRACTION.tif', active/'2019_EXPERIMENT_WEIGHT.tif', census_path]
    for f in inputs: protected[str(f)] = sha(f)
    for path, digest in protected.items():
        if sha(path) != digest: raise ValueError('Protected input drift: '+path)
    grid = Grid(Affine(10, 0, 308460, 0, -10, 3794750), 256, 256, 'EPSG:32649')
    binding = manual_reference_binding()
    code = [Path(__file__).resolve(), ROOT/'src/xian_r5/manual_roof_reference.py',
            ROOT/'tests/test_xian_manual_roof_reference.py']
    out.mkdir()
    save_json(out/'FROZEN_MANUAL_REFERENCE_BINDING.json', {
        'run': out.name, 'reference_path': str(gt), 'binding': binding,
        'user_confirmation_is_not_a_fabricated_reviewer_signature': True,
        'prior_reports_or_contracts_rewritten': False, 'source_observed_date': '2018-11-01',
        'experiment_year': 2019, 'source_GT_geometry_not_edited': True,
        'tau': .43, 'threshold_comparison': 'STRICT_GREATER_THAN_DIAGNOSTIC_ONLY',
        'protected_sha256': protected, 'new_code_sha256': {str(f): sha(f) for f in code}})
    ll = Transformer.from_crs(32649, 4326, always_xy=True)
    metric = Transformer.from_crs(4326, 32649, always_xy=True)
    corners = [(308460,3792190), (311020,3792190), (311020,3794750), (308460,3794750)]
    ring = [ll.transform(a[0]+(b[0]-a[0])*n/64, a[1]+(b[1]-a[1])*n/64)
            for a,b in zip(corners,corners[1:]+corners[:1]) for n in range(64)]
    refs = {}; records = []
    with fiona.open(gt) as s:
        if s.crs.to_epsg() != 4326: raise ValueError('Reference CRS changed')
        for f in s.filter(bbox=Polygon(ring).bounds):
            fid = str(f.id)
            if fid not in expected: continue
            g = transform(metric.transform, shape(f.geometry))
            digest = hashlib.sha256(g.wkb).hexdigest()
            if digest != expected[fid]: raise ValueError('Reference geometry changed: '+fid)
            refs[fid] = g
            records.append({'source_fid': fid, 'geometry_sha256_metric': digest,
                            'reference_role': binding['reference_role']})
    if set(refs) != set(expected): raise ValueError('Missing frozen reference objects')
    print('Confirmed immutable manual roofs loaded; measuring exact 10m union fractions.', flush=True)
    data = manual_roof_fractions(refs, grid)
    fraction = data['fraction']
    def write(name, array, bands):
        array = np.asarray(array)
        if array.ndim == 2: array = array[None]
        export_raster(out/name, array, grid, bands, root=out)
        with rasterio.open(out/name, 'r+') as s:
            s.update_tags(ROLE='USER_CONFIRMED_MANUAL_ROOF_REFERENCE',
                GT_SOURCE='HUMAN_ANNOTATION_USER_CONFIRMED', REFERENCE_NOMINAL_YEAR='2019',
                UNBIASED_BASEMAP_ASSERTION='USER_CONFIRMED_NOT_NUMERIC_ERROR_BOUND',
                ABSOLUTE_POSITION_ACCURACY='UNKNOWN', GEOMETRY='ROOF_NOT_GROUND_FOOTPRINT',
                TRAINING_ADMISSION='NOT_AUTOMATIC', UNANNOTATED_AREA='UNKNOWN_NOT_PROVEN_BACKGROUND',
                DENOMINATOR='FULL_10M_PIXEL_100_M2', SCOPE='CURRENT_FIXED_256x256_REFERENCE_COMPARISON')
        with rasterio.open(out/name) as s:
            expected_array = array.astype(np.float32) if array.dtype.kind == 'f' else array
            if s.transform != grid.affine or str(s.crs) != grid.crs or not np.array_equal(s.read(), expected_array):
                raise ValueError('Reference raster readback mismatch')
    write('2019_MANUAL_REFERENCE_ROOF_FRACTION_10m.tif', fraction, ['MANUAL_ROOF_UNION_FRACTION'])
    write('2019_MANUAL_REFERENCE_OBJECT_COUNT.tif', data['object_count'], ['ANNOTATION_OBJECT_COUNT'])
    quality = np.stack([(fraction > 0), data['overlap_conflict'], data['roi_truncated_object'],
                        data['known_background'], data['training_admission']]).astype(np.uint8)
    write('2019_MANUAL_REFERENCE_QUALITY.tif', quality,
          ['ANNOTATED_ROOF_PRESENT','MANUAL_POLYGON_OVERLAP','ROI_TRUNCATED_ROOF',
           'KNOWN_BACKGROUND','AUTOMATIC_TRAINING_ADMISSION'])
    arrays = []
    for f in inputs[:3]:
        with rasterio.open(f) as s:
            if Grid(s.transform,s.width,s.height,str(s.crs)) != grid:
                raise ValueError('Reference/baseline comparison grid mismatch')
            arrays.append(s.read(1))
    raw, candidate, weight = arrays; selected = weight > 0
    if int(selected.sum()) != 12903: raise ValueError('Recommended baseline selection drift')
    for name, values in [('RAW', raw), ('ACTIVE_CANDIDATE', candidate)]:
        mask = np.isfinite(values) & (values >= 0) & (values <= 1)
        if name == 'ACTIVE_CANDIDATE': mask &= selected
        diff = np.full_like(fraction, -9999.); diff[mask] = values[mask]-fraction[mask]
        # Difference NoData remains explicit; never turn unsupported cells into zeros.
        export_raster(out/f'2019_{name}_MINUS_MANUAL_REFERENCE.tif', diff[None], grid,
                      [name+'_MINUS_MANUAL_ROOF_FRACTION'], root=out, nodata=-9999.)
        with rasterio.open(out/f'2019_{name}_MINUS_MANUAL_REFERENCE.tif', 'r+') as s:
            s.update_tags(ROLE='DEVELOPMENT_REFERENCE_COMPARISON_NOT_EXTERNAL_ACCURACY',
                          NO_NEW_DISPLACEMENT_OR_LABELS='YES', THRESHOLD_SEARCH='NO')
    comparisons = {}
    for name, values, mask in [('raw_full_window',raw,(raw>=0)&(raw<=1)),
                               ('raw_same_selected_pixels',raw,selected),
                               ('active_same_selected_pixels',candidate,selected)]:
        delta = values[mask].astype(np.float64)-fraction[mask]
        comparisons[name] = {'pixels': int(mask.sum()),
            'mean_absolute_fraction_difference': float(np.abs(delta).mean()),
            'mean_signed_fraction_difference': float(delta.mean()),
            'tau_043_disagreements': int(((values[mask]>.43)!=(fraction[mask]>.43)).sum()),
            'is_independent_validation_or_acceptance_statistic': False}
    summary = {'manual_reference_role_confirmed': True,
        'manual_reference_roofs_in_current_roi': len(refs),
        'reference_union_area_m2': data['union_area_m2'],
        'area_conservation_error_m2': data['area_conservation_error_m2'],
        'manual_roof_positive_pixels': int((fraction>0).sum()),
        'manual_polygon_overlap_pixels': int(data['overlap_conflict'].sum()),
        'roi_truncated_roof_pixels': int(data['roi_truncated_object'].sum()),
        'comparisons': comparisons, 'new_displacements_or_applications': 0,
        'new_training_labels': 0, 'recommended_experimental_pixels_unchanged': 12903,
        'external_unseen_validation': False, 'absolute_position_accuracy': 'UNKNOWN',
        'whole_window_governance_completed': False}
    save_json(out/'MANUAL_REFERENCE_OBJECT_PROVENANCE.json', {'records': records, 'source_sha256':sha(gt)})
    save_json(out/'MANUAL_REFERENCE_FRACTION_SUMMARY.json', summary)
    write_new(out/'MANUAL_REFERENCE_BINDING_REPORT.md',
        '# 2019人工屋顶GT参考接入\n\n'
        '按用户明确确认，西安建筑轮廓数据为无偏底图上的2019人工屋顶标注；'
        '正式用于屋顶位置及10m面积分数比较，不再仅限身份辅助。用户确认不是伪造的人工FIT/HOLDOUT签署。\n\n'
        f'当前窗口读取{len(refs)}个既有参考对象，保留原几何及FID，生成固定256×256的并集面积分数。'
        '分母为完整100m²像元；重叠按并集去重，边界截断与未标注区域单独保留。'
        '不把未标注区域当作已验证背景，不删除小于100m²的人工参考对象。\n\n'
        '与原始及当前候选fraction比较的误差只是开发集差异，不是外部精度认证。'
        '参考已参与开发，不能重新包装成未见过的holdout；不自动填写米级精度，不将屋顶称为地面足迹。'
        '已知源影像2018-11-01仍保留，实验年为用户授权2019。\n\n'
        '本轮未拟合或应用新位移，未修改GT/原始TIF/旧合同/已交付标签。'
        '原B04正式review/accept/reject仍0、unresolved=9，其余正式身份/足迹/城市级流程未运行。'
        '推荐实验像元仍12,903，整个窗口治理尚未完成。\n\n'
        '后续使用本参考检验分来源校正及fraction误差；源掩膜选择须先按原图语义冻结，'
        '不能选取最贴GT的SAM候选或将源屋顶直接吸附到GT。\n')
    for path, digest in protected.items():
        if sha(path) != digest: raise ValueError('Protected input changed: '+path)
    save_json(out/'MANUAL_REFERENCE_BINDING_RECEIPT.json', {
        'summary': summary, 'protected_artifacts_checked':len(protected),
        'protected_artifacts_changed':0, 'new_code_sha256':{str(f):sha(f) for f in code},
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
