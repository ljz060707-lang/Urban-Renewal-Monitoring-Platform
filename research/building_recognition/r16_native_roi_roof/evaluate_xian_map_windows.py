"""Measure overlap with designated 2019 map polygons on frozen Xi'an TUNE windows."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

R8 = Path('/root/xian20261111_r8')
R16 = Path('/root/xian20261111_r16_roi_roof')
OUT = R16 / 'output' / 'xian_map_reference'
def read_verified(path, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Input hash mismatch {path}')
def center(transform, projection, x, y):
    lon, lat = transform * (x + 0.5, y + 0.5)
    return np.asarray(projection.transform(lon, lat))
def evaluate(args):
    row, candidate_manifest = args
    name = row['id']
    assert row['role'] == 'TUNE' and name.startswith('TUNE_')
    native = Path(row['image_path']).parent
    source = row['source_record']
    gt = native / 'PROVIDED_MAP_GT.gpkg'
    read_verified(gt, source['files'][gt.name]['sha256'])
    read_verified(row['query_cache_path'], row['query_cache_sha256'])
    with np.load(row['query_cache_path']) as data:
        raw = data['instances'].copy()
        support = data['support'].astype(bool)
    pred_dir = R8 / 'output' / 'xian72_governance' / name
    receipt = json.loads((pred_dir / 'WINDOW_RECEIPT.json').read_text())
    governed = pred_dir / 'ROOF_GOVERNED_OBSERVATION_ID.tif'
    digest = next(r['sha256'] for r in receipt['artifacts'] if Path(r['path']).name == governed.name)
    read_verified(governed, digest)
    transform = Affine(*row['native_affine'])
    with rasterio.open(governed) as ds:
        assert ds.transform == transform and str(ds.crs) == row['native_crs']
        labels = {'RAW': raw, 'AFTER_FACADE': ds.read(1)}
    if candidate_manifest:
        selected = candidate_manifest[name]
        read_verified(selected['path'], selected['sha256'])
        with np.load(selected['path']) as data:
            labels['ROI_SELECTED'] = data[selected['variant']].copy()
    for lab in labels.values():
        assert lab.shape == support.shape and lab.dtype.kind in 'iu' and (lab >= 0).all()
        lab[~support] = 0
    proj = Transformer.from_crs(row['native_crs'], 32649, always_xy=True)
    polygons = pyogrio.read_dataframe(gt)
    designated = polygons.loc[polygons['fusion_role'] == 'DESIGNATED_GT2019_ANCHOR'].copy()
    assert designated.original_gt_fid.is_unique
    designated = designated.to_crs(row['native_crs'])
    height, width = support.shape
    references = []
    invisible = []
    for fid, geometry in zip(designated.original_gt_fid, designated.geometry):
        flag = rasterize([(geometry, 1)], out_shape=support.shape, transform=transform, dtype='uint8').astype(bool)
        supported = flag & support
        if not supported.any():
            invisible.append(int(fid))
            continue
        yy, xx = np.where(supported)
        touches = min(xx.min(), yy.min(), width-1-xx.max(), height-1-yy.max()) == 0
        references.append((int(fid), yy, xx, bool(touches), center(transform, proj, xx.mean(), yy.mean())))
    results = {}
    details = {k: [] for k in labels}
    for method, lab in labels.items():
        sizes = np.bincount(lab.ravel().astype('int64'))
        ids = np.flatnonzero(sizes[1:]) + 1
        by_id = {qid:i for i,qid in enumerate(ids)}
        centroids = ndimage.center_of_mass(lab > 0, labels=lab, index=ids) if len(ids) else []
        overlap = np.zeros((len(references), len(ids)))
        covered = []
        for i, (_, yy, xx, _, _) in enumerate(references):
            values = lab[yy, xx]
            hits, counts = np.unique(values[values > 0], return_counts=True)
            covered.append(int(counts.sum()))
            for qid, count in zip(hits, counts):
                overlap[i, by_id[qid]] = count / (len(yy) + sizes[qid] - count)
        valid = overlap >= .5
        n = min(overlap.shape)
        rows, cols = linear_sum_assignment(-valid.astype(float) - overlap / (n + 1)) if n else ([], [])
        matched = {int(i):int(j) for i,j in zip(rows,cols) if valid[i,j]}
        for i, (fid, yy, xx, touches, gt_center) in enumerate(references):
            j = matched.get(i)
            distance = None
            if j is not None:
                y, x = centroids[j]
                distance = float(np.linalg.norm(gt_center - center(transform, proj, x, y)))
            details[method].append(dict(fid=fid, supported_reference_pixels=len(yy), covered_pixels=covered[i],
                window_edge=touches, best_IoU=float(overlap[i].max()) if len(ids) else 0.,
                IoU50_match=j is not None, matched_IoU=float(overlap[i,j]) if j is not None else None,
                matched_query_id=int(ids[j]) if j is not None else None, centroid_m=distance))
        results[method] = dict(predictions=len(ids), predicted_union_pixels=int(np.count_nonzero(lab)))
    return dict(window=name, epoch=row['epoch'], date=source['actual_date'], date_votes=source['actual_date_pixel_votes'],
        stratum=source['stratum'], site=row['site'], without_supported_reference_pixels=invisible, stats=results, details=details)
def summarize(rows, method):
    rr=[r for window in rows for r in window['details'][method]]
    if not rr:
        return dict(reference_observations=0)
    gt_px=sum(r['supported_reference_pixels'] for r in rr)
    return dict(reference_observations=len(rr), window_edge_observations=sum(r['window_edge'] for r in rr),
        zero_overlap=sum(r['covered_pixels']==0 for r in rr), matches_at_IoU50=sum(r['IoU50_match'] for r in rr),
        object_recall_IoU50=np.mean([r['IoU50_match'] for r in rr]), mean_best_IoU=np.mean([r['best_IoU'] for r in rr]),
        reference_pixel_coverage=sum(r['covered_pixels'] for r in rr)/gt_px)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-manifest',type=Path)
    args=parser.parse_args()
    candidate=None
    if args.candidate_manifest:
        candidate={r['frame_id']:r for r in json.loads(args.candidate_manifest.read_text())['rows']}
    windows=[r for line in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))['role']=='TUNE']
    assert len(windows)==72 and len(set(r['site'] for r in windows))==24
    if candidate is not None:
        assert set(candidate)==set(w['id'] for w in windows)
    with ProcessPoolExecutor(6) as pool:
        rows=list(pool.map(evaluate,[(w,candidate) for w in windows]))
    result={}
    for epoch in sorted({r['epoch'] for r in rows}):
        ww=[r for r in rows if r['epoch']==epoch]
        summary={k:summarize(ww,k) for k in ww[0]['details']}
        for method in list(summary)[1:]:
            pairs=[(a,b) for w in ww for a,b in zip(w['details']['RAW'],w['details'][method])]
            assert all(a['fid']==b['fid'] for a,b in pairs)
            shared=[(a,b) for a,b in pairs if a['matched_query_id'] is not None and a['matched_query_id']==b['matched_query_id']]
            summary[method]['paired_same_ID_at_IoU50']=len(shared)
            summary[method]['mean_centroid_m_raw_paired']=np.mean([a['centroid_m'] for a,b in shared]) if shared else None
            summary[method]['mean_centroid_m_after_paired']=np.mean([b['centroid_m'] for a,b in shared]) if shared else None
            summary[method]['lost_IoU50']=sum(a['IoU50_match'] and not b['IoU50_match'] for a,b in pairs)
            summary[method]['gained_IoU50']=sum(b['IoU50_match'] and not a['IoU50_match'] for a,b in pairs)
        result[epoch]=dict(dates=sorted({d for w in ww for d in w['date_votes']}),metrics=summary)
    OUT.mkdir(exist_ok=True,parents=True)
    label='SELECTED' if candidate is not None else 'BASELINE'
    for suffix, data in [('OBJECTS',rows),('SUMMARY',dict(by_epoch=result,windows=72,sites=24,role='TUNE',scope='2019 designated map reference on supported native window grid; repeated sites; different dates are not presence labels; no complete background or independent native roof/facade truth',outside_support_or_subpixel_by_window={w['window']:len(w['without_supported_reference_pixels']) for w in rows},one_to_one_assignment='max cardinality at IoU>=0.50 then full assignment IoU',protected_data_read=0))]:
        dest=OUT/f'{label}_{suffix}.json'
        tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,allow_nan=False))
        tmp.replace(dest)
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    main()

