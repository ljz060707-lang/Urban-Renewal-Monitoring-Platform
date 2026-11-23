"""Fresh readback, complete zip manifest and pixel-rule verification."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
import numpy as np,rasterio
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment').resolve()
    if run.parent!=parent or (run/'INDEPENDENT_DELIVERY_READBACK.json').exists():raise ValueError('Fresh readback annex required')
    receipt=json.loads((run/'CURRENT_WINDOW_DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))
    for name,digest in receipt['output_sha256'].items():
        if sha(run/name)!=digest:raise ValueError('Delivered artifact drift: '+name)
    manifest=json.loads((run/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))['files_sha256']
    archive=run/'CURRENT_WINDOW_REFERENCE_SCREENED_EXPERIMENT.zip'
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(manifest)|{'PACKAGE_MANIFEST.json'}:raise ValueError('Archive scope mismatch')
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Archive member drift: '+name)
        if z.testzip() is not None:raise ValueError('Archive CRC failure')
    cfg=json.loads((run/'RECOMMENDED_EXPERIMENT_DATASET.json').read_text(encoding='utf-8'))
    reference=None;rasters={};grid=None
    for path in run.glob('*.tif'):
        with rasterio.open(path) as s:
            current=(tuple(s.transform),str(s.crs),s.shape)
            if grid is not None and current!=grid:raise ValueError('Raster grids differ')
            grid=current;rasters[path.name]=s.read()
            if s.shape!=(256,256) or str(s.crs)!='EPSG:32649':raise ValueError('Raster ROI drift')
    label=rasters[cfg['label']][0];weight=rasters[cfg['weight']][0]
    positive=rasters[cfg['positive_only_label']][0];pw=rasters[cfg['positive_only_weight']][0]
    fraction=rasters[cfg['fraction']][0].astype(np.float64)
    ref=rasters['2019_MANUAL_REFERENCE_ROOF_FRACTION.tif'][0].astype(np.float64)
    q=rasters[cfg['reference_quality']]
    selected=weight>0
    if not np.array_equal(selected,label!=255) or not np.isin(label,[0,1,255]).all():raise ValueError('Label ignore/weight mismatch')
    if not np.isin(weight,[0,1]).all() or not np.array_equal(label[selected],(fraction[selected]>.43).astype(np.uint8)):
        raise ValueError('Source strict threshold contract differs')
    if not np.array_equal(label[selected],(ref[selected]>.43).astype(np.uint8)):raise ValueError('Selected manual reference mismatch')
    if q[2][selected].any() or q[3][selected].any() or q[7:].any():raise ValueError('Risk or formal accuracy promotion')
    if not np.array_equal(pw>0,positive==1) or not np.isin(positive,[1,255]).all():raise ValueError('Positive-only companion drift')
    if not np.array_equal(positive==1,(label==1)&selected):raise ValueError('Positive source/reference subset drift')
    scope=np.zeros((256,256),bool)
    for rr,cc in ((1,2),(2,1),(3,0),(3,1)):scope[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
    if selected[~scope].any():raise ValueError('Unresolved unit gained selection')
    n=int(selected.sum());p=int((positive==1).sum());ignore=int((label==255).sum())
    if n!=10096 or p!=5014 or ignore!=55440 or n+ignore!=65536:raise ValueError('Delivered full-window counts differ')
    affine=cfg['grid']['affine'];gdal=cfg['grid']['gdal_geotransform']
    if gdal!=[affine[2],affine[0],affine[1],affine[5],affine[3],affine[4]]:raise ValueError('Affine/GDAL ordering differs')
    save_json(run/'INDEPENDENT_DELIVERY_READBACK.json',{
        'verifier_sha256':sha(Path(__file__).resolve()),'main_receipt_sha256':sha(run/'CURRENT_WINDOW_DELIVERY_RECEIPT.json'),
        'archive_sha256':sha(archive),'archive_members_verified':len(manifest)+1,
        'raster_grids_verified':len(rasters),'all_pixel_counts_accounted':True,
        'selected_experiment_pixels':n,'reference_corroborated_positives':p,'ignore_pixels':ignore,
        'selected_label_fraction_and_reference_agreement':True,
        'formal_GT_or_accuracy_promoted':False,'new_displacements_applied':False,
        'source_actual_date_preserved':cfg['source_actual_date']=='2018-11-01',
        'unresolved_scope_leak_pixels':0,'reader_ignore_index':cfg['loss_ignore_index'],
        'scope':'ENGINEERING_READBACK_NOT_INDEPENDENT_SPATIAL_ACCURACY'})
    print(json.dumps({'verified':True,'raster_grids':len(rasters),'selected':n,'positive_only':p,'ignore':ignore}))


if __name__=='__main__':main()
