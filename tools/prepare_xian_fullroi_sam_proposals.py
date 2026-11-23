"""Full frozen-ROI source-prior SAM proposal plan; all eligible parents included."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np,cv2,rasterio,fiona,shapely
from rasterio.windows import Window
from shapely.geometry import shape,mapping
from shapely.ops import transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.source_roof_batch_scope import source_roof_batch_scope


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment');prior=parent/'SINGLE2019_CURRENT_ROOF_BANK_20261012_19'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh full-ROI proposal run required')
    old_plan=json.loads((prior/'FROZEN_CURRENT_ROOF_BANK_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'CURRENT_SOURCE_ROOF_BANK_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(old_plan['protected_sha256']);protected.update(old_plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'CURRENT_SOURCE_ROOF_BANK_RECEIPT.json')]=sha(prior/'CURRENT_SOURCE_ROOF_BANK_RECEIPT.json')
    annex=json.loads((prior/'ROOF_BANK_PROGRESS_ANNEX_RECEIPT.json').read_text(encoding='utf-8'))
    protected.update({str(prior/n):h for n,h in annex['new_annex_sha256'].items()})
    protected[str(prior/'ROOF_BANK_PROGRESS_ANNEX_RECEIPT.json')]=sha(prior/'ROOF_BANK_PROGRESS_ANNEX_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source drift: '+path)
    with fiona.open(parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
        if s.crs.to_epsg()!=32649:raise ValueError('Source frame differs')
        objects={str(f.properties['component_key']):shape(f.geometry) for f in s}
    field=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    regions={c['region_id']:shape(c['support_geometry']) for c in field}
    work=source_roof_batch_scope(objects,regions)
    if len(work['targets'])!=1396:raise ValueError('Frozen structural source census changed')
    runtime=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/22_roof_candidate_segmentation/SAM_VITB_RUNTIME_20261012_02')
    code=[ROOT/'src/xian_r5/source_roof_batch_scope.py',ROOT/'tests/test_xian_source_roof_batch_scope.py',
        Path(__file__).resolve(),ROOT/'tools/infer_xian_fullroi_sam_cpu.py']
    plan={'run':out.name,'scope':'ENTIRE_FIXED_256x256_GRID_16_FROZEN_UNITS_NOT_CITYWIDE',
        'source_units_requested':16,'source_units_matched':len(work['per_unit_targets']),
        'source_objects_requested':1396,'selection':work['selection'],
        'weak_prior_prompts_are_not_roof_GT_or_registration_ties':True,
        'source_observed_date':'2018-11-01','experiment_year':2019,'actual_date_not_overwritten':True,
        'backend':'OFFICIAL_SAM_VIT_B_NOT_RBF_SAM','runtime':str(runtime),
        'checkpoint':str(runtime/'sam_vit_b_01ec64.pth'),
        'checkpoint_sha256':'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912',
        'device':'cpu','torch_threads':2,'multimask_output':True,'automatic_mask_selection':False,
        'GT_or_target_pixels_or_predictions_used':False,'old_masks_or_vectors_modified':False,
        'context_margin_m':20,'native_input_resampling':'NONE; model internal resize is upstream default',
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}}
    out.mkdir();save_json(out/'FROZEN_FULLROI_SAM_PLAN.json',plan);save_json(out/'SOURCE_ONLY_WORKLIST.json',work)
    targets=[];packets=[];source=Path('F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif')
    roi=shapely.union_all(list(regions.values()));to_ll=Transformer.from_crs(32649,4326,always_xy=True)
    with rasterio.open(source) as s:
        if s.crs.to_epsg()!=4326:raise ValueError('Native source frame changed')
        inv=~s.transform
        def native(x,y,z=None):
            x,y=to_ll.transform(x,y);return inv.a*np.asarray(x)+inv.b*np.asarray(y)+inv.c,inv.d*np.asarray(x)+inv.e*np.asarray(y)+inv.f
        for rid,support in regions.items():
            ll=transform(to_ll.transform,shapely.segmentize(support.buffer(20).intersection(roi),32.))
            left,bottom,right,top=ll.bounds;corners=[inv*(left,bottom),inv*(right,top)]
            c0=max(0,math.floor(min(p[0] for p in corners)));r0=max(0,math.floor(min(p[1] for p in corners)))
            c1=min(s.width,math.ceil(max(p[0] for p in corners)));r1=min(s.height,math.ceil(max(p[1] for p in corners)))
            window=Window(c0,r0,c1-c0,r1-r0);rgb=np.moveaxis(s.read([1,2,3],window=window),0,-1)
            name=rid+'_NATIVE_SOURCE.png'
            if not cv2.imwrite(str(out/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError('Native packet write failed')
            packet={'region_id':rid,'source_parent_component':rid,'image_path':str(out/name),'image_sha256':sha(out/name),
                'native_rgb_pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'source_window':[c0,r0,c1-c0,r1-r0],
                'local_image_shape':list(rgb.shape[:2]),'source_raster_affine':list(s.transform)[:6],
                'source_raster_crs':'EPSG:4326','native_resampling':'NONE','source_raster_path':str(source),
                'source_observed_date':'2018-11-01','experiment_year':2019,
                'exact_support':mapping(support),'support_is_verified_camera_acquisition_unit':False}
            packets.append(packet)
            for row in [r for r in work['targets'] if r['region_id']==rid]:
                g=objects[row['component_key']];point=transform(native,g.representative_point())
                pixel=transform(native,g);x0,y0,x1,y1=pixel.bounds
                coord=[point.x-c0-.5,point.y-r0-.5]
                bbox=[x0-c0-.5,y0-r0-.5,x1-c0-.5,y1-r0-.5]
                if not 0<=coord[0]<rgb.shape[1] or not 0<=coord[1]<rgb.shape[0]:raise ValueError('Weak source prompt leaves native frame')
                targets.append({**row,'group_id':'FULLROI_'+row['component_key'],'source_packet_id':rid,
                    'point_coords_pixel_index_xy':[coord],'point_labels':[1],'source_box_pixel_index_xyxy':bbox,
                    'box_is_source_prior_not_measured_roof_boundary':True,'reference_point':None,
                    'localization_bound_m':None,'formal_review':False,'full_roof_verified':False})
    save_json(out/'FULLROI_NATIVE_SOURCE_PACKETS.json',{'records':packets})
    save_json(out/'FULLROI_SOURCE_PRIOR_PROMPTS.json',{'records':targets,'role':'WEAK_SOURCE_SEGMENTATION_PROPOSALS_NOT_ROOF_TRUTH'})
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    save_json(out/'FULLROI_PREPARATION_RECEIPT.json',{'source_units':len(packets),'source_targets':len(targets),
        'new_code_sha256':plan['new_code_sha256'],'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'GT_or_target_geometry_read_for_selection':False,
        'formal_gt':0,'inference':'NOT_YET_RUN'})
    print(json.dumps({'source_units':len(packets),'source_targets':len(targets),'excluded':len(work['excluded_components']),'scope':'FULL_FIXED_ROI'}))


if __name__=='__main__':main()
