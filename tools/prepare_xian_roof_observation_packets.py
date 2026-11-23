"""Raw, unshifted native GE packets for model-only roof observation candidates."""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2,numpy as np,rasterio
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment');prior=parent/'SINGLE2019_NATIVE_FRAGMENTS_20261012_14'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh roof-observation pilot required')
    old_plan=json.loads((prior/'FROZEN_NATIVE_FRAGMENT_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'NATIVE_FRAGMENT_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(old_plan['protected_sha256']);protected.update(old_plan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'NATIVE_FRAGMENT_RECEIPT.json')]=sha(prior/'NATIVE_FRAGMENT_RECEIPT.json')
    annex=json.loads((prior/'EVIDENCE_CLOSURE_ANNEX_RECEIPT.json').read_text(encoding='utf-8'))
    protected.update({str(prior/n):h for n,h in annex['new_annex_sha256'].items()})
    protected[str(prior/'EVIDENCE_CLOSURE_ANNEX_RECEIPT.json')]=sha(prior/'EVIDENCE_CLOSURE_ANNEX_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
    plan={'run':out.name,'scope':'BOUNDED_SOURCE_ROOF_OBSERVATION_PILOT_WITHIN_CURRENT_256x256',
        'observation_method':'VISION_MODEL_DIGITIZATION_FROM_RAW_SOURCE_ONLY',
        'annotation_frame':'LOCAL_PIXEL_EDGE_XY_NO_HEADER_NO_RESIZE',
        'source_observed_date':'2018-11-01','experiment_year':2019,
        'manual_GT_or_Sentinel_coordinates_shown_during_source_digitization':False,
        'candidate_displacement_applied':False,'original_mask_replaced':False,
        'human_review_generated':False,'localization_bounds_generated':False,
        'formal_GT_or_clean_roof_certification':False,
        'protected_sha256':protected,'new_code_sha256':{str(Path(__file__).resolve()):sha(__file__)}}
    out.mkdir();save_json(out/'FROZEN_ROOF_OBSERVATION_PLAN.json',plan)
    provenance=json.loads((prior/'NATIVE_VIEW_PROVENANCE.json').read_text(encoding='utf-8'))['records']
    source=Path('F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif');packets=[]
    with rasterio.open(source) as s:
        for row in provenance:
            if row['source_window'][2:]!=list(reversed(row['native_read_out_shape'])):
                raise ValueError('Pilot requires original native pixels, not resized evidence')
            window=Window(*row['source_window']);rgb=np.moveaxis(s.read([1,2,3],window=window),0,-1)
            fingerprint=hashlib.sha256(rgb.tobytes()).hexdigest()
            if fingerprint!=row['native_window_pixel_sha256']:raise ValueError('Original native pixel replay differs')
            name=row['component_key']+'_RAW_GE.png'
            if not cv2.imwrite(str(out/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError('Raw packet PNG failed')
            packets.append({'source_parent_component':row['component_key'],'image_path':str(out/name),
                'image_sha256':sha(out/name),'native_rgb_pixel_sha256':fingerprint,
                'source_raster_path':str(source),'source_window':row['source_window'],
                'source_raster_crs':s.crs.to_string(),'source_raster_affine':list(s.transform)[:6],
                'local_image_shape':list(rgb.shape[:2]),'native_resampling':'NONE',
                'source_observed_date':'2018-11-01','experiment_year':2019,
                'roof_observations':[],'model_source_annotation_status':'PENDING_NOT_COMPLETED'})
    save_json(out/'RAW_SOURCE_PACKETS.json',{'records':packets,'coordinate_convention':'LOCAL_PIXEL_EDGE_XY',
        'world_coordinate_claim':'SOURCE_IMAGE_GEOGRAPHIC_GRID_ONLY_ABSOLUTE_ACCURACY_UNKNOWN'})
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Frozen evidence changed: '+path)
    save_json(out/'RAW_SOURCE_PACKET_RECEIPT.json',{'new_code_sha256':plan['new_code_sha256'],
        'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},
        'source_packets':len(packets),'candidate_vectors_changed':0,'formal_reviews':0,'formal_gt':0})
    print(json.dumps({'run':out.name,'native_packets':len(packets),'observations':'NOT_YET_GENERATED'}))


if __name__=='__main__':main()
