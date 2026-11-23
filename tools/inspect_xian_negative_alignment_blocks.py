"""Native imagery and exact fixed-cohort traces for known-negative blocks."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import cv2,fiona,numpy as np,rasterio
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import shape,box,Polygon
from shapely.ops import transform
from pyproj import Transformer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.alignment.source_template_alignment import _sample_target
from xian_r5.alignment.base_tie import _pearson_correlation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh native evidence run required')
    prior=parent/'SINGLE2019_RISK_QUARANTINED_WINDOW_20261012_25'
    oldplan=json.loads((prior/'FROZEN_RISK_QUARANTINE_PLAN.json').read_text(encoding='utf-8'))
    oldreceipt=json.loads((prior/'RISK_QUARANTINE_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(oldplan['protected_sha256']);protected.update(oldplan['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in oldreceipt['output_sha256'].items()})
    protected[str(prior/'RISK_QUARANTINE_RECEIPT.json')]=sha(prior/'RISK_QUARANTINE_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input drift: '+path)
    cohort_root=parent/'SINGLE2019_REFERENCE_CORE_REPLAY_20261012_24_02'
    replay=json.loads((cohort_root/'REFERENCE_CORE_ALIGNMENT_REPLAY.json').read_text(encoding='utf-8'))
    records={r['region_id']:r for r in replay['records']}
    failures=json.loads((prior/'RISK_BLOCK_SOURCE_TARGET_LINEAGE.json').read_text(encoding='utf-8'))['records']
    cases=[r for r in failures if r['target_pixel_window'] is not None]
    if len(cases)!=2:raise ValueError('Known-negative mapped block census changed')
    out.mkdir();save_json(out/'FROZEN_NATIVE_RISK_EVIDENCE_PLAN.json',{
        'scope':'TWO_KNOWN_NEGATIVE_BLOCKS_WITHIN_FULL_CURRENT_WINDOW_NOT_FAVORABLE_SAMPLE_SELECTION',
        'source_native_resampling':'NONE','visual_context_m':20,
        'no_new_mask_geometry_or_vector_or_label':True,
        'feature_sample_pixels_are_NOT_formal_ground_ties':True,
        'source_actual_date':'2018-11-01','experiment_year':2019,
        'human_review_or_localization_bounds_created':False,
        'protected_sha256':protected,'new_code_sha256':{str(Path(__file__).resolve()):sha(__file__)}})
    with rasterio.open(cohort_root/'SOURCE_REFERENCE_CORE_COHORT.tif') as s:mask=s.read(1).astype(bool)
    with rasterio.open(parent/'SINGLE2019_SUPPORT_REPAIR_20261012_07/2019_BOUNDED_GE_RGB_CACHE.tif') as s:
        source=np.moveaxis(s.read(),0,-1);grid=s.transform
    with rasterio.open('F:/xian/10m底图/Xian_2019_TrueColor.tif') as s:
        roiwindow=Window(1516,1765,256,256);target=np.moveaxis(s.read([1,2,3],window=roiwindow),0,-1)
        target_valid=(s.read_masks([1,2,3],window=roiwindow)>0).all(axis=0)
    with rasterio.open(parent/'SINGLE2019_20261012_06/2019_ROOF_FRACTION_RAW.tif') as s:raw=s.read(1)
    with rasterio.open(prior/'2019_MANUAL_REFERENCE_ROOF_FRACTION.tif') as s:reference=s.read(1)
    with rasterio.open(prior/'2019_ROOF_FRACTION_CANDIDATE.tif') as s:candidate=s.read(1)
    ll=Transformer.from_crs(32649,4326,always_xy=True);metric=Transformer.from_crs(4326,32649,always_xy=True)
    sourcepath=Path('F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif')
    gtpath=Path('F:/xian/西安建筑轮廓数据/西安.shp')
    teacherpath=Path('D:/weixin chat/xwechat_files/wxid_w9xgzlr6hnvp22_40c8/msg/file/2026-10/2019_BUILDING_BOUNDARY_ge100m2.tif')
    yy,xx=np.indices((256,256));check=(yy//16+xx//16)%2==1;packets=[]
    def png(path,rgb):
        if not cv2.imwrite(str(path),cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2BGR)):
            raise ValueError('Evidence PNG write failed')
    with rasterio.open(sourcepath) as native,rasterio.open(teacherpath) as teacher:
        if teacher.transform!=native.transform or teacher.shape!=native.shape:raise ValueError('Native source/teacher grids differ')
        inv=~native.transform
        for case in cases:
            rid=case['region_id'];sc,sr,w,h=case['source_pixel_window'];tc,tr,_,_=case['target_pixel_window']
            rr,cc=(int(v[1:]) for v in rid.split('_')[1:]);unit=np.zeros((256,256),bool)
            unit[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
            common=mask&unit&check&np.isfinite(source).all(axis=2)
            tv=target_valid&unit&check&np.isfinite(target).all(axis=2)
            for dx in range(-3,4):
                for dy in range(-3,4):common&=_sample_target(tv[None],dx,dy)[0]
            block=np.zeros_like(common);block[sr:sr+16,sc:sc+16]=True;common&=block
            rows,cols=np.where(common)
            dx=int(case['mapping_vector_m'][0]/10);dy=int(-case['mapping_vector_m'][1]/10)
            a=source[rows,cols];z=target[rows,cols];c=target[rows+dy,cols+dx]
            gains=[_pearson_correlation(a[:,k],c[:,k])-_pearson_correlation(a[:,k],z[:,k]) for k in range(3)]
            gain=float(np.mean(gains))
            if len(rows)!=case['distinct_source_samples'] or abs(gain-case['check_gain'])>1e-12:
                raise ValueError('Fixed-cohort negative block replay differs')
            bounds=box(*sorted([])) if False else box(*(grid*(sc,sr+16)),*(grid*(sc+16,sr)))
            context=bounds.buffer(20,cap_style=3,join_style=2)
            llpoly=transform(ll.transform,context.segmentize(10));left,bottom,right,top=llpoly.bounds
            p0=inv*(left,top);p1=inv*(right,bottom)
            c0,r0=math.floor(min(p0[0],p1[0])),math.floor(min(p0[1],p1[1]))
            c1,r1=math.ceil(max(p0[0],p1[0])),math.ceil(max(p0[1],p1[1]))
            window=Window(c0,r0,c1-c0,r1-r0);rgb=np.moveaxis(native.read([1,2,3],window=window),0,-1)
            if min(c0,r0)<0 or c1>native.width or r1>native.height:raise ValueError('Native context outside source')
            png(out/(rid+'_GE_NATIVE_RAW.png'),rgb)
            overlay=rgb.copy();binary=teacher.read(1,window=window)==1
            contours,_=cv2.findContours(binary.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(50,255,50),1)
            png(out/(rid+'_GE_NATIVE_TEACHER_OUTLINE.png'),overlay)
            ref_overlay=rgb.copy();fids=[]
            def native_coords(x,y,z=None):
                x,y=ll.transform(x,y);return inv.a*np.asarray(x)+inv.b*np.asarray(y)+inv.c-c0,inv.d*np.asarray(x)+inv.e*np.asarray(y)+inv.f-r0
            with fiona.open(gtpath) as s:
                for f in s.filter(bbox=llpoly.bounds):
                    g=transform(metric.transform,shape(f.geometry))
                    if not g.intersects(context):continue
                    fids.append(str(f.id));local=transform(native_coords,g)
                    parts=[local] if local.geom_type=='Polygon' else list(local.geoms)
                    for part in parts:
                        for ring in [part.exterior,*part.interiors]:
                            points=np.rint(np.array(ring.coords)).astype(np.int32)
                            cv2.polylines(ref_overlay,[points],True,(255,80,80),1)
            png(out/(rid+'_GE_NATIVE_MANUAL_REFERENCE_OVERLAY.png'),ref_overlay)
            strip=[]
            for title,image,r,cx in [('GE_10M_SOURCE',source,sr,sc),('S2_ZERO',target,sr,sc),('S2_FROZEN_SHIFT',target,tr,tc)]:
                crop=np.clip(image[r-2:r+18,cx-2:cx+18],0,255).astype(np.uint8)
                png(out/(rid+'_'+title+'_NATIVE_20x20.png'),crop)
                display=cv2.resize(crop,(400,400),interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(display,(40,40),(359,359),(255,255,0),1)
                for py,px in zip(rows-sr+2,cols-sc+2):
                    cv2.circle(display,(int(px*20+10),int(py*20+10)),3,(255,80,80),-1)
                header=np.zeros((28,400,3),np.uint8);cv2.putText(header,title,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                strip.append(np.concatenate([header,display],axis=0))
            png(out/(rid+'_10M_SAMPLE_COMPARISON_DISPLAY_ONLY.png'),np.concatenate(strip,axis=1))
            sel=np.s_[tr:tr+16,tc:tc+16];sample_trace=[]
            for i,(y,x) in enumerate(zip(rows,cols)):
                sample_trace.append({'source_row':int(y),'source_col':int(x),'target_row':int(y+dy),'target_col':int(x+dx),
                    'source_RGB':a[i].tolist(),'zero_target_RGB':z[i].tolist(),'frozen_candidate_target_RGB':c[i].tolist(),
                    'not_a_registration_tie':True})
            packet={'region_id':rid,'source_native_window':[c0,r0,c1-c0,r1-r0],
                'native_raw_png':rid+'_GE_NATIVE_RAW.png','native_raw_png_sha256':sha(out/(rid+'_GE_NATIVE_RAW.png')),
                'source_native_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'native_resampling':'NONE',
                'source_raster_affine':list(native.transform)[:6],'source_raster_crs':str(native.crs),
                'source_actual_date':'2018-11-01','experiment_year':2019,
                'source_block_pixel_window':case['source_pixel_window'],'target_block_pixel_window':case['target_pixel_window'],
                'frozen_vector_m':case['mapping_vector_m'],'reference_fids_in_context':fids,
                'per_band_check_gain':gains,'mean_check_gain_recomputed':gain,
                'raw_fraction_reference_MAE_same_target_cells':float(np.abs(raw[sel].astype(float)-reference[sel]).mean()),
                'candidate_fraction_reference_MAE_same_target_cells':float(np.abs(candidate[sel].astype(float)-reference[sel]).mean()),
                'fixed_feature_sample_trace':sample_trace,'inspection_conclusion':'NOT_YET_INSPECTED',
                'new_geometry_vector_or_labels':False,'absolute_position_accuracy':'UNKNOWN'}
            packets.append(packet)
    save_json(out/'NATIVE_NEGATIVE_BLOCK_EVIDENCE.json',{'records':packets,'coordinate_ties_or_bounds_created':False,
        'GT_overlays_are_diagnostic_NOT_source_digitization_prompts':True})
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input changed: '+path)
    save_json(out/'NATIVE_NEGATIVE_BLOCK_PACKET_RECEIPT.json',{'packets':len(packets),'sample_traces':sum(len(r['fixed_feature_sample_trace']) for r in packets),
        'new_code_sha256':{str(Path(__file__).resolve()):sha(__file__)},'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'new_geometry_vectors_labels':0})
    print(json.dumps({'packets':len(packets),'native_source_shapes':[r['source_native_window'][2:] for r in packets]}))


if __name__=='__main__':main()
