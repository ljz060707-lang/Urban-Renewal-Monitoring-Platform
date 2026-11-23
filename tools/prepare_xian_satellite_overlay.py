"""Plotting-only real imagery / frozen polygon transfer; never writes labels."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import fiona, numpy as np, rasterio
from rasterio.features import shapes, rasterize
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling, transform_bounds
from affine import Affine
from shapely.geometry import shape, box, mapping
from shapely.affinity import translate
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_xian_source_partition_alignment import sha, save_json
P = Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
OUT = P / 'SINGLE2019_SATELLITE_OVERLAY_CASES_20261012_29'
GE = Path('F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif')
S2 = Path('F:/xian/10m底图/Xian_2019_TrueColor.tif')
CURRENT = P / 'SINGLE2019_RISK_QUARANTINED_WINDOW_20261012_25'
COMP = P / 'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
SUPPORT = P / 'SINGLE2019_COHERENT_TRANSFER_20261012_12/DERIVED_SUPPORT_LINEAGE.json'

def rings(g):
    return [list(p.exterior.coords) for p in ([g] if g.geom_type == 'Polygon' else g.geoms)]

def photos(bounds, key):
    """Real RGB only: GE reprojected for display, S2 native 10m grid unchanged."""
    x0,y0,x1,y1 = bounds
    n = int(round((x1-x0)/0.25)); h = int(round((y1-y0)/0.25))
    dst_aff = Affine(0.25,0,x0,0,-0.25,y1)
    with rasterio.open(GE) as src:
        ll = transform_bounds(32649,src.crs,*bounds,densify_pts=21)
        inv = ~src.transform
        corners = [inv*(ll[0],ll[3]),inv*(ll[2],ll[1])]
        c0 = max(0,math.floor(min(c[0] for c in corners))-3)
        r0 = max(0,math.floor(min(c[1] for c in corners))-3)
        c1 = min(src.width,math.ceil(max(c[0] for c in corners))+3)
        r1 = min(src.height,math.ceil(max(c[1] for c in corners))+3)
        win = Window(c0,r0,c1-c0,r1-r0); native = src.read(window=win)
        aff = src.window_transform(win)
        dst = np.zeros((3,h,n),dtype=np.uint8)
        for b in range(3):
            reproject(native[b],dst[b],src_transform=aff,src_crs=src.crs,
                dst_transform=dst_aff,dst_crs=32649,resampling=Resampling.bilinear)
        native_hash = hashlib.sha256(native.tobytes()).hexdigest()
        native_meta={'native_window':[c0,r0,c1-c0,r1-r0],
            'native_window_array_sha256':native_hash,'native_crs':str(src.crs),
            'native_window_affine':list(aff)[:6]}
    Image.fromarray(np.moveaxis(dst,0,-1)).save(OUT/(key+'_GE_DISPLAY.png'))
    with rasterio.open(S2) as src:
        inv=~src.transform; col,row=inv*(x0,y1)
        win=Window(round(col),round(row),round((x1-x0)/10),round((y1-y0)/10))
        arr=src.read(window=win)
        assert src.window_transform(win) == Affine(10,0,x0,0,-10,y1)
        Image.fromarray(np.moveaxis(arr,0,-1)).save(OUT/(key+'_SENTINEL_NATIVE_10M.png'))
        native_meta['sentinel_window']=[int(win.col_off),int(win.row_off),int(win.width),int(win.height)]
    return native_meta

def shortlist():
    if OUT.exists(): raise ValueError('Fresh append-only output required')
    protected = {str(p):sha(p) for p in [COMP,SUPPORT,S2,
        CURRENT/'2019_FROZEN_CANDIDATE_FIELD.tif',CURRENT/'2019_RECOMMENDED_PSEUDO_LABEL.tif',
        CURRENT/'2019_RECOMMENDED_PSEUDO_WEIGHT.tif',CURRENT/'2019_TARGET_ALIGNMENT_RISK_QUARANTINE.tif',
        CURRENT/'2019_SOURCE_ALIGNMENT_RISK_DIAGNOSTIC.tif']}
    support=json.loads(SUPPORT.read_text(encoding='utf-8'))
    regs=[(g['region_id'],shape(g['geometry']),g['correction_m']) for g in support['derived_geometries'] if g['correction_m'] is not None]
    with rasterio.open(CURRENT/'2019_SOURCE_ALIGNMENT_RISK_DIAGNOSTIC.tif') as src:
        a=src.read(1); aff=src.transform
        source_risks=[shape(g) for g,v in shapes((a>0).astype('uint8'),transform=aff) if v==1]
    with rasterio.open(CURRENT/'2019_TARGET_ALIGNMENT_RISK_QUARANTINE.tif') as src:
        a=src.read(1); target_risks=[shape(g) for g,v in shapes((a>0).astype('uint8'),transform=aff) if v==1]
    with rasterio.open(CURRENT/'2019_RECOMMENDED_PSEUDO_LABEL.tif') as src:
        label=src.read(1); aff=src.transform
    with rasterio.open(CURRENT/'2019_FROZEN_CANDIDATE_FIELD.tif') as src: field=src.read()
    objects=[]
    with fiona.open(COMP,layer='native_positive_components') as src:
        for f in src: objects.append((str(f.properties['component_key']),shape(f.geometry),dict(f.properties)))
    candidates=[]
    for key,g,props in objects:
        parts=[g] if g.geom_type=='Polygon' else list(g.geoms)
        if len(parts)!=1:continue
        p=parts[0]
        if not p.is_valid or len(p.interiors)>0 or not 400<=p.area<=4000:continue
        rect=p.area/p.minimum_rotated_rectangle.area
        if rect<0.86:continue
        for rid,sup,vec in regs:
            if not sup.covers(p):continue
            after=translate(p,xoff=vec[0],yoff=vec[1])
            if not sup.covers(after):continue
            if any(p.intersects(r) for r in source_risks) or any(after.intersects(r) for r in target_risks):continue
            mask=rasterize([(mapping(after),1)],out_shape=label.shape,transform=aff,all_touched=True).astype(bool)
            positives=int(((label==1)&mask).sum())
            if positives<3:continue
            if not np.allclose(field[:,mask],np.array(vec)[:,None],rtol=0,atol=0):raise ValueError('Frozen vector mismatch')
            neighbours=[o.distance(p) for k,o,_ in objects if k!=key and o.bounds[0]<p.bounds[2]+8 and o.bounds[2]>p.bounds[0]-8 and o.bounds[1]<p.bounds[3]+8 and o.bounds[3]>p.bounds[1]-8]
            distance=min(neighbours,default=8.)
            if distance<1.5:continue
            bb=p.union(after).bounds
            cx=(bb[0]+bb[2])/2;cy=(bb[1]+bb[3])/2
            extent=math.ceil(max(bb[2]-bb[0],bb[3]-bb[1])+40)
            size=int(math.ceil(extent/10))*10
            x0=math.floor((cx-size/2-293300)/10)*10+293300
            y0=math.floor((cy-size/2-3812400)/10)*10+3812400
            bounds=[x0,y0,x0+size,y0+size]
            candidates.append({'key':key,'region_id':rid,'source_properties':props,
                'vector_m':vec,'before_rings_utm':rings(p),'after_rings_utm':rings(after),
                'before_wkb_sha256':hashlib.sha256(p.wkb).hexdigest(),
                'after_wkb_sha256':hashlib.sha256(after.wkb).hexdigest(),
                'area_m2':p.area,'rectangularity':rect,'nearest_component_distance_m':distance,
                'recommended_positive_pixels_intersecting_after':positives,'bounds_utm':bounds,
                'source_risk_overlap':False,'target_risk_overlap':False,
                'source_shape_score':rect+min(distance,8)/80,
                'vector_was_refit':False,'gt_was_used_for_visual_selection':False})
    # Shape clarity alone, spatially separated; never rank by GT or shift improvement.
    candidates.sort(key=lambda q:(-q['source_shape_score'],q['key']))
    chosen=[];counts={}
    for q in candidates:
        if counts.get(q['region_id'],0)>=4:continue
        b=box(*q['bounds_utm'])
        if any(b.intersection(box(*old['bounds_utm'])).area>b.area*0.4 for old in chosen):continue
        chosen.append(q);counts[q['region_id']]=counts.get(q['region_id'],0)+1
        if len(chosen)>=12:break
    OUT.mkdir()
    for q in chosen:q['photo_provenance']=photos(q['bounds_utm'],q['key'])
    save_json(OUT/'SATELLITE_OVERLAY_SHORTLIST.json',{'candidates':chosen,
        'selection_basis':'SOURCE_SINGLE_COMPONENT_BOUNDARY_CLARITY_NOT_CORRECTION_ACCURACY',
        'protected_sha256':protected,'source_actual_date':'2018-11-01','experiment_nominal_year':2019,
        'ge_source_path':str(GE),'ge_source_file_stat':{'size':GE.stat().st_size,'mtime_ns':GE.stat().st_mtime_ns},
        'reference_path':str(S2),'crs':'EPSG:32649','display_GE_resolution_m':0.25,
        'GE_DISPLAY_REPROJECTION_IS_NOT_AN_ALIGNMENT_CORRECTION':True,
        'formal_alignment_accepted':False,'ground_footprints':0,'absolute_position_accuracy':'UNKNOWN',
        'new_labels':0,'new_vectors':0,'code_sha256':sha(Path(__file__))})
    # Native photo contact sheet: no GT overlay or alignment outcome shown.
    sheet=Image.new('RGB',(1200,math.ceil(len(chosen)/4)*330),'white')
    draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,q in enumerate(chosen):
        im=Image.open(OUT/(q['key']+'_GE_DISPLAY.png')).resize((290,290))
        x=(i%4)*300;y=(i//4)*330;sheet.paste(im,(x,y+30))
        draw.text((x+3,y+2),str(i+1)+' '+q['key'].replace('RASTER_COMPONENT_',''),font=font,fill='black')
    sheet.save(OUT/'SOURCE_ONLY_SELECTION_CONTACT.png')
    print(json.dumps({'shortlisted':len(chosen),'keys':[q['key'] for q in chosen]}))

if __name__=='__main__':shortlist()
