"""RGB visibility diagnostics for complete development windows, kept separate from geometry and temporal labels."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyogrio
import rasterio
from affine import Affine
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import rasterize
from scipy import ndimage

R=Path('/root/xian20261111_r17_proposals')
D=R/'output/observability'
M=Path('/root/xian20261111_r8/QUERY_CACHE_RECORDS.jsonl')
PRE=Path('/root/xian20261111_r16_roi_roof/output/xian_map_reference/BASELINE_OBJECTS.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args):
    row,previous=args
    s=row['source_record'];native=Path(row['image_path']).parent
    assert sha(row['image_path'])==row['image_sha256'] and sha(row['support_path'])==s['files']['SOURCE_SUPPORT.tif']['sha256']
    assert sha(native/'PROVIDED_MAP_GT.gpkg')==s['files']['PROVIDED_MAP_GT.gpkg']['sha256']
    with rasterio.open(row['image_path']) as im,rasterio.open(row['support_path']) as mask:
        rgb=im.read([1,2,3]).transpose(1,2,0);support=mask.read(1)>0;tr=im.transform;crs=im.crs
    y=rgb.astype('float32')@np.array([.299,.587,.114],np.float32)
    cells=np.full((4,16,16),-1,dtype=np.float32)
    for i,j in np.ndindex((16,16)):
        ys=slice(i*64,(i+1)*64);xs=slice(j*64,(j+1)*64)
        block=y[ys,xs];valid=support[ys,xs]
        if valid.sum()<2048:continue
        lo,md,hi=np.percentile(block[valid],[10,50,90])
        vx=valid[:,1:]&valid[:,:-1];vy=valid[1:]&valid[:-1]
        grad=(np.abs(np.diff(block,axis=1))[vx].sum()+np.abs(np.diff(block,axis=0))[vy].sum())/max(1,vx.sum()+vy.sum())
        cells[:,i,j]=[md,hi-lo,grad,float(md>=145 and hi-lo<=36 and grad<=9)]
    dest=D/row['id'];dest.mkdir(parents=True,exist_ok=True)
    tif=dest/'RGB_LOCAL_VISIBILITY_64PX.tif';bands=['median_luminance','luminance_p90_minus_p10','mean_neighbor_absdiff','bright_flat_diagnostic']
    with rasterio.open(tif,'w',driver='GTiff',width=16,height=16,count=4,dtype='float32',nodata=-1,crs=crs,transform=tr*Affine.scale(64),compress='deflate') as f:
        f.write(cells)
        for i,b in enumerate(bands,1):f.set_band_description(i,b)
    with rasterio.open(tif) as f:assert np.array_equal(f.read(),cells) and f.descriptions==tuple(bands)
    flat=np.repeat(np.repeat(cells[3]==1,64,0),64,1)
    # This join follows image-only filtering; reference status is a separately saved field.
    info={r['fid']:r for r in previous['details']['RAW']}
    ref=pyogrio.read_dataframe(native/'PROVIDED_MAP_GT.gpkg').to_crs(crs)
    ref=ref[ref.fusion_role=='DESIGNATED_GT2019_ANCHOR']
    flag=[];edge=np.zeros((1024,1024),bool)
    for fid,poly in zip(ref.original_gt_fid,ref.geometry):
        if fid not in info:continue
        covered=rasterize([(poly,1)],out_shape=support.shape,transform=tr,dtype='uint8').astype(bool)&support
        assert int(covered.sum())==info[fid]['supported_reference_pixels']
        fraction=float(flat[covered].mean())
        flag.append(dict(fid=int(fid),raw_zero_overlap=info[fid]['covered_pixels']==0,fraction_reference_in_bright_flat_blocks=fraction))
        edge|=covered^ndimage.binary_erosion(covered,iterations=2)
    if row['site'] in {'DENSE_REFERENCE_00','RANDOM_02','RANDOM_03'}:
        assert sha(row['query_cache_path'])==row['query_cache_sha256']
        with np.load(row['query_cache_path']) as z:pred=z['instances']>0
        contour=pred^ndimage.binary_erosion(pred,iterations=2)
        overlay=rgb.copy();overlay[edge]=[255,211,36];overlay[contour]=[171,83,231]
        for suffix,im in [('rgb',rgb),('overlay',overlay)]:
            Image.fromarray(im).resize((428,428),Image.Resampling.LANCZOS).save(dest/(suffix+'.png'))
    return dict(frame=row['id'],site=row['site'],epoch=row['epoch'],dates=list(s['actual_date_pixel_votes']),sha256=sha(tif),
        relative_file=str(tif.relative_to(D)),flat_block_fraction=float((cells[3]==1).sum()/max(1,(cells[3]>=0).sum())),rows=flag)
def main():
    D.mkdir(parents=True,exist_ok=True)
    previous={r['window']:r for r in json.loads(PRE.read_text())}
    windows=[r for line in M.read_text().splitlines() if (r:=json.loads(line))['role']=='TUNE']
    assert len(windows)==72 and set(previous)=={r['id'] for r in windows}
    with ProcessPoolExecutor(6) as pool:rows=list(pool.map(run,[(r,previous[r['id']]) for r in windows]))
    counts={}
    for ep in sorted({r['epoch'] for r in rows}):
        items=[t for r in rows if r['epoch']==ep for t in r['rows']]
        counts[ep]=dict(map_observations=len(items),zero_raw_overlap=sum(t['raw_zero_overlap'] for t in items),
            majority_bright_flat=sum(t['fraction_reference_in_bright_flat_blocks']>=.5 for t in items),
            zero_and_majority_bright_flat=sum(t['raw_zero_overlap'] and t['fraction_reference_in_bright_flat_blocks']>=.5 for t in items))
    report=dict(summary=counts,rows=rows,scope='72 reused target development windows; exploratory bright/low-detail flag, not cloud labels, certified visibility, building absence or native roof truth',
        input='64x64 native RGB blocks; supported area >=50%; grayscale median >=145 and p90-p10<=36 and adjacent mean absolute difference<=9',
        inspected='DENSE_REFERENCE_00 shows change in surface appearance around a retained sports ground. RANDOM_02 and RANDOM_03 show bright low-detail patches. No formal building presence or demolition labels assigned.',
        prior_baseline_sha256=sha(PRE),production_changed=False,protected_reads=0)
    (D/'DIAGNOSTIC.json').write_text(json.dumps(report,indent=2))
    sites=['DENSE_REFERENCE_00','RANDOM_02','RANDOM_03'];epochs=['xa180429','xa190101','xa201231']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for mode in ['rgb','overlay']:
        canvas=Image.new('RGB',(1284,3*463+32),'white');draw=ImageDraw.Draw(canvas)
        for i,site in enumerate(sites):
            for j,ep in enumerate(epochs):
                record=next(r for r in rows if r['site']==site and r['epoch']==ep)
                tile=Image.open(D/record['frame']/(mode+'.png'))
                canvas.paste(tile,(j*428,i*463))
                draw.text((j*428+5,i*463+432),site+' | '+', '.join(record['dates']),font=font,fill='black')
        caption='Yellow: 2019 map references. Purple: raw roof candidates.' if mode=='overlay' else 'Actual acquisition dates, same 3 locations. Viewed development windows.'
        draw.text((6,3*463+1),caption,font=font,fill='black')
        canvas.save(D/f'THREE_SITE_{mode.upper()}.jpg',quality=94)
    print(json.dumps(counts))
if __name__=='__main__': main()


