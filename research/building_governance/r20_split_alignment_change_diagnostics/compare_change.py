"""Compare binary evidence with prior roof occupancy on the very same supported pixels."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
from PIL import Image,ImageDraw
from diagnose import check,input_rows,save
R=Path(__file__).resolve().parent; out=R/'change_evidence'
def main():
    rows={(r['site'],r['epoch']):r for r in input_rows()}
    rec=json.loads((out/'RECEIPT.json').read_text())
    eps=sorted({k[1] for k in rows})
    site_images=[];comparisons=[]
    for v in rec['records']:
        i,j=v['pair'];a,b=(rows[v['site'],eps[k-1]] for k in (i,j))
        roof=[];rgb=[]
        case=v['site'] in ('DENSE_REFERENCE_00','RANDOM_02','RANDOM_03') and [i,j]==[2,3]
        for r in (a,b):
            check(r['query_cache_path'],r['query_cache_sha256'])
            with np.load(r['query_cache_path']) as z:roof.append(z['instances']>0)
            if case:
                with rasterio.open(r['image_path']) as ds:rgb.append(ds.read([1,2,3]).transpose(1,2,0))
        path=out/v['file'];check(path,v['sha256'])
        with rasterio.open(path) as ds:p=ds.read(1)
        valid=p>=0;change=(p>=.5)&valid;xor=(roof[0]^roof[1])&valid
        groups=[~roof[0]&~roof[1],roof[0]&~roof[1],~roof[0]&roof[1],roof[0]&roof[1]]
        comparisons.append(dict(site=v['site'],pair=[i,j],supported=int(valid.sum()),binary=int(change.sum()),roof_xor=int(xor.sum()),
            intersection=int((change&xor).sum()),binary_by_raw_occupancy_00_10_01_11=[int((change&g).sum()) for g in groups]))
        if case:
            over=rgb[1].copy();over[change]=(.55*over[change]+.45*np.array([133,46,125])).astype('uint8')
            panel=Image.new('RGB',(3*320,346),'white');draw=ImageDraw.Draw(panel)
            for k,img in enumerate(rgb+[over]):panel.paste(Image.fromarray(img).resize((320,320),Image.Resampling.LANCZOS),(320*k,26))
            for k,txt in enumerate([str(a['source_record']['actual_date']),str(b['source_record']['actual_date']),'WHU binary response >=0.50']):
                draw.text((320*k+4,6),v['site']+'  '+txt,fill='black')
            site_images.append(panel)
    pi=Image.new('RGB',(960,346*len(site_images)),'white')
    for i,im in enumerate(site_images):pi.paste(im,(0,346*i))
    pi.save(out/'THREE_SITE_CHANGE.jpg',quality=93)
    save(out/'RAW_OCCUPANCY_COMPARISON.json',comparisons)
if __name__=='__main__':main()

