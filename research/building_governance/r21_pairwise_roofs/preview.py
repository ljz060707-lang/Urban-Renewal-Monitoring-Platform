from pathlib import Path
import json
import numpy as np,rasterio,pyogrio
from rasterio.warp import reproject,Resampling
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage
from PIL import Image,ImageDraw
from prepare import R,rows,save,digest
A=R/'selective_split/limited_fragmentation';D=R/'review';D.mkdir(exist_ok=True)
def main():
    ws=json.loads((A/'POSITION_DETAILS.json').read_text());database=rows();reports=[];pics=[]
    for w in ws:
        for a,z in zip(w['baseline'],w['split']):
            if a['match'] or not z['match']:continue
            r=database[w['id']];gt=pyogrio.read_dataframe(Path(r['image_path']).parent/'PROVIDED_MAP_GT.gpkg').to_crs(32649)
            g=gt[(gt.fusion_role=='DESIGNATED_GT2019_ANCHOR')&(gt.original_gt_fid==a['fid'])].geometry.iloc[0]
            src=A/'position'/(w['id']+'.tif')
            with rasterio.open(src) as ds:
                cy,cx=ds.index(g.centroid.x,g.centroid.y);side=min(ds.width,ds.height,max(66,int(max(g.bounds[2]-g.bounds[0],g.bounds[3]-g.bounds[1])+45)))
                win=Window(np.clip(cx-side//2,0,ds.width-side),np.clip(cy-side//2,0,ds.height-side),side,side)
                transforms=ds.window_transform(win);labs=ds.read([2,3],window=win);crs=ds.crs
            rgb=np.zeros((3,side,side),'uint8')
            with rasterio.open(r['image_path']) as ds:
                for k in range(3):reproject(rasterio.band(ds,k+1),rgb[k],src_transform=ds.transform,src_crs=ds.crs,dst_transform=transforms,dst_crs=crs,resampling=Resampling.bilinear)
            ref=rasterize([(g,1)],out_shape=(side,side),transform=transforms,dtype='uint8')
            edge=ref.astype(bool)&~ndimage.binary_erosion(ref)
            image=rgb.transpose(1,2,0);panel=Image.new('RGB',(900,326),'white');draw=ImageDraw.Draw(panel)
            for i in range(3):
                base=image.copy()
                if i:
                    lab=labs[i-1];m=lab>0
                    b=m&(ndimage.minimum_filter(lab,size=3)!=ndimage.maximum_filter(lab,size=3))
                    base[m]=(.78*base[m]+.22*np.array([145,58,170])).astype('uint8');base[b]=[173,70,201]
                base[edge]=[255,214,40]
                panel.paste(Image.fromarray(base).resize((300,300),Image.Resampling.NEAREST),(i*300,26))
                label=['RGB and designated GT','Existing date-field IDs','R21 split and date-field IDs'][i]
                draw.text((i*300+5,6),label,fill='black')
            path=D/(w['id']+'_FID'+str(a['fid'])+'.png');panel.save(path)
            pics.append(panel)
            reports.append(dict(window=w['id'],fid=a['fid'],old_best=a['iou'],new_best=z['iou'],png=path.name,sha=digest(path),actual_dates=r['source_record']['actual_date_pixel_votes'],instance_TIF=src.name))
    if pics:
        full=Image.new('RGB',(900,346*len(pics)),'white');draw=ImageDraw.Draw(full)
        for i,image in enumerate(pics):
            draw.text((3,i*346+2),reports[i]['window']+' | FID '+str(reports[i]['fid']),fill='black')
            full.paste(image,(0,i*346+20))
        full.save(D/'GAINED_MATCHES.png')
    save(D/'CASES.json',reports)
if __name__=='__main__':main()

