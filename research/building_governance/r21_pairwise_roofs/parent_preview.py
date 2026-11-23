"""Show the entire predecessor object and its selected child in the same extent."""
import json
from pathlib import Path
import numpy as np,rasterio,pyogrio
from scipy import ndimage
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.warp import reproject,Resampling
from PIL import Image,ImageDraw
from prepare import rows,R,save,digest
D=R/'review';A=R/'selective_split/limited_fragmentation'
def main():
    reports=[];panels=[];reg=rows()
    ws={x['id']:x for x in json.loads((A/'POSITION_DETAILS.json').read_text())}
    for case in json.loads((D/'CASES.json').read_text()):
        r=reg[case['window']];fid=case['fid']
        q=next(x['qid'] for x in ws[r['id']]['split'] if x['fid']==fid)
        gt=pyogrio.read_dataframe(Path(r['image_path']).parent/'PROVIDED_MAP_GT.gpkg').to_crs(32649)
        g=gt[(gt.fusion_role=='DESIGNATED_GT2019_ANCHOR')&(gt.original_gt_fid==fid)].geometry.iloc[0]
        with rasterio.open(A/'position'/(r['id']+'.tif')) as ds:
            _,old,new=ds.read();aff=ds.transform;crs=ds.crs
        ids,n=np.unique(old[new==q],return_counts=True);par=int(ids[n.argmax()])
        masks=[old==par,new==q];ref=rasterize([(g,1)],out_shape=old.shape,transform=aff,dtype='uint8')>0
        yy,xx=np.where(masks[0]|masks[1]|ref)
        y0=max(0,int(yy.min())-9);y1=min(len(old),int(yy.max())+10)
        x0=max(0,int(xx.min())-9);x1=min(old.shape[1],int(xx.max())+10)
        win=Window(x0,y0,x1-x0,y1-y0);tr=rasterio.windows.transform(win,aff);crop=np.s_[y0:y1,x0:x1]
        rgb=np.zeros((3,y1-y0,x1-x0),'uint8')
        with rasterio.open(r['image_path']) as ds:
            for k in range(3):reproject(rasterio.band(ds,k+1),rgb[k],src_transform=ds.transform,src_crs=ds.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
        base=rgb.transpose(1,2,0);panel=Image.new('RGB',(960,355),'white');draw=ImageDraw.Draw(panel)
        draw.text((3,3),str(r['source_record']['actual_date'])+' | FID '+str(fid)+' | polygon IoU '+format(case['old_best'],'.3f')+' -> '+format(case['new_best'],'.3f'),fill='black')
        for i in range(3):
            col=base.copy()
            if i:
                m=masks[i-1][crop];color=np.array([85,190,233]) if i==1 else np.array([171,65,194])
                edge=m&~ndimage.binary_erosion(m);col[m]=.62*col[m]+.38*color;col[edge]=color
            z=ref[crop];col[z&~ndimage.binary_erosion(z)]=[255,214,44]
            im=Image.fromarray(col);scale=300/max(im.size);dims=tuple(max(1,round(v*scale)) for v in im.size)
            im=im.resize(dims,Image.Resampling.NEAREST)
            panel.paste(im,(320*i+(320-im.width)//2,48+(302-im.height)//2))
            draw.text((320*i+5,27),['Imagery / reference in yellow','Original full parent','Selected child after split'][i],fill='black')
        p=D/(r['id']+'_PARENT_CHILD.png');panel.save(p);reports.append(dict(window=r['id'],fid=fid,parent=par,child=q,file=p.name,sha=digest(p),before_iou=case['old_best'],after_iou=case['new_best']));panels.append(panel)
    p=D/'PARENT_CHILD_COMPARISON.png';img=Image.new('RGB',(960,355*len(panels)),'white')
    for i,panel in enumerate(panels):img.paste(panel,(0,355*i))
    img.save(p);save(D/'PARENT_CHILD.json',dict(cases=reports,sha=digest(p)))
if __name__=='__main__':main()

