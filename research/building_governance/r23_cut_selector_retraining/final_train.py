import sys,json,os,csv
from pathlib import Path
P=Path(__file__).resolve().parent
R22=Path('/root/xian20261111_r22_cuts');sys.path.insert(0,str(R22))
from cuts import OLD,D,save,verify,digest
import prepare as prep
from prepare import rows,init_map,partition
from learn import Net
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np,torch,rasterio
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.affinity import translate
from shapely.ops import transform
from rasterio.features import rasterize
sys.path.insert(0,'/root/xian20261111_r13/code')
from multiview_reference_model import infer
fieldfile=Path('/root/xian20261111_r13/output/BASELINE_PARAMETERS.json')
field=json.loads(fieldfile.read_text())
def target(inp):
    rec,r=inp;k=r['id'];role=rec['role'];af=Affine(*r['native_affine'])
    to=Transformer.from_crs(r['native_crs'],32649,always_xy=True).transform
    back=Transformer.from_crs(32649,r['native_crs'],always_xy=True).transform
    roi=Polygon([to(*(af*p)) for p in [(0,0),(1024,0),(1024,1024),(0,1024)]])
    with rasterio.open(r['date_path']) as ds:dates=ds.read(1)
    with np.load(r['query_cache_path']) as f:raw=f['instances']
    with np.load(OLD/'data'/(k+'.npz')) as f:boxes=f['boxes']
    with np.load(D/(k+'.npz')) as f:groups=f['groups'];masks=f['masks'];feat=f['feat']
    # Use only nonedge, assigned-role polygons of known date; ambiguous and absent labels ignored.
    known=[]
    for o in prep.MAPS.iloc[prep.TREE.query(roi,predicate='intersects')].itertuples():
        g=o.geometry;fid=int(o.original_gt_fid);c=g.centroid;grp=(int(c.x//2000),int(c.y//2000))
        if roi.contains(g) and (grp[0]*31+grp[1])%5!=0 and 'GT_'+str(fid) not in prep.LEGACY and partition(grp)==(role=='MAP_INNER'):known.append((g,fid))
    labels=np.full((len(groups),2),-1,np.int8)
    for i in np.unique(groups):
        q,y0,y1,x0,x1=boxes[i];ii=np.flatnonzero(groups==i);mask=raw[y0:y1,x0:x1]==q;yy,xx=np.where(mask);codes=np.unique(dates[y0:y1,x0:x1][mask])
        if len(codes)!=1 or codes[0] not in (1,2):continue
        date={1:'2019-12-02',2:'2019-12-10'}[int(codes[0])]
        pt=np.array(to(*(af*(x0+xx.mean()+.5,y0+yy.mean()+.5))));pred,_=infer({date:pt},field);dx,dy=pred[date]-pt
        poly=translate(Polygon([to(*(af*p)) for p in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]]),dx,dy)
        ca=np.unpackbits(masks[ii],axis=1)[:,(yy*64//(y1-y0))*64+xx*64//(x1-x0)];areas=ca.sum(1);pa=len(xx)
        hits=[];ma=[];mb=[];harm=np.zeros(len(ii),bool)
        for geom,fid in known:
            if not poly.intersects(geom):continue
            gg=transform(back,translate(geom,-dx,-dy));a0=af*Affine.translation(int(x0),int(y0))
            patch=rasterize([(gg,1)],out_shape=mask.shape,transform=a0,dtype='uint8');v=patch[yy,xx]>0;s=v.sum()
            if not s:continue
            ga=geom.area/(poly.area/((x1-x0)*(y1-y0)));va=ca[:,v].sum(1);vb=s-va
            ma.append(va/np.maximum(1,ga+areas-va)>=.5);mb.append(vb/np.maximum(1,ga+pa-areas-vb)>=.5)
            hits.append(s/(ga+pa-s)>=.5);harm|=(va>=.1*ga)&(vb>=.1*ga)
        if hits:
            a,b=np.array(ma).T,np.array(mb).T
            delta=(a|b).any(1).astype(int)+(a.any(1)&b.any(1)&((a|b).sum(1)>1)).astype(int)-int(any(hits))
            labels[ii,0]=delta;labels[ii,1]=harm|(delta<0)
    return feat,groups,labels
def main():
    reg=rows();meta=[r for r in json.loads((R22/'CANDIDATES_FROZEN.json').read_text()) if r['role'] in ('MAP_FIT','MAP_INNER')]
    with ProcessPoolExecutor(8,initializer=init_map) as pool:res=list(pool.map(target,[(r,reg[r['id']]) for r in meta]))
    ck=torch.load(R22/'SELECTOR.pt',map_location='cpu',weights_only=True);mu,sd=ck['mu'],ck['sd'];net=Net().cuda();net.load_state_dict(ck['model']);torch.manual_seed(2311)
    sets={}
    for role in ['MAP_FIT','MAP_INNER']:
        rs=[out for r,out in zip(meta,res) if r['role']==role];xs=[];gs=[];ys=[];offset=0
        for x,g,y in rs:
            if not len(g):continue
            xs.append(x);gs.append(g+offset);ys.append(y);offset+=int(g.max())+1
        xx,gg,yy=map(np.concatenate,[xs,gs,ys]);np.savez_compressed(P/(role+'.npz'),x=xx,g=gg,y=yy)
        sets[role]=(((torch.tensor(xx)-mu)/sd).cuda(),gg,yy)
    x,g,y=sets['MAP_FIT'];a,h,b=sets['MAP_INNER']
    pos=torch.tensor(np.where((y[:,0]>0)&(y[:,1]==0))[0],device='cuda');neg=torch.tensor(np.where(y[:,1]==1)[0],device='cuda')
    assert len(pos) and len(neg)
    starts=np.r_[0,np.flatnonzero(np.diff(h))+1];ends=np.r_[starts[1:],len(h)];best=None;curve=[];opt=torch.optim.AdamW(net.parameters(),lr=.00035,weight_decay=.002)
    for ep in range(1,21):
        net.train();losses=[]
        for _ in range(36):
            ix=torch.cat([pos[torch.randint(len(pos),(80,),device='cuda')],neg[torch.randint(len(neg),(240,),device='cuda')]])
            v=net(x[ix]);loss=torch.nn.functional.binary_cross_entropy_with_logits(v,(torch.arange(320,device='cuda')<80).float());opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        net.eval()
        with torch.no_grad():v=net(a).cpu().numpy()
        idx=np.array([s+v[s:t].argmax() for s,t in zip(starts,ends)]);rank=[]
        for c in list(np.arange(-1,4.01,.25))+[100.]:
            t=b[idx[v[idx]>=c]];good=int(((t[:,0]>0)&(t[:,1]==0)).sum());bad=int((t[:,1]==1).sum());rank.append((good-3*bad,-bad,good,c))
        u,nb,good,c=max(rank);curve.append(dict(epoch=ep,loss=np.mean(losses),good=good,bad=-nb,score=u,cut=c))
        if best is None or (u,nb)>best:
            best=(u,nb);torch.save(dict(model=net.state_dict(),mu=mu,sd=sd,cut=float(c),epoch=ep),P/'SELECTOR.pt')
    from assess import table
    with open(P/'CURVES.csv','w') as f:w=csv.DictWriter(f,fieldnames=curve[0]);w.writeheader();w.writerows(curve)
    ck=torch.load(P/'SELECTOR.pt',weights_only=True)
    save(P/'SELECTOR_FROZEN.json',dict(sha256=digest(P/'SELECTOR.pt'),epoch=ck['epoch'],cut=ck['cut'],inner=curve[ck['epoch']-1],positive_cuts=len(pos),harmful_cuts=len(neg),updates=720,lr=.00035,weight_decay=.002,batch=320,seed=2311,training_field=digest(fieldfile),training='single-date parent center approximation to frozen date field, complete assigned manual polygons. FIT 160 windows INNER 36 windows. Missing references ignored.'))
    import predict,assess
    predict.P=P
    # predict reads the shared frozen candidate manifest from its working directory.
    (P/'CANDIDATES_FROZEN.json').write_bytes((R22/'CANDIDATES_FROZEN.json').read_bytes());predict.main();assess.P=P;assess.main()
if __name__=='__main__':main()

