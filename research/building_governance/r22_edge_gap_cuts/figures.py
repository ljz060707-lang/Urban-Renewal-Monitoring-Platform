from assess import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.warp import reproject,Resampling
from rasterio.transform import from_origin
from scipy.ndimage import maximum_filter,minimum_filter
reg=rows()
cases=list(csv.DictReader((P/'REFERENCE_COMPARISON.csv').open()))
take=[next(a for a in cases if a['gained']=='1'),next(a for a in cases if a['lost']=='1'),next(a for a in cases if a['broken_previously_matched']=='1' and a['lost']=='0')]
fig,axes=plt.subplots(3,3,figsize=(11,10))
names=['Recovered match','Lost match','Matched but fragmented']
for index,r in enumerate(take):
    row=reg[r['window']];_,_,refs=load_window_geometry(row);ref=refs[int(r['fid'])]
    x0,y0,x1,y1=ref.bounds;sz=max(35,x1-x0,y1-y0)*1.45;cx=(x0+x1)/2;cy=(y0+y1)/2;n=280;tf=from_origin(cx-sz/2,cy+sz/2,sz/n,sz/n)
    def warp(ds,band,dtype,res):
        out=np.zeros((n,n),dtype);reproject(rasterio.band(ds,band),out,dst_transform=tf,dst_crs='EPSG:32649',resampling=res);return out
    with rasterio.open(row['image_path']) as ds:rgb=np.stack([warp(ds,j,'uint8',Resampling.bilinear) for j in [1,2,3]],2)
    gt=rasterize([(ref,1)],out_shape=(n,n),transform=tf,dtype='uint8')
    with rasterio.open(P/'position'/(r['window']+'.tif')) as ds:ids=[warp(ds,j,'uint32',Resampling.nearest) for j in (3,4)]
    for j,ax in enumerate(axes[index]):
        ax.imshow(rgb)
        if j:
            lab=ids[j-1];edge=(maximum_filter(lab,3)!=minimum_filter(lab,3))
            color=np.zeros((n,n,4));color[:,:,0]=1;color[:,:,1]=.24;color[:,:,3]=edge*.97;ax.imshow(color)
        ax.contour(gt,levels=[.5],colors=['#40ffde'],linewidths=1,linestyles='dashed')
        ax.set_xticks([]);ax.set_yticks([])
        if index==0:ax.set_title(['Imagery + manual reference','Previous split (R21)','Edge / gap selector'][j])
        if j==0:ax.set_ylabel(names[index]+'\nFID '+r['fid']+'\n'+r['actual_date'])
fig.suptitle('Xi\'an development windows | cyan: 2019 reference; orange: positioned instance boundaries')
fig.tight_layout();fig.savefig(P/'CASES.png',dpi=145);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,3.1))
for d,label in [(P,'Edge / gap'),(P/'contour','Full contour')]:
    records=list(csv.DictReader((d/'CURVES.csv').open()));ep=[int(v['epoch']) for v in records];axs[0].plot(ep,[float(v['loss']) for v in records],label=label);axs[1].plot(ep,[float(v.get('utility',v.get('inner_utility'))) for v in records],label=label)
for ax in axs:ax.set_xlabel('Epoch');ax.legend()
axs[0].set_title('Training loss');axs[1].set_title('Selection utility (arm-specific)')
fig.tight_layout();fig.savefig(P/'CURVES.png',dpi=140);plt.close(fig)

