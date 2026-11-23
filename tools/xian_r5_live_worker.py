"""Existing local Torch runtime worker for inference and nontraining readback."""
import sys, os, json, argparse, time, hashlib
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import numpy as np
from PIL import Image

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def put(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def infer(m,o):
    import torch
    rtpath=Path(m['inputs']['roof_runtime']['path']);sys.path.insert(0,str(rtpath.parent))
    import roof_runtime as rt
    pin=m['inputs']['native_rgb'];cp=m['inputs']['B0']
    assert sha(pin['path'])==pin['sha256'] and sha(cp['path'])==cp['sha256']==rt.B0_SHA
    assert m['model_role']=='ORIGINAL_B0_ONLY'
    torch.manual_seed(20261020);torch.cuda.manual_seed_all(20261020)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    # No graph, GT, labels, target imagery, target coordinates or prompts passed.
    metadata={'actual_date':m['packet']['source_observed_date'],'source_image_id':m['packet']['region_id'],'pixel_frame':'SOURCE_PIXEL_EDGE_XY'}
    result=rt.predict_roofs(pin['path'],metadata,cp['path'])
    probability=result.pop('ROOF_PROBABILITY');mask=result.pop('ROOF_MASK')
    assert np.isfinite(probability).all() and np.array_equal(mask,(probability>.5).astype('uint8'))
    np.save(o/'B0_ROOF_PROBABILITY_NATIVE.npy',probability,allow_pickle=False)
    Image.fromarray(mask).save(o/'B0_ROOF_MASK_NATIVE.png')
    put(o/'ROOF_CANDIDATES_NATIVE.json',result)
    put(o/'S4_12_RGB_ONLY_INFERENCE.json',{'execution':'EXECUTED_REAL_FORWARD','GT_FREE_INFERENCE_PASS':True,
        'source_RGB_path':pin['path'],'source_RGB_SHA256':pin['sha256'],'checkpoint':cp['path'],'checkpoint_SHA256':cp['sha256'],
        'config_SHA256':m['inputs']['B0_config']['sha256'],'runtime_SHA256':sha(rtpath),'semantics':'VISIBLE_ROOF_IN_SOURCE_IMAGE_FRAME',
        'actual_date':metadata['actual_date'],'model_role':'ORIGINAL_B0_ONLY','input_metadata_keys':list(metadata),
        'input_shape':list(probability.shape),'candidate_count':len(result['ROOF_INSTANCE_CANDIDATES']),
        'threshold':.5,'threshold_selection':'ORIGINAL_RUNTIME_FIXED_NOT_SEARCHED','probability':'UNCALIBRATED',
        'protocol':m['inference_protocol'],'raw_mask_positive_pixels':int(mask.sum()),'native_resampling':'NONE',
        'GT_or_label_geometry_used':False,'GT_or_target_prompts_used':False,'student_prediction_used':False,
        'training_updates':0,'checkpoint_created':False,'device':torch.cuda.get_device_name(0),
        'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
        'seconds':time.perf_counter()-start,'probability_SHA256':sha(o/'B0_ROOF_PROBABILITY_NATIVE.npy'),'mask_SHA256':sha(o/'B0_ROOF_MASK_NATIVE.png')})
    print('B0_FORWARD_COMPLETE candidates='+str(len(result['ROOF_INSTANCE_CANDIDATES'])),flush=True)

def readback_figure(m,o):
    import torch
    from torch.utils.data import Dataset,DataLoader
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from xian_r5.label_transfer import load_rasters
    info=json.loads((o/'FIGURE_INPUTS.json').read_text(encoding='utf-8'))
    paths=info['label_read_paths'];back=load_rasters(paths);a=back['arrays'];raw=a[0][0]
    class ShadowReadback(Dataset):
        def __len__(self):return 1
        def __getitem__(self,index):
            return {'candidate_fraction':torch.from_numpy(a[0].copy()),'quality':torch.from_numpy(a[1].astype('int32')),
                    'valid':torch.from_numpy(a[3][0].copy()),'unknown':torch.from_numpy(a[3][1].copy()),
                    'weight':torch.from_numpy(a[4][0].copy()),'ignore':torch.from_numpy(a[5][0].copy()),
                    'target':torch.from_numpy(a[6].copy())}
    batch=next(iter(DataLoader(ShadowReadback(),batch_size=1,shuffle=False,num_workers=0)))
    assert torch.all(batch['weight']==0) and torch.all(batch['valid']==0) and torch.all(batch['unknown']==1) and torch.all(batch['ignore']==255)
    assert torch.all(batch['candidate_fraction'][:,1]==-9999)
    put(o/'DATALOADER_READBACK.json',{'execution':'EXECUTED','reader':'xian_r5.label_transfer.load_rasters + torch.utils.data.DataLoader',
        'batch_shapes':{k:list(v.shape) for k,v in batch.items()},'grid_affine':list(back['grid'].affine)[:6],
        'crs':back['grid'].crs,'width':back['grid'].width,'height':back['grid'].height,'file_SHA256':{p:sha(p) for p in paths},
        'qualified_pixels':0,'nonzero_training_weight':0,'all_unknown':True,'registered_fraction_all_NoData':True,
        'training':'NOT_RUN','optimizer_created':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from shapely.geometry import shape
    from affine import Affine
    from pyproj import Transformer
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    rgb=np.array(Image.open(m['inputs']['native_rgb']['path']).convert('RGB'))
    prob=np.load(o/'B0_ROOF_PROBABILITY_NATIVE.npy',allow_pickle=False);mask=np.array(Image.open(o/'B0_ROOF_MASK_NATIVE.png'))
    obs=json.loads((o/'FRESH_OBSERVATIONS_RAW.json').read_text(encoding='utf-8'));records=obs['records']+obs['invalid_native_geometries_unmodified']
    fig,axs=plt.subplots(3,3,figsize=(15,13))
    axs[0,0].imshow(rgb);axs[0,0].set_title('Original GE RGB | 2018-11-01')
    p=axs[0,1].imshow(prob,vmin=0,vmax=1,cmap='viridis');axs[0,1].set_title('B0 roof probability | uncalibrated');fig.colorbar(p,ax=axs[0,1],fraction=.04,pad=.02)
    axs[0,2].imshow(rgb);rgba=np.zeros((*mask.shape,4));rgba[mask>0]=[0,.45,.7,.35];axs[0,2].imshow(rgba);axs[0,2].set_title('B0 mask | original fixed p > 0.5')
    axs[1,0].imshow(rgb)
    for r in records:
        geom=shape(r['native_pixel_geometry']);parts=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
        for g in parts:
            x,y=g.exterior.xy;axs[1,0].plot(x,y,color='#E69F00',linewidth=.5)
    # Reference grid drawn in source pixel coordinates, without image warp.
    packet=m['packet'];aff=Affine(*packet['source_raster_affine']);c0,r0,ww,hh=packet['source_window']
    tf=Transformer.from_crs('EPSG:32649',packet['source_raster_crs'],always_xy=True)
    for v in range(0,65):
        xx,yy=tf.transform(np.repeat(308460+10*v,2),np.array([3794110,3794750]));px,py=(~aff)*(xx,yy);axs[1,0].plot(px-c0,py-r0,color='#56B4E9',alpha=.35,linewidth=.25)
        xx,yy=tf.transform(np.array([308460,309100]),np.repeat(3794750-10*v,2));px,py=(~aff)*(xx,yy);axs[1,0].plot(px-c0,py-r0,color='#56B4E9',alpha=.35,linewidth=.25)
    axs[1,0].set_xlim(0,ww);axs[1,0].set_ylim(hh,0);axs[1,0].set_title('Candidate outlines + fixed 10 m grid')
    axs[1,1].imshow(rgb)
    axs[1,1].text(.5,.55,'NO_ACCEPTED_ALIGNMENT',transform=axs[1,1].transAxes,ha='center',va='center',fontsize=13,color='white',bbox={'facecolor':'black','alpha':.8,'pad':8})
    axs[1,1].text(.5,.39,'All displacement vectors = UNKNOWN\nNo arrows; no corrected geometry',transform=axs[1,1].transAxes,ha='center',color='white',bbox={'facecolor':'black','alpha':.8})
    axs[1,1].set_title('A: no qualified TARGET_B / CHECK')
    f=axs[1,2].imshow(np.ma.masked_equal(raw,-9999),vmin=0,vmax=1,cmap='viridis');axs[1,2].set_facecolor('#d9d9d9');axs[1,2].set_title('Native roof fraction | grey = NoData');fig.colorbar(f,ax=axs[1,2],fraction=.04,pad=.02)
    axs[2,0].imshow(a[2][0],vmin=0,vmax=1,cmap='Blues');axs[2,0].set_title('Source spatial support | NOT valid GT')
    axs[2,1].imshow(a[3][1],vmin=0,vmax=1,cmap=ListedColormap(['#009E73','#E69F00']));axs[2,1].set_title('Supervision UNKNOWN = 1 everywhere')
    axs[2,1].text(.5,.5,'valid = 0\nignore = 255\ntraining weight = 0',transform=axs[2,1].transAxes,ha='center',va='center',fontsize=13,bbox={'facecolor':'white','alpha':.9,'pad':10})
    axs[2,2].set_axis_off();q=json.loads((o/'S4_12_GT10M_SUPERVISION_ELIGIBILITY.json').read_text(encoding='utf-8'))
    axs[2,2].text(0,1,'Actual handoff / unresolved evidence',va='top',weight='bold',fontsize=11)
    axs[2,2].text(0,.9,f'RGB hash: {m["inputs"]["native_rgb"]["sha256"][:16]}...\nB0 hash: {m["inputs"]["B0"]["sha256"][:16]}...\n\nSource-native candidates: {len(records)}\nValid polygon candidates: {info["valid_candidates"]}\nInvalid raw polygons retained: {info["invalid_candidates"]}\nHistorical 53 source links: preserved, NOT inherited\nPhysical IDs / applied shifts: 0 / 0\nPresence: U | single date, no change interval\nSupported target pixels: {q["source_supported_pixels"]}\nPositive native fraction pixels: {q["positive_native_fraction_pixels"]}\nQualified training pixels: 0\n\nShadow only; no position-accuracy claim.',va='top',linespacing=1.6,fontsize=9)
    for ax in axs.flat:
        if ax.axison:ax.set_xlabel('native column' if ax in axs[:2,:].ravel() and ax!=axs[1,2] else '10 m target column');ax.set_ylabel('native row' if ax in axs[:2,:].ravel() and ax!=axs[1,2] else '10 m target row')
    fig.suptitle('S4.12 live B0 -> A / I -> Presence -> GT10m SHADOW | NO_ACCEPTED_ALIGNMENT',fontsize=14)
    fig.text(.5,.014,'Existing roi_r0_c0; all candidates shown, not selected by quality. Grey outside inferred support is unknown, never background. Absolute position accuracy UNKNOWN.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.96]);fig.savefig(o/'S4_12_END_TO_END_FIGURE.png',dpi=300);fig.savefig(o/'S4_12_END_TO_END_FIGURE.pdf');plt.close(fig)
    put(o/'FIGURE_RECEIPT.json',{'data_only':True,'no_generated_geometry':True,'accepted_vectors':0,'NO_ACCEPTED_ALIGNMENT':True,'inputs':{'manifest':sha(o/'S4_12_END_TO_END_RUN_MANIFEST.json'),'raw_candidates':sha(o/'FRESH_OBSERVATIONS_RAW.json'),'fraction':sha(o/'S4_12_GT10M_FRACTION_SHADOW.tif')},'outputs':{n:sha(o/n) for n in ['S4_12_END_TO_END_FIGURE.png','S4_12_END_TO_END_FIGURE.pdf']}})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['infer','readback-figure']);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    m=json.loads(a.manifest.read_text(encoding='utf-8'));o=a.output
    assert o.exists() and m['execution_mode']=='LIVE_B0_RGB_SHADOW_V1'
    if a.stage=='infer':infer(m,o)
    else:readback_figure(m,o)
if __name__=='__main__':main()
