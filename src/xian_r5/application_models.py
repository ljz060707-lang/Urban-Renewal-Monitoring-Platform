"""Inference-only serial workers. Never construct an optimizer or read labels."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['GDAL_PAM_ENABLED']='NO'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import time
from .application_contract import read,save,sha,check_model_route

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def run_worker(request):
    import numpy as np
    import torch
    from PIL import Image
    req=read(request); routes=req['routing'];check_model_route(routes)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    action=req['action'];started=time.monotonic();calls=0
    if action=='roof':
        rt=load('application_original_roof',routes['roof']['runtime'])
        original_build=rt.build_model
        def frozen_build(checkpoint):
            import types
            model=original_build(checkpoint);model.requires_grad_(False)
            def direct_grid(pe,size):
                h,w=size;d=pe.positional_encoding_gaussian_matrix.device
                yy=(torch.arange(h,device=d,dtype=torch.float32)+.5)/h
                xx=(torch.arange(w,device=d,dtype=torch.float32)+.5)/w
                return pe._pe_encoding(torch.stack((xx[None,:].expand(h,w),yy[:,None].expand(h,w)),-1)).permute(2,0,1)
            model.decode_head.pe_emb.forward=types.MethodType(direct_grid,model.decode_head.pe_emb)
            return model
        rt.build_model=frozen_build
        original_forward=rt.forward
        def counted_forward(*args,**kwargs):
            nonlocal calls
            calls+=1
            return original_forward(*args,**kwargs)
        rt.forward=counted_forward
        for job in req['jobs']:
            result=rt.predict_roofs(job['image'],{'actual_date':job.get('actual_date'),
                'source_image_id':job['source_id'],'pixel_frame':'SOURCE_PIXEL_EDGE_XY'},routes['roof']['checkpoint'])
            out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
            np.save(out/'probability.npy',result.pop('ROOF_PROBABILITY'))
            np.save(out/'mask.npy',result.pop('ROOF_MASK'));save(out/'instances.json',result)
    elif action=='facade':
        sys.path.insert(0,str(Path(routes['facade']['factory']).parent))
        tr=load('application_s_factory',routes['facade']['factory']);root=Path(routes['facade']['payload'])
        model=tr.make_s_model(root);cp=torch.load(routes['facade']['checkpoint'],map_location='cpu',weights_only=False)
        if cp['successful_steps']!=120: raise ValueError('E3 step mismatch')
        model.load_state_dict(cp['S'],strict=True);model.set_stage('frozen').cuda().eval()
        # Existing validated direct FP32 coordinate grid; cumsum is not used.
        import types
        def direct_grid(pe,size):
            h,w=size;d=pe.positional_encoding_gaussian_matrix.device
            yy=(torch.arange(h,device=d,dtype=torch.float32)+.5)/h
            xx=(torch.arange(w,device=d,dtype=torch.float32)+.5)/w
            return pe._pe_encoding(torch.stack((xx[None,:].expand(h,w),yy[:,None].expand(h,w)),-1)).permute(2,0,1)
        model.b0.decode_head.pe_emb.forward=types.MethodType(direct_grid,model.b0.decode_head.pe_emb)
        api=tr._adapter(root)
        for job in req['jobs']:
            rgb=np.asarray(Image.open(job['image']).convert('RGB'));h,w=rgb.shape[:2];p=np.zeros((h,w),np.float32)
            with torch.no_grad():
                for y in range(0,h,512):
                    for x in range(0,w,512):
                        hh,ww=min(512,h-y),min(512,w-x);tile=np.zeros((512,512,3),np.uint8);tile[:hh,:ww]=rgb[y:y+hh,x:x+ww]
                        q=model(api.normalize_rgb(torch.from_numpy(tile.transpose(2,0,1).copy())[None].cuda()))
                        p[y:y+hh,x:x+ww]=q['facade_logits'][0,0,:hh,:ww].sigmoid().cpu().numpy();calls+=1
            np.save(Path(job['output'])/'facade_probability.npy',p)
    elif action=='alignment':
        from affine import Affine
        import rasterio
        from rasterio.features import rasterize
        from rasterio.warp import reproject,Resampling
        from shapely.geometry import shape
        import math
        r=routes['alignment'];sys.path.insert(0,str(Path(r['runtime']).parent))
        rt=load('runtime',r['runtime']);rt.setup()
        # Reuse the existing exact prefix algorithm on CPU FP32; its CUDA
        # cumsum has no deterministic implementation in this pinned runtime.
        import support
        original_mass=support.supported_cell_mass
        def deterministic_mass(mask,sv,rv,offsets):
            if bool((rv==1).all()):
                return original_mass(mask.float().cpu(),sv.float().cpu(),rv.float().cpu(),offsets.cpu()).to(mask.device)
            return original_mass(mask,sv,rv,offsets)
        support.supported_cell_mass=deterministic_mass
        from matching_v2 import CorrespondencePredictor
        cp=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        if cp['successful_updates_each']!=3634:raise ValueError('V2 step mismatch')
        model=CorrespondencePredictor(rt.original.AlignmentConfig(**cp['config']))
        model.load_state_dict(cp['models']['M_CORR'],strict=True);model.cuda().eval();model.requires_grad_(False)
        for job in req['jobs']:
            rgb=np.asarray(Image.open(job['image']).convert('RGB'));h,w=rgb.shape[:2]
            sv=np.load(job['valid']);aff=Affine(*job['affine']);rows=read(job['instances'])['ROOF_INSTANCE_CANDIDATES'];output=[]
            with rasterio.open(job['lr_image']) as ref:
                for row in rows:
                    g=shape(row['geometry_pixel_edge_xy']);cx,cy=g.centroid.coords[0]
                    ox=int(np.clip(math.floor(cx-256),0,max(0,w-512)));oy=int(np.clip(math.floor(cy-256),0,max(0,h-512)))
                    complete=bool(not row['source_edge_partial_risk'] and w>=512 and h>=512 and g.bounds[0]>=ox and g.bounds[1]>=oy and g.bounds[2]<=ox+512 and g.bounds[3]<=oy+512)
                    rec=dict(candidate_id=row['candidate_id'],complete_instance_context=complete,shift_model_xy=None,
                        model_to_native_scale=2.,origin_xy=[ox,oy],status='UNKNOWN_PARTIAL_OR_CONTEXT',formal_position='UNRESOLVED')
                    if complete:
                        hm=rasterize([(g,1)],out_shape=(512,512),transform=Affine.translation(ox,oy),dtype='uint8')
                        hr=np.asarray(Image.fromarray(rgb[oy:oy+512,ox:ox+512]).resize((256,256),Image.Resampling.BILINEAR)).transpose(2,0,1).astype('float32')[None]/255
                        mask=np.asarray(Image.fromarray(hm).resize((256,256),Image.Resampling.NEAREST)).astype('float32')[None,None]
                        valid=np.asarray(Image.fromarray(sv[oy:oy+512,ox:ox+512]).resize((256,256),Image.Resampling.NEAREST)).astype('float32')[None,None]
                        la=aff*Affine.translation(ox,oy)*Affine.scale(8,8)
                        lr=np.zeros((3,64,64),np.float32);lv=np.zeros((64,64),np.uint8)
                        for b in range(3):reproject(rasterio.band(ref,b+1),lr[b],src_transform=ref.transform,src_crs=ref.crs,dst_transform=la,dst_crs=job['crs'],resampling=Resampling.bilinear)
                        # Read only the source window of the mask via a WarpedVRT; no city-wide mask allocation.
                        from rasterio.vrt import WarpedVRT
                        with WarpedVRT(ref,crs=job['crs'],transform=la,width=64,height=64,resampling=Resampling.nearest) as vrt:lv=(vrt.dataset_mask()>0).astype('float32')[None,None]
                        if job.get('lr_valid_mask'):
                            with rasterio.open(job['lr_valid_mask']) as vm:
                                with WarpedVRT(vm,crs=job['crs'],transform=la,width=64,height=64,resampling=Resampling.nearest) as vrt:
                                    lv*=((vrt.read(1)>0)&(vrt.dataset_mask()>0)).astype('float32')[None,None]
                        tensors={k:torch.from_numpy(v.copy()).cuda() for k,v in dict(hr=hr,lr=(lr/255)[None]*lv,masks=mask,source_valid=valid,lr_valid=lv).items()}
                        if mask.sum()>0 and valid.sum()>0 and lv.sum()>0:
                            with torch.no_grad():p=model(tensors['hr'],tensors['lr'],tensors['masks'],source_valid=tensors['source_valid'],lr_valid=tensors['lr_valid'])
                            calls+=1;d=p['resolved_total_shift_px']
                            if d is not None and d.numel():
                                rec.update(shift_model_xy=d[0].detach().cpu().tolist(),status=p['application_status'],unmatched_score_uncalibrated=p['unmatched_probability'].detach().cpu().tolist())
                        else:rec['status']='UNKNOWN_MISSING_SUPPORT'
                    output.append(rec)
            save(Path(job['output'])/'alignment.json',dict(records=output,checkpoint_sha256=r['checkpoint_sha256'],normalization=r['normalization']))
    else:raise ValueError('Unknown inference action')
    save(req['receipt'],dict(action=action,actual_forwards=calls,jobs=len(req['jobs']),seconds=time.monotonic()-started,
        model_parameters_frozen=True,optimizer_updates=0,GT_reads=0,cloud_operations=0))

if __name__=='__main__':run_worker(sys.argv[1])
