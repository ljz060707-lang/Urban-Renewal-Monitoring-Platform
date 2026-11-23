"""Bounded A synthetic/real weak training using the existing torch loop idiom.

Labels enter loss only. Scene weak supervision is not a certified building
correspondence. Public pixel-GSD unknown stays pixels. No production admission.
"""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F

from xian_r5.cloud_joint.alignment import (AlignmentConfig, AlignmentPredictor,
    alignment_checkpoint, load_alignment_checkpoint, known_transform_loss,
    translate_and_aggregate, reference_map_loss, real_weak_alignment_loss)


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for x in iter(lambda:s.read(4194304),b''):h.update(x)
    return h.hexdigest()
def atomic_json(path,data):
    temp=Path(str(path)+'.tmp')
    with temp.open('w') as f:json.dump(data,f,indent=2,allow_nan=False)
    temp.replace(path)
def scalar(x):return float(x.detach().float().cpu()) if isinstance(x,torch.Tensor) else x
def append(path,data):
    with Path(path).open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')
def deadline(protocol,seconds):
    return min(time.time()+seconds,datetime.fromisoformat(protocol['hard_training_deadline']).timestamp())


def component_masks(mask,device='cuda'):
    """Semantic components are candidate proxies, never physical building IDs."""
    ids,n=ndimage.label(mask>0,structure=np.ones((3,3),int))
    if not n:return None
    return torch.from_numpy(np.stack([ids==i for i in range(1,n+1)]).astype('float32'))[None].to(device)


def load_scene(root,row,device='cuda'):
    # The input file contains RGB/candidate/support/affine only, no GT.
    with np.load(root/row['input'],allow_pickle=False) as d:
        hr=torch.from_numpy(d['hr_rgb'].copy()).float()[None].to(device)/255
        lr=torch.from_numpy(d['lr_rgb'].copy()).float()[None].to(device)/255
        masks=component_masks(d['roof_mask'],device)
        valid=torch.from_numpy(d['source_support'].copy())[None,None].bool().to(device)
    return hr,lr,masks,valid


def source_public_case(root,row,index,seed,device='cuda'):
    """TRAIN-only synthetic sample; content and masks are moved together."""
    rng=random.Random(seed+index)
    with Image.open(root/row['image']) as im:
        w,h=im.size;x=rng.randrange(max(1,(w-512)//512+1))*512;y=rng.randrange(max(1,(h-512)//512+1))*512
        image=np.asarray(im.convert('RGB').crop((x,y,x+512,y+512)).resize((256,256),Image.Resampling.BILINEAR)).copy()
    with Image.open(root/row['label']) as im:
        label=np.asarray(im.crop((x,y,x+512,y+512)).resize((256,256),Image.Resampling.NEAREST)).copy()
    roof=(label==1)&(label!=255);masks=component_masks(roof,device)
    if masks is None:return None
    hr=torch.from_numpy(image.transpose(2,0,1)).float()[None].to(device)/255
    # Target reference is degraded from the preserved target frame. This is
    # NOT an off-nadir physical camera simulation or a new real pairing.
    lr=F.interpolate(F.avg_pool2d(hr,3,1,1),size=(64,64),mode='area')
    q=torch.tensor([rng.uniform(-6,6),rng.uniform(-6,6)],device=device)[None,None]
    residual=torch.tensor([[rng.uniform(-2,2),rng.uniform(-2,2)] for _ in range(masks.shape[1])],device=device)[None]
    applied=q+residual
    original_rgb_masks=hr[:,None]*masks[:,:,None]
    moved=translate_and_aggregate(masks,applied)['moved_masks']
    # Preserve original texture under each translated roof; overlap/occlusion
    # artifacts remain explicitly synthetic and never become real GT.
    background_shift=q.expand(1,3,2).contiguous()
    bg=translate_and_aggregate(hr,background_shift)['moved_masks']
    pasted=[]
    for c in range(3):
        pieces=translate_and_aggregate(original_rgb_masks[:,:,c],applied)['moved_masks']
        pasted.append(pieces.amax(1))
    pasted=torch.stack(pasted,1)
    union=moved.amax(1,keepdim=True)
    source=bg*(1-union)+pasted
    support=torch.ones_like(moved[:,:1],dtype=torch.bool)
    complete=(moved.sum((-2,-1))>=masks.sum((-2,-1))-.01)&(moved.sum((-2,-1))>0)
    if not complete.any():return None
    return source.clamp(0,1),lr,moved,support,-applied,complete


def save_a(path,model,optimizer,epoch,step,cursor,protocol,extra=None):
    payload=dict(model=alignment_checkpoint(model),optimizer=optimizer.state_dict(),
                 epoch=epoch,successful_steps=step,sampler_cursor=cursor,
                 torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                 numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                 scheduler=dict(kind='FIXED_LR_STAGE_PROTOCOL',lr=[g['lr'] for g in optimizer.param_groups]),
                 scaler=None,AMP=False,protocol=protocol,extra=extra or {})
    tmp=Path(str(path)+'.tmp');torch.save(payload,tmp);tmp.replace(path)


def run_a(payload_root,output,protocol):
    root=Path(payload_root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed']);random.seed(protocol['seed'])
    torch.set_num_threads(4)
    config=AlignmentConfig(width=16,context_size=32,max_source_shift_px=12,max_residual_shift_px=4)
    model=AlignmentPredictor(config).cuda()
    data=read(root/'config/segmentation_data.json')['records']
    public=[r for r in data if r['split']=='TRAIN']
    public.sort(key=lambda r:hashlib.sha256((str(protocol['seed'])+r['id']).encode()).hexdigest())
    adata=read(root/'config/alignment_data.json')['records']
    train=[r for r in adata if r['split']=='TRAIN'];tune=[r for r in adata if r['split']=='TUNE']
    if not public or not train or not tune:raise ValueError('Missing frozen A data roles')
    atomic_json(out/'A_PROTOCOL.json',dict(**protocol,model_config=config.__dict__,
        coordinate_convention='TARGET_CANVAS_PIXEL_DX_RIGHT_DY_DOWN',
        synthetic='CONTENT_TRANSLATION_WITH_PASTE_ARTIFACTS_NOT_CAMERA_SIMULATION',
        target_semantics='REFERENCE_BUILDING_PLANIMETRY',weak_GT_external_background='UNKNOWN_NOT_NEGATIVE_GT',
        physical_correspondence_GT=False,track_constraint='NOT_AVAILABLE_NO_CERTIFIED_CROSS_DATE_LINK'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol.get('A_lr',.001),weight_decay=.0001)
    syn_end=deadline(protocol,protocol.get('A_synthetic_max_seconds',1200))
    steps=0;skipped=0;epochs_done=0;last_save=time.monotonic();started=time.monotonic()
    # Real image TRAIN batch numerical test, not a successful update.
    example=None
    for i,r in enumerate(public[:32]):
        example=source_public_case(root,r,i,protocol['seed'])
        if example:break
    if not example:raise ValueError('No real TRAIN roof sample for A batch')
    hr,lr,masks,valid,target,matched=example
    pred=model(hr,lr,masks,source_valid=valid)
    loss=known_transform_loss(pred,target,matched,supervision_valid=matched)['loss'];loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('A batch gradients nonfinite')
    restored=load_alignment_checkpoint(alignment_checkpoint(model),expected_config=config).cuda()
    with torch.no_grad():
        check=restored(hr,lr,masks,source_valid=valid)['total_shift_px']
    if not torch.equal(check,pred['total_shift_px']):raise ValueError('A strict reload mismatch')
    model.zero_grad(set_to_none=True)
    atomic_json(out/'A_REAL_BATCH_TEST.json',dict(status='PASS',optimizer_updates=0,source='REAL_OFFICIAL_TRAIN_RGB_WITH_SYNTHETIC_KNOWN_TRANSFORM',
         loss=scalar(loss),strict_reload=True,allocated_bytes=torch.cuda.max_memory_allocated(),
         reserved_bytes=torch.cuda.max_memory_reserved(),instances=int(masks.shape[1])))
    for epoch in range(protocol.get('A_synthetic_epochs',30)):
        if time.time()>=syn_end:break
        completed=True
        for cursor in range(protocol.get('A_synthetic_cases_per_epoch',128)):
            if time.time()>=syn_end:completed=False;break
            idx=epoch*protocol.get('A_synthetic_cases_per_epoch',128)+cursor
            row=public[idx%len(public)];sample=source_public_case(root,row,idx,protocol['seed'])
            if sample is None:skipped+=1;continue
            hr,lr,masks,valid,target,matched=sample;t=time.monotonic()
            optimizer.zero_grad(set_to_none=True);pred=model(hr,lr,masks,source_valid=valid)
            terms=known_transform_loss(pred,target,matched,supervision_valid=matched)
            terms['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step();steps+=1
            append(out/'A_TRAINING_STEPS.jsonl',dict(stage='SYNTHETIC',epoch=epoch,cursor=cursor,successful_update=steps,
                AMP_skipped=0,loss=scalar(terms['loss']),known_transform=scalar(terms['known_transform']),
                gradient_norm=scalar(norm),seconds=time.monotonic()-t,source_id=row['id'],instances=masks.shape[1],
                unit='SYNTHETIC_PIXEL_NOT_REAL_METRE_EPE'))
            if time.monotonic()-last_save>=1800:
                save_a(out/'latest.pth',model,optimizer,epoch,steps,cursor,protocol);last_save=time.monotonic()
        if completed:epochs_done+=1
        save_a(out/'latest.pth',model,optimizer,epoch,steps,cursor,protocol)
    save_a(out/'synthetic_final.pth',model,optimizer,epochs_done,steps,0,protocol)
    # Real-domain scene loss: use only positive reference support until label
    # completeness/time continuity is established; unmatched cannot erase loss.
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol.get('A_real_lr',.0001),weight_decay=.0001)
    weak_end=deadline(protocol,protocol.get('A_real_max_seconds',1200));weak_done=0;best=float('inf');history=[]
    for epoch in range(protocol.get('A_real_epochs',60)):
        if time.time()>=weak_end:break
        complete=True
        for cursor,row in enumerate(sorted(train,key=lambda r:hashlib.sha256((str(epoch)+r['id']).encode()).hexdigest())):
            if time.time()>=weak_end:complete=False;break
            hr,lr,masks,sv=load_scene(root,row)
            if masks is None:skipped+=1;continue
            # This label read occurs only after constructing the prediction input.
            with np.load(root/row['training_supervision'],allow_pickle=False) as d:
                gt=torch.from_numpy(d['reference_roof'].copy())[None,None].float().cuda()
            optimizer.zero_grad(set_to_none=True);pred=model(hr,lr,masks,source_valid=sv)
            raster=translate_and_aggregate(masks,pred['total_shift_px'],factor=4,target_valid=sv)
            valid=(gt>0)&sv
            if not valid.any():skipped+=1;continue
            # Original-instance denominators and the weak unmatched-mass
            # floor prevent dropping hard objects; missing tracks stay None.
            terms=real_weak_alignment_loss(pred,raster,gt,valid,stable_edges=None)
            loss=terms['loss']
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step();steps+=1
            append(out/'A_TRAINING_STEPS.jsonl',dict(stage='REAL_SCENE_WEAK',epoch=epoch,cursor=cursor,successful_update=steps,
                AMP_skipped=0,loss=scalar(loss),gradient_norm=scalar(norm),source_id=row['id'],
                reference_positive_pixels=int(valid.sum()),outside_GT_support='UNKNOWN_NOT_NEGATIVE_GT',
                unmatched_probability_mean=scalar(pred['unmatched_probability'].mean()),
                uncertainty_m=None,track_loss=None,certified_building_match=False,
                weak_loss_weights=terms['weights']))
        if complete:weak_done+=1
        if epoch%2==1 or epoch==0:
            metrics=evaluate_a(model,root,tune)
            history.append(dict(epoch=epoch,**metrics))
            atomic_json(out/'A_TUNE_HISTORY.json',history)
            if metrics['selection_positive_support_fraction_error']<best:
                best=metrics['selection_positive_support_fraction_error'];save_a(out/'best.pth',model,optimizer,epoch,steps,cursor,protocol,metrics)
        save_a(out/'latest.pth',model,optimizer,epoch,steps,cursor,protocol)
    if not (out/'best.pth').exists():
        save_a(out/'best.pth',model,optimizer,weak_done,steps,0,protocol,{'selection':'NO_REAL_TUNE_EVALUATION_NOT_QUALIFIED'})
    result=dict(TRAINING_EXECUTED=steps>0,synthetic_coverage_epochs=epochs_done,real_weak_epochs=weak_done,
        successful_updates=steps,skipped_empty_or_unsupported=skipped,elapsed_seconds=time.monotonic()-started,
        best_checkpoint=str(out/'best.pth'),best_checkpoint_sha256=sha(out/'best.pth'),
        latest_checkpoint_sha256=sha(out/'latest.pth'),tune_best_positive_support_error=None if best==float('inf') else best,
        POSITION_UNCERTAINTY='UNKNOWN',MODEL_TARGET_MAP_VALIDATED=False,physical_GT_pairs_used=0,
        independent_building_EPE_m=None,production_admission=False,
        inference_model_accepts_GT=False,scope='WEAK_SCENE_MAP_REFERENCE_NOT_CERTIFIED_BUILDING_ALIGNMENT')
    atomic_json(out/'A_TRAINING_RESULT.json',result);return result


@torch.no_grad()
def evaluate_a(model,root,rows):
    errors=[];before=[];positive_recall=[];full_diagnostics=[];denominators=[]
    for row in rows:
        hr,lr,masks,sv=load_scene(root,row)
        if masks is None:continue
        pred=model(hr,lr,masks,source_valid=sv)
        raster=translate_and_aggregate(masks,pred['total_shift_px'],factor=4,target_valid=sv)
        # GT enters only after the prediction and transported raster exist.
        with np.load(root/row['training_supervision'],allow_pickle=False) as d:
            gt=torch.from_numpy(d['reference_roof'].copy())[None,None].float().cuda()
        target=F.avg_pool2d(gt,4,4);raw_union=masks.amax(1,keepdim=True)
        coverage=(gt>0)&sv
        fixed_denominator=F.avg_pool2d(coverage.float(),4,4).sum()
        if not bool(fixed_denominator>0):continue
        # Unknown exterior is never scored as reliable negative GT for model
        # selection. Target fractions still use the full pixel area, while
        # this error's separate fixed known-support denominator is explicit.
        errors.append(scalar(F.avg_pool2d((raster['union']-gt)*coverage,4,4).abs().sum()/fixed_denominator))
        before.append(scalar(F.avg_pool2d((raw_union-gt)*coverage,4,4).abs().sum()/fixed_denominator))
        full_diagnostics.append(scalar((raster['fractions']-target).abs().mean()))
        denominators.append(scalar(fixed_denominator))
        positive_recall.append(scalar((raster['union']*(gt>0)).sum()/gt.sum().clamp_min(1)))
    if not errors:raise ValueError('No measurable A TUNE reference denominator')
    return dict(selection_positive_support_fraction_error=float(np.mean(errors)),raw_positive_support_error=float(np.mean(before)),
         full_canvas_reference_polygon_set_MAE_diagnostic=float(np.mean(full_diagnostics)),
         positive_reference_overlap=float(np.mean(positive_recall)),regions=len(errors),
         selection_known_support_denominator_per_region=denominators,original_grid_pixel_denominator_per_region=4096,
         unit='POSITIVE_SUPPORT_NORMALIZED_ERROR_NOT_FULL_ROI_FRACTION_MAE_GOAL',
         source_date_reference_date_compatibility='UNRESOLVED',reference_map_completeness='NOT_INDEPENDENTLY_CERTIFIED',
         not_ground_footprint_or_absolute_position_validation=True)
