"""Small matched S/A joint continuation; native S supervision remains anchored.

No map GT is a forward input. Observed candidate masks are differentiable
proxies in a pre-mapped feature canvas, not certified physical identities.
"""
from datetime import datetime
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from xian_cloud_s_train import make_s_model
from xian_cloud_a_train import (read,sha,atomic_json,append,scalar,deadline,load_scene,
                                source_public_case,evaluate_a)
from xian_r5.cloud_joint.segmentation import normalize_rgb,bandon_targets,native_segmentation_loss
from xian_r5.cloud_joint.alignment import (alignment_checkpoint,load_alignment_checkpoint,
                                         translate_and_aggregate,reference_map_loss,known_transform_loss,
                                         real_weak_alignment_loss)


def native_anchor(root,row,index):
    rng=random.Random(20261020+index)
    with Image.open(root/row['image']) as im:
        w,h=im.size;x=rng.randrange(max(1,(w-512)//512+1))*512;y=rng.randrange(max(1,(h-512)//512+1))*512
        rgb=np.asarray(im.convert('RGB').crop((x,y,x+512,y+512))).copy()
    with Image.open(root/row['label']) as im:label=np.asarray(im.crop((x,y,x+512,y+512))).copy()
    x=torch.from_numpy(rgb.transpose(2,0,1))[None].cuda()
    y=torch.from_numpy(label)[None].cuda()
    if row['dataset']=='BANDON':roof,facade=bandon_targets(y)
    else:roof,facade=y,None
    return x,roof,facade


def joint_checkpoint(path,s,a,optimizer,scaler,epoch,steps,cursor,protocol,parents):
    tmp=Path(str(path)+'.tmp')
    torch.save(dict(schema='XIAN_CLOUD_JOINT_V1',S=s.state_dict(),A=alignment_checkpoint(a),
        optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),epoch=epoch,successful_steps=steps,cursor=cursor,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),scheduler=dict(kind='FIXED_JOINT_LR',lr=[g['lr'] for g in optimizer.param_groups]),
        protocol=protocol,parents=parents,GT_forward=False),tmp)
    tmp.replace(path)


def run_joint(root,output,protocol,s_checkpoint,a_checkpoint):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
    s=make_s_model(root,s_checkpoint).cuda().set_stage('heads').train()
    ap=torch.load(a_checkpoint,map_location='cpu',weights_only=False)
    a=load_alignment_checkpoint(ap['model']).cuda().train()
    parents=dict(S_checkpoint_sha256=sha(s_checkpoint),A_checkpoint_sha256=sha(a_checkpoint))
    public=[r for r in read(root/'config/segmentation_data.json')['records'] if r['split']=='TRAIN']
    regions=read(root/'config/alignment_data.json')['records'];train=[r for r in regions if r['split']=='TRAIN']
    tune=[r for r in regions if r['split']=='TUNE']
    optimizer=torch.optim.AdamW([*s.parameter_groups(head_lr=protocol['S_head_lr']),
        dict(params=list(a.parameters()),lr=protocol['A_lr'])],weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=256)
    stop=deadline(protocol,protocol.get('max_seconds',1200));steps=0;skips=0;epochs=0;best=float('inf');history=[]
    params=[p for group in optimizer.param_groups for p in group['params']]
    frozen={n:p.detach().cpu().clone() for n,p in s.named_parameters() if not p.requires_grad}
    atomic_json(out/'JOINT_PROTOCOL.json',dict(protocol=protocol,parents=parents,
        scene_loss_scope='REFERENCE_POSITIVE_SUPPORT_ONLY_UNKNOWN_EXTERIOR',
        native_roof_anchor='REAL_OFFICIAL_TRAIN_RGB_AND_NATIVE_LABEL',
        unmatched_not_used_to_remove_loss=True,track_loss=None,GT_in_forward=False,
        real_scene_hard_candidate_support='OBSERVED_CANDIDATE_PROXY_NOT_MANUAL_GT'))
    started=time.monotonic()
    for epoch in range(protocol.get('max_epochs',20)):
        if time.time()>=stop:break
        complete=True
        for cursor,row in enumerate(train):
            if time.time()>=stop:complete=False;break
            idx=epoch*len(train)+cursor
            hr,lr,masks,valid=load_scene(root,row)
            if masks is None:skips+=1;continue
            px,roof,facade=native_anchor(root,public[idx%len(public)],idx)
            with np.load(root/row['training_supervision'],allow_pickle=False) as d:
                target=torch.from_numpy(d['reference_roof'].copy())[None,None].float().cuda()
            supervision=(target>0)&valid
            if not supervision.any():skips+=1;continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                native=s(normalize_rgb(px))
                anchor=native_segmentation_loss(native,roof,facade_target=facade,boundary_trusted=True)
                source=s(normalize_rgb((hr*255).round().byte()))['roof_logits'].float().sigmoid()
            if anchor['skip_update']:skips+=1;continue
            # S probabilities can change; the native anchor and mass counters
            # must expose shrinkage. A itself remains a rigid translation.
            soft_masks=masks*source
            prediction=a(hr,lr,soft_masks.detach(),source_valid=valid)
            raster=translate_and_aggregate(soft_masks,prediction['total_shift_px'],factor=4,target_valid=valid)
            weak=real_weak_alignment_loss(prediction,raster,target,supervision,stable_edges=None)
            ref=weak['components']
            support=ref['support_overflow']
            collision=ref['collision']
            regularization=(prediction['total_shift_px']/16).square().mean()
            sample=source_public_case(root,public[(idx*17)%len(public)],idx,protocol['seed'])
            syn=None
            if sample is not None:
                sh,sl,sm,sv,st,matched=sample
                syn=known_transform_loss(a(sh,sl,sm,source_valid=sv),st,matched,supervision_valid=matched)['loss']
            loss=protocol['lambda_native']*anchor['loss']+protocol['lambda_map']*ref['reference_map']+\
                protocol['lambda_regularization']*regularization
            for term,weight in ((support,protocol['lambda_support']),(collision,protocol['lambda_collision']),
                                (ref['unmatched_coverage'],.1*protocol['lambda_map'])):
                if term is not None:loss=loss+weight*term
            if syn is not None:loss=loss+protocol['lambda_synthetic']*syn
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            finite=all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
            norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True) if finite else None
            if any(p.grad is not None for p in s.parameters() if not p.requires_grad):raise ValueError('Joint frozen gradient leak')
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            success=finite and scaler.get_scale()>=before
            if success:steps+=1
            else:skips+=1
            append(out/'JOINT_TRAINING_STEPS.jsonl',dict(epoch=epoch,cursor=cursor,successful_steps=steps,
                AMP_skipped=not success,loss=scalar(loss),native_anchor_loss=scalar(anchor['loss']),
                reference_map_loss=scalar(ref['reference_map']),known_transform_loss=None if syn is None else scalar(syn),
                gradient_norm=None if norm is None else scalar(norm),source_id=row['id'],native_source_id=public[idx%len(public)]['id'],
                predicted_unmatched=scalar(prediction['unmatched_probability'].mean()),
                source_proxy_roof_mass=scalar(masks.sum()),S_soft_roof_mass=scalar(soft_masks.sum()),
                calibrated_uncertainty_m=None,track_loss=None,no_soft_instances=not bool(soft_masks.sum()>0)))
        if complete:epochs+=1
        # No frozen S parameter is allowed to drift while the heads update.
        for name,p in s.named_parameters():
            if name in frozen and not torch.equal(p.detach().cpu(),frozen[name]):raise ValueError('Joint frozen S parameter changed')
        metrics=evaluate_a(a,root,tune)
        history.append(dict(epoch=epoch,**metrics));atomic_json(out/'JOINT_TUNE_HISTORY.json',history)
        joint_checkpoint(out/'latest.pth',s,a,optimizer,scaler,epoch,steps,cursor,protocol,parents)
        if metrics['selection_positive_support_fraction_error']<best:
            best=metrics['selection_positive_support_fraction_error']
            joint_checkpoint(out/'best.pth',s,a,optimizer,scaler,epoch,steps,cursor,protocol,parents)
    if not (out/'latest.pth').exists():
        joint_checkpoint(out/'latest.pth',s,a,optimizer,scaler,0,steps,0,protocol,parents)
    result=dict(TRAINING_EXECUTED=steps>0,successful_updates=steps,AMP_skips=skips,
         completed_epochs=epochs,minimum_epoch_target=protocol.get('minimum_epoch_target',5),
         elapsed_seconds=time.monotonic()-started,model_converged='NOT_PROVEN',
         best_checkpoint=str(out/'best.pth') if (out/'best.pth').exists() else None,
         latest_checkpoint=str(out/'latest.pth'),GT_in_forward=False,
         public_native_post_joint_retention='PENDING_FROZEN_RELEASE_EVALUATION',
         no_production_promotion=True,track_constraint='NOT_AVAILABLE',
         selection_scope='A_ONLY_TUNE_METRIC_NOT_FULL_JOINT_S_GUARD_VALIDATION')
    atomic_json(out/'JOINT_TRAINING_RESULT.json',result)
    del s,a,optimizer;gc.collect();torch.cuda.empty_cache()
    return result
