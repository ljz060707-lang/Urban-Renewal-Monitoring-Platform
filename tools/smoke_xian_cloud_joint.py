"""Actual TRAIN RGB+native-label CUDA smoke, never an optimizer update."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.payload.resolve()
    sys.path[:0]=[str(root/'code/src'),str(root/'code/RSBuilding')]
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    import opencd.models
    from opencd.registry import MODELS
    from xian_r5.cloud_joint.segmentation import RoofFacadeSegmenter,normalize_rgb,bandon_targets,native_segmentation_loss
    torch.manual_seed(20261020);torch.set_num_threads(4);init_default_scope('opencd')
    cfg=Config.fromfile(str(root/'config/B0_model.py'))
    def factory():return MODELS.build(cfg.model)
    model=RoofFacadeSegmenter.from_b0(factory,root/'weights/B0.pth').cuda().set_stage('last').train()
    rows=json.loads((root/'config/segmentation_data.json').read_text())['records']
    row=next(r for r in rows if r['split']=='TRAIN' and r['dataset']=='BANDON')
    rgb=np.asarray(Image.open(root/row['image']).convert('RGB').crop((0,0,512,512))).copy()
    label=np.asarray(Image.open(root/row['label']).crop((0,0,512,512))).copy()
    results=[]
    for batch in (1,4):
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            x=torch.from_numpy(rgb.transpose(2,0,1))[None].repeat(batch,1,1,1).cuda()
            y=torch.from_numpy(label)[None].repeat(batch,1,1).cuda();roof,facade=bandon_targets(y)
            elapsed=[]
            for repeat in range(3):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=model(normalize_rgb(x));loss=native_segmentation_loss(pred,roof,facade_target=facade,boundary_trusted=True)['loss']
                loss.backward();torch.cuda.synchronize();elapsed.append(time.monotonic()-start)
                if not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):raise ValueError('nonfinite gradients')
                if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise ValueError('frozen gradient leak')
            results.append(dict(batch=batch,status='PASS',input_size=512,stage='last',
                 median_fwbw_seconds=float(np.median(elapsed)),allocated_bytes=torch.cuda.max_memory_allocated(),
                 reserved_bytes=torch.cuda.max_memory_reserved(),loss=float(loss.detach()),successful_optimizer_updates=0))
            del x,y,roof,facade,pred,loss
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            results.append(dict(batch=batch,status='OOM_NO_UPDATES',successful_optimizer_updates=0))
    passed=[r for r in results if r['status']=='PASS']
    if not passed:raise ValueError('No real 512 batch fit')
    # Strict model-state roundtrip; original B0 bytes unchanged.
    model.zero_grad(set_to_none=True)
    state_path=a.output.parent/'SMOKE_STATE_NOT_TRAINED.pth'
    if state_path.exists() or a.output.exists():raise ValueError('Append-only smoke output already exists')
    state={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in model.state_dict().items()}
    torch.save(state,state_path)
    restored=RoofFacadeSegmenter(factory())
    restored.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True),strict=True)
    report=dict(status='PASS',real_TRAIN_sample=row['id'],measurements=results,
                recommended_batch=max(r['batch'] for r in passed),strict_state_reload=True,
                optimizer_updates=0,validation_scores_read=False,protected_tests_read=False,
                scientific_validation=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
