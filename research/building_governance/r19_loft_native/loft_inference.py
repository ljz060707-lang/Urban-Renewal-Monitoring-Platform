"""Author LOFT tensors and author offset methods in the installed MMDetection 3.x inference runtime."""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmcv.ops import Conv2d
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules

R=Path('/root/xian20261111_r19_loft')
A=R/'assets'
O=R/'output';O.mkdir(exist_ok=True)
WEIGHT=A/'BONAI-LOFT-ROA-F1Score-64.31.pth'
DIGEST='2003f790d9a8a6e829cfc25b245a5db106abb0ca45b455a8910d699bec521e6b'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def author_inference_methods():
    root=A/'official_source/mmdet'
    coder=next(root.rglob('delta_xy_offset_coder.py'));head=next(root.rglob('offset_head_expand_feature.py'))
    class BaseBBoxCoder: pass
    env=dict(torch=torch,np=np,nn=nn,F=F,math=math,_pair=_pair,Conv2d=Conv2d,BaseBBoxCoder=BaseBBoxCoder)
    nodes=[]
    for node in ast.parse(coder.read_text()).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('delta2offset','offset2delta'):nodes.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='DeltaXYOffsetCoder':
            node.decorator_list=[];nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(coder),'exec'),env)
    def build_coder(cfg):
        assert cfg['type']=='DeltaXYOffsetCoder'
        return env['DeltaXYOffsetCoder'](**{k:v for k,v in cfg.items() if k!='type'})
    env.update(build_bbox_coder=build_coder,build_loss=lambda cfg:nn.SmoothL1Loss())
    cls=next(n for n in ast.parse(head.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='OffsetHeadExpandFeature')
    cls.decorator_list=[]
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'__init__','forward','expand_feature','offset_fusion','get_offsets'}]
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(head),'exec'),env)
    return env['OffsetHeadExpandFeature'],dict(offset_head_sha256=sha(head),offset_coder_sha256=sha(coder))
def model_components():
    assert sha(WEIGHT)==DIGEST
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    saved_cfg=state['meta']['config']
    for token in ["scales=[8]","num_convs=10","type='soft_nms'","bbox_type='building'","mask_type='roof'","target_stds=[0.5, 0.5]","img_scale=(1024, 1024)"]:
        assert token in saved_cfg,token
    register_all_modules(init_default_scope=True)
    cfg=Config.fromfile('/root/xian20261110_r6/code/model_base.py').model
    cfg.backbone.init_cfg=None
    cfg.roi_head.bbox_head.num_classes=1;cfg.roi_head.mask_head.num_classes=1
    cfg.test_cfg.rpn.update(nms_pre=3000,max_per_img=3000,nms=dict(type='nms',iou_threshold=.70),min_bbox_size=0)
    cfg.test_cfg.rcnn.update(score_thr=.05,nms=dict(type='soft_nms',iou_threshold=.5),max_per_img=2000,mask_thr_binary=.50)
    assert cfg.rpn_head.anchor_generator.scales==[8] and cfg.roi_head.bbox_head.reg_class_agnostic is False
    model=MODELS.build(cfg)
    assert model.rpn_head.num_convs==1
    head_cls,source_sha=author_inference_methods()
    head=head_cls(expand_feature_num=4,share_expand_fc=True,rotations=[0,90,180,270],num_fcs=2,fc_out_channels=1024,num_convs=10)
    sd=state['state_dict'];prefix='roi_head.offset_head.'
    detector={k:v for k,v in sd.items() if not k.startswith(prefix)}
    offsets={k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)}
    model.load_state_dict(detector,strict=True);head.load_state_dict(offsets,strict=True)
    assert len(sd)==446 and len(offsets)==86
    # Independently check encoded rectangle offsets and their native pixel scale.
    boxes=torch.tensor([[10.,10.,110.,50.],[0.,0.,16.,20.]])
    native=torch.tensor([[8.,-3.],[-2.,4.]])
    encoded=head.offset_coder.encode(boxes,native)
    assert torch.allclose(encoded,native/(boxes[:,2:]-boxes[:,:2])/.5)
    assert torch.allclose(head.offset_coder.decode(boxes,encoded,max_shape=(1024,1024)),native)
    save(O/'LOAD_RECEIPT.json',dict(checkpoint_sha256=DIGEST,strictly_loaded_tensors=len(sd),detection_mask_tensors=len(detector),offset_tensors=len(offsets),
        author_training_epoch_metadata=state['meta'].get('epoch'),author_iteration_metadata=state['meta'].get('iter'),
        author_mmdet=state['meta'].get('mmdet_version'),author_mmcv=state['meta'].get('mmcv_version'),
        author_offset_source_bindings=source_sha,native_pixel_coder_roundtrip=True,operator_runtime='local mmdetection 3.0.0/mmcv 2.0.1; author old runtime bit-equivalence not established',
        training_split_overlap_with_reused_source_dev_possible=True,protected_image_reads=0))
    return model.cuda().eval(),head.cuda().eval()
@torch.inference_mode()
def predict(model,head,rgb):
    assert rgb.shape==(1024,1024,3) and rgb.dtype==np.uint8
    inputs=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float().cuda()[None]
    inp=(inputs-inputs.new_tensor([123.675,116.28,103.53])[None,:,None,None])/inputs.new_tensor([58.395,57.12,57.375])[None,:,None,None]
    sample=DetDataSample(metainfo=dict(img_shape=(1024,1024),ori_shape=(1024,1024),batch_input_shape=(1024,1024),pad_shape=(1024,1024),scale_factor=(1.,1.),flip=False))
    f=model.extract_feat(inp);proposals=model.rpn_head.predict(f,[sample],rescale=False)
    inst=model.roi_head.predict(f,proposals,[sample],rescale=False)[0]
    boxes=inst.bboxes
    if not torch.is_tensor(boxes):boxes=boxes.tensor
    all_offsets=[]
    for start in range(0,len(boxes),24):
        b=boxes[start:start+24]
        rois=torch.cat([torch.zeros_like(b[:,:1]),b],1)
        # The author offset extractor and bbox extractor have identical ROIAlign and FPN settings.
        roi=model.roi_head.bbox_roi_extractor(f[:4],rois)
        y=head(roi);native=head.get_offsets(y,b,scale_factor=(1.,1.),rescale=False,img_shape=[1024,1024])
        all_offsets.append(native)
    offsets=np.concatenate(all_offsets) if all_offsets else np.empty((0,2),np.float32)
    assert np.isfinite(offsets).all() and offsets.shape==(len(boxes),2)
    return inst.scores.cpu().numpy(),boxes.cpu().numpy(),inst.masks.cpu().numpy(),offsets

