"""Resumable full-ROI weak source proposals, per-group hash-verified receipts."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment').resolve()
    if root.parent!=parent or (root/'FULLROI_INFERENCE_RECEIPT.json').exists():raise ValueError('Dedicated incomplete full-ROI run only')
    plan=json.loads((root/'FROZEN_FULLROI_SAM_PLAN.json').read_text(encoding='utf-8'))
    prep=json.loads((root/'FULLROI_PREPARATION_RECEIPT.json').read_text(encoding='utf-8'))
    if plan['device']!='cpu' or plan['automatic_mask_selection'] or plan['GT_or_target_pixels_or_predictions_used']:
        raise ValueError('Source-only CPU policy breach')
    for path,digest in plan['protected_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen source drift: '+path)
    for path,digest in plan['new_code_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen batch code drift: '+path)
    for name,digest in prep['output_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Source packet/preparation drift: '+name)
    os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.path.insert(0,str(Path(plan['runtime'])/'sam_source'))
    import torch,torchvision,cv2,numpy as np
    from segment_anything import SamPredictor,sam_model_registry
    torch.set_num_threads(plan['torch_threads']);torch.set_num_interop_threads(1)
    model=sam_model_registry['vit_b'](checkpoint=None)
    model.load_state_dict(torch.load(plan['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    model=model.to('cpu').eval();predictor=SamPredictor(model)
    if any(p.device.type!='cpu' for p in model.parameters()):raise RuntimeError('Model device not CPU')
    packets=json.loads((root/'FULLROI_NATIVE_SOURCE_PACKETS.json').read_text(encoding='utf-8'))['records']
    prompts=json.loads((root/'FULLROI_SOURCE_PRIOR_PROMPTS.json').read_text(encoding='utf-8'))['records']
    results=[];started=time.monotonic()
    with torch.inference_mode():
        for packet in packets:
            groups=[r for r in prompts if r['source_packet_id']==packet['region_id']];pending=[]
            for row in groups:
                done=root/(row['group_id']+'_RECEIPT.json')
                if done.exists():
                    record=json.loads(done.read_text(encoding='utf-8'))
                    if sha(root/record['npz'])!=record['npz_sha256']:raise ValueError('Completed group hash drift')
                    results.append(record)
                else:pending.append(row)
            if not pending:continue
            image=cv2.cvtColor(cv2.imread(packet['image_path']),cv2.COLOR_BGR2RGB)
            if hashlib.sha256(image.tobytes()).hexdigest()!=packet['native_rgb_pixel_sha256']:raise ValueError('Native source pixel mismatch')
            print('CPU encoding '+packet['region_id']+'; source proposals '+str(len(pending)),flush=True)
            predictor.set_image(image)
            for row in pending:
                path=root/(row['group_id']+'_CANDIDATES.npz')
                if path.exists():raise ValueError('Uncommitted mask file retained; cannot overwrite or silently restart: '+str(path))
                xy=np.asarray(row['point_coords_pixel_index_xy'],np.float32);labels=np.asarray(row['point_labels'],np.int32)
                bbox=np.asarray(row['source_box_pixel_index_xyxy'],np.float32)
                masks,scores,logits=predictor.predict(point_coords=xy,point_labels=labels,box=bbox,multimask_output=True)
                if masks.shape!=(3,*image.shape[:2]) or not np.isfinite(scores).all() or not np.isfinite(logits).all():raise RuntimeError('SAM output invalid')
                np.savez_compressed(path,masks=masks.astype(np.uint8),predicted_mask_quality=scores)
                # Compact evidence view only; original masks/native image are
                # saved intact. No compact coordinates drive segmentation.
                positive=np.argwhere(masks.any(axis=0));x0,y0,x1,y1=bbox
                if len(positive):
                    y0=min(y0,positive[:,0].min());y1=max(y1,positive[:,0].max()+1)
                    x0=min(x0,positive[:,1].min());x1=max(x1,positive[:,1].max()+1)
                x0=max(0,int(x0)-20);y0=max(0,int(y0)-20);x1=min(image.shape[1],int(x1)+21);y1=min(image.shape[0],int(y1)+21)
                if x1<=x0 or y1<=y0:raise RuntimeError('Evidence crop invalid')
                panels=[image[y0:y1,x0:x1].copy()]
                for index,mask in enumerate(masks):
                    rgb=image[y0:y1,x0:x1].copy();part=mask[y0:y1,x0:x1]
                    rgb[part]=(rgb[part].astype(float)*.65+np.array([40,255,40])*.35).astype(np.uint8)
                    cv2.putText(rgb,'MASK '+str(index),(6,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);panels.append(rgb)
                panel=np.concatenate(panels,axis=1)
                scale=min(1.,1800/panel.shape[1])
                if scale<1:panel=cv2.resize(panel,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                preview=root/(row['group_id']+'_RAW_AND_MASKS.png')
                if not cv2.imwrite(str(preview),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR)):raise RuntimeError('Evidence view failed')
                record={'group_id':row['group_id'],'component_key':row['component_key'],'region_id':packet['region_id'],
                    'source_geometry_wkb_sha256':row['source_geometry_wkb_sha256'],
                    'image_sha256':packet['image_sha256'],'source_window':packet['source_window'],
                    'original_mask_shape':list(masks.shape),'npz':path.name,'npz_sha256':sha(path),
                    'predicted_mask_quality_diagnostic_only':[float(v) for v in scores],
                    'mask_pixels':[int(m.sum()) for m in masks],'preview':preview.name,'preview_sha256':sha(preview),
                    'preview_native_crop':[x0,y0,x1-x0,y1-y0],'preview_scale':scale,
                    'weak_prior_foreground_is_verified_roof':False,'selected_mask_index':None,
                    'roof_or_full_instance_verified':False,'formal_gt':False,'registration_tie':False,'physical_id':None}
                save(root/(row['group_id']+'_RECEIPT.json'),record);results.append(record)
                if len(results)%10==0:print('Hash-committed source groups '+str(len(results))+'/'+str(len(prompts)),flush=True)
            predictor.reset_image()
            save(root/(packet['region_id']+'_UNIT_RECEIPT.json'),{'region_id':packet['region_id'],'source_groups':len(groups),
                'all_groups_hash_committed':True,'roof_candidates_selected':0,'formal_gt':0})
    if len(results)!=plan['source_objects_requested']:raise RuntimeError('Full-ROI requested group count differs from completed count')
    summary={'source_units_completed':len(packets),'source_groups_completed':len(results),
        'unselected_source_masks':3*len(results),'elapsed_seconds':time.monotonic()-started,
        'device':'cpu','GPU_execution':False,'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,
        'source_model':'OFFICIAL_SAM_VIT_B_NOT_RBF_SAM','candidate_masks_selected':0,
        'new_registration_vectors_or_apps':0,'new_training_labels':0,'formal_gt':0,'whole_window_governance_completed':False}
    save(root/'FULLROI_INFERENCE_SUMMARY.json',summary)
    for path,digest in plan['protected_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Protected source changed: '+path)
    save(root/'FULLROI_INFERENCE_RECEIPT.json',{'summary':summary,'new_code_sha256':plan['new_code_sha256'],
        'protected_artifacts_checked':len(plan['protected_sha256']),'protected_artifacts_changed':0,
        'group_receipts_sha256':{(r['group_id']+'_RECEIPT.json'):sha(root/(r['group_id']+'_RECEIPT.json')) for r in results},
        'whole_window_governance_completed':False})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
