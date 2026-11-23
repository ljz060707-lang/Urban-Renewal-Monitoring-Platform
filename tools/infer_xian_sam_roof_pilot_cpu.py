"""Official SAM inference in the unchanged Python3.9 model environment, CPU only.

No imports from the Python3.10 geospatial pipeline. All output masks are
unselected source candidates, not roof labels, RBF outputs, or registration.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as s:
        for chunk in iter(lambda:s.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    with path.open('x',encoding='utf-8') as s:json.dump(value,s,ensure_ascii=False,indent=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment').resolve()
    if root.parent!=parent or (root/'SAM_INFERENCE_RECEIPT.json').exists():raise ValueError('Fresh inference in frozen pilot required')
    plan=json.loads((root/'FROZEN_SAM_PILOT_PLAN.json').read_text(encoding='utf-8'))
    if plan['device']!='cpu' or plan['automatic_mask_selection_or_formal_acceptance'] or plan['GT_or_target_positions_used']:
        raise ValueError('CPU source-only pilot policy breach')
    for path,digest in plan['protected_sha256'].items():
        if sha(path)!=digest:raise ValueError('Protected input drift: '+path)
    for path,digest in plan['new_code_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen driver drift: '+path)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    runtime=Path(plan['runtime']);sys.path.insert(0,str(runtime/'sam_source'))
    import torch,torchvision,cv2,numpy as np
    from segment_anything import SamPredictor,sam_model_registry
    torch.set_num_threads(plan['torch_threads']);torch.set_num_interop_threads(1)
    model=sam_model_registry['vit_b'](checkpoint=None)
    state=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);del state;model=model.to('cpu').eval()
    if any(p.device.type!='cpu' for p in model.parameters()):raise RuntimeError('Model left CPU')
    predictor=SamPredictor(model)
    prompts=json.loads((root/'SAM_SOURCE_ONLY_PROMPTS.json').read_text(encoding='utf-8'))['records']
    grouped={}
    for row in prompts:grouped.setdefault(row['source_parent_component'],[]).append(row)
    output=[];started=time.monotonic()
    with torch.inference_mode():
        for key,groups in grouped.items():
            image_path=groups[0]['image_path'];image=cv2.imread(image_path)
            if image is None:raise RuntimeError('Source image not readable')
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            if hashlib.sha256(image.tobytes()).hexdigest()!=groups[0]['native_rgb_pixel_sha256']:
                raise RuntimeError('Native pixel hash differs in inference environment')
            print('CPU image embedding: '+key,flush=True);predictor.set_image(image)
            print('CPU embedding complete: '+key,flush=True)
            for row in groups:
                if sha(image_path)!=row['image_sha256'] or row['reference_or_target_positions_used']:
                    raise RuntimeError('Source prompt provenance breach')
                coords=np.array(row['point_coords_pixel_index_xy'],dtype=np.float32);labels=np.array(row['point_labels'],dtype=np.int32)
                masks,scores,logits=predictor.predict(point_coords=coords,point_labels=labels,multimask_output=True)
                if masks.shape!=(3,*image.shape[:2]) or not np.isfinite(scores).all() or not np.isfinite(logits).all():
                    raise RuntimeError('SAM candidate output shape/finite contract failed')
                name=row['group_id']+'_CANDIDATES.npz';path=root/name
                if path.exists():raise ValueError('No mask overwrite')
                np.savez_compressed(path,masks=masks.astype(np.uint8),predicted_mask_quality=scores,low_resolution_logits=logits)
                panel=[]
                for index,mask in enumerate(masks):
                    rgb=image.copy();rgb[mask]=(rgb[mask].astype(float)*.65+np.array([40,255,40])*.35).astype(np.uint8)
                    cv2.putText(rgb,'MASK '+str(index)+' model-score '+format(float(scores[index]),'.3f'),(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                    for point,label in zip(coords,labels):
                        cv2.circle(rgb,tuple(np.rint(point).astype(int)),4,(40,255,40) if label else (255,40,40),-1)
                    panel.append(rgb)
                view=root/(row['group_id']+'_ALL_MASKS.png')
                if not cv2.imwrite(str(view),cv2.cvtColor(np.concatenate(panel,axis=1),cv2.COLOR_RGB2BGR)):raise RuntimeError('SAM mask overlay failed')
                output.append({'group_id':row['group_id'],'source_parent_component':key,'npz':name,'npz_sha256':sha(path),
                    'overlay':view.name,'overlay_sha256':sha(view),'mask_pixels':[int(m.sum()) for m in masks],
                    'predicted_mask_quality':[float(v) for v in scores],'selected_mask_index':None,
                    'roof_or_full_instance_verified':False,'mask_score_is_accuracy_or_uncertainty':False})
                print('Saved three unselected masks: '+row['group_id'],flush=True)
            predictor.reset_image()
    summary={'backend':'OFFICIAL_SAM_VIT_B_NOT_RBF_SAM','checkpoint_sha256':plan['checkpoint_sha256'],
        'sam_commit':plan['sam_commit'],'device':'cpu','torch_version':torch.__version__,'torchvision_version':torchvision.__version__,
        'python_version':sys.version,'image_embeddings':len(grouped),'source_groups_inferred':len(output),
        'unselected_candidate_masks':3*len(output),'elapsed_seconds':time.monotonic()-started,
        'source_records':output,'optimization_steps':0,'GPU_execution':False,
        'target_positions_or_predictions_read':False,'new_training_label_pixels':0,'registered_roofs':0,
        'physical_ids':0,'ground_footprints':0,'formal_reviews':0,'absolute_position_accuracy':'UNKNOWN',
        'RBF_SAM_status':plan['RBF_SAM_status']}
    save(root/'SAM_INFERENCE_SUMMARY.json',summary)
    for path,digest in plan['protected_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Protected source/runtime changed: '+path)
    save(root/'SAM_INFERENCE_RECEIPT.json',{'summary':{k:v for k,v in summary.items() if k!='source_records'},
        'new_code_sha256':plan['new_code_sha256'],'protected_artifacts_checked':len(plan['protected_sha256']),
        'protected_artifacts_changed':0,'output_sha256':{p.name:sha(p) for p in root.iterdir() if p.is_file()},
        'whole_window_governance_completed':False})
    print(json.dumps({k:v for k,v in summary.items() if k!='source_records'}),flush=True)


if __name__=='__main__':main()
