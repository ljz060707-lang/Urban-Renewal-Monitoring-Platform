"""Source-only compact mask sheets; no score rank, review verdict or coordinates."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--unit',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.run.resolve();out=args.output.resolve()
    if out.parent!=root or out.exists():raise ValueError('Fresh source sheet directory inside batch only')
    work=json.loads((root/'SOURCE_ONLY_WORKLIST.json').read_text(encoding='utf-8'))
    if args.unit not in work['per_unit_targets']:raise ValueError('Unit outside frozen source scope')
    if not (root/(args.unit+'_UNIT_RECEIPT.json')).is_file():raise ValueError('Source unit not complete; do not snapshot partial groups')
    rows=[]
    for target in sorted([r for r in work['targets'] if r['region_id']==args.unit],key=lambda r:(-r['source_area_m2'],r['component_key'])):
        path=root/('FULLROI_'+target['component_key']+'_RECEIPT.json')
        row=json.loads(path.read_text(encoding='utf-8'))
        if row['component_key']!=target['component_key'] or sha(root/row['preview'])!=row['preview_sha256']:
            raise ValueError('Source preview identity/hash mismatch')
        rows.append(row)
    if len(rows)!=work['per_unit_targets'][args.unit]:raise ValueError('Completed unit group count mismatch')
    out.mkdir();index=[]
    # One case has raw + three alternatives, each 160px, with preserved aspect
    # ratio. Six cases per page keep labels and roof/facade context readable.
    side=160;case_width=side*4;case_height=side+36
    for start in range(0,len(rows),6):
        canvas=np.zeros((case_height*3,case_width*2,3),np.uint8);page_rows=[]
        for offset,row in enumerate(rows[start:start+6]):
            original=cv2.imread(str(root/row['preview']))
            if original is None or original.shape[1]%4!=0:raise ValueError('Four-panel source preview cannot be read')
            width=original.shape[1]//4;rr,cc=divmod(offset,2);y=rr*case_height;x=cc*case_width
            cv2.putText(canvas,row['component_key']+' / '+args.unit,(x+6,y+16),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
            for k in range(4):
                image=original[:,k*width:(k+1)*width]
                scale=min(side/image.shape[0],side/image.shape[1]);h=max(1,round(image.shape[0]*scale));w=max(1,round(image.shape[1]*scale))
                thumb=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
                canvas[y+36+(side-h)//2:y+36+(side-h)//2+h,x+k*side+(side-w)//2:x+k*side+(side-w)//2+w]=thumb
                cv2.putText(canvas,'RAW' if k==0 else 'MASK '+str(k-1),(x+k*side+5,y+31),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
            page_rows.append({'component_key':row['component_key'],'group_id':row['group_id'],'source_receipt':str(root/(row['group_id']+'_RECEIPT.json')),
                'original_preview':str(root/row['preview']),'original_preview_sha256':row['preview_sha256'],'sheet_slot':offset,
                'source_semantic_disposition':'NOT_YET_INSPECTED','candidate_index':None})
        name='SOURCE_SHEET_'+str(start//6+1).zfill(3)+'.png'
        if not cv2.imwrite(str(out/name),canvas):raise RuntimeError('Source sheet PNG failed')
        index.append({'page':name,'sha256':sha(out/name),'cases':page_rows})
    save_json(out/'SOURCE_SHEET_INDEX.json',{'source_unit':args.unit,'source_case_count':len(rows),'pages':index,
        'ordering':'SOURCE_PARENT_AREA_DESC_THEN_COMPONENT_ID_NOT_SAM_SCORE_OR_GT_OUTCOME',
        'sheet_pixels_not_for_coordinate_digitization':True,'no_human_reviews_or_gt_generated':True})
    print(json.dumps({'unit':args.unit,'source_cases':len(rows),'source_pages':len(index),'output':str(out)}))


if __name__=='__main__':main()
