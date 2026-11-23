from pathlib import Path
import json,numpy as np
R=Path('/root/xian20261111_r15');O=R/'output/roof_decoder'
d=json.loads((R/'output/source_sam_pilot/QUERY_GT_CEILING_DIAGNOSTIC.json').read_text());rows=d['rows'];raw=np.array([r['raw_IoU'] for r in rows]);best=np.array([r['oracle_best_candidate']['IoU'] for r in rows]);score=np.array([max(r['candidates'],key=lambda x:x['predicted_quality'])['IoU'] for r in rows])
out=dict(n=len(rows),raw_mean_IoU=float(raw.mean()),best_of_9_GT_diagnostic_only_mean_IoU=float(best.mean()),self_quality_selected_mean_IoU=float(score.mean()),raw_IoU50=int((raw>=.5).sum()),best_candidate_IoU50=int((best>=.5).sum()),oracle_not_product_or_evaluation=True)
(R/'output/source_sam_pilot/COMPACT_QUERY_CEILING_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2));print('CEILING',json.dumps(out))
base=json.loads((O/'INNER_RAW_BASELINE.json').read_text());keys=['F1_50','pixel_IoU','pixel_recall','complete_roof_recall90','boundary_F1_native2px','merged_predictions','fragmented_GT'];print('INNER_RAW',json.dumps({k:base[k] for k in keys}))
for p in sorted(O.glob('iter_*_INNER.json'),key=lambda x:int(x.stem.split('_')[1])):
 v=json.loads(p.read_text());print(p.name,json.dumps({n:{k:m[k] for k in keys} for n,m in v['metrics'].items()}))
print('STATUS', (O/'STATUS.json').read_text())
