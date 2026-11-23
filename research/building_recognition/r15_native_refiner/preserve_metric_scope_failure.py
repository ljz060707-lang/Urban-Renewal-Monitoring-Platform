from pathlib import Path
import json,hashlib,time
R=Path('/root/xian20261111_r15');O=R/'output/source_sam_pilot';failure=O/'FAILURE_RECEIPT.json';history=O/'failure_history';history.mkdir(exist_ok=True)
if failure.exists():
 old=json.loads(failure.read_text());p=history/f"FAILURE_{int(old['time']*1000)}.json"
 if not p.exists():p.write_bytes(failure.read_bytes())
 records=json.loads((O/'ALL_PREDICTIONS_FROZEN_BEFORE_GT_JOIN.json').read_text())['frames']
 for r in records:
  d=O/'frames'/r['id']
  for name,key in [('CANDIDATES.json','candidate_sha256'),('MODEL_COMPARISON_INSTANCES.npz','instance_maps_sha256'),('IMAGE_EMBEDDING_FP32.npz','embedding_sha256')]:assert hashlib.sha256((d/name).read_bytes()).hexdigest()==r[key]
 (O/'METRIC_SCOPE_FIX_PRECHECK.json').write_text(json.dumps(dict(time=time.time(),original_failure_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original_failure_path=str(p),all_12_prediction_and_embedding_hashes_unchanged=True,fix='EXPOSE_EXTRACTED_METRIC_DEPENDENCIES_IN_FUNCTION_GLOBAL_NAMESPACE',new_model_optimizer_updates=0,model_or_input_threshold_changed=False),indent=2));print('PRESERVED_SCOPE_FAILURE_AND_VERIFIED12_FRAMES')
