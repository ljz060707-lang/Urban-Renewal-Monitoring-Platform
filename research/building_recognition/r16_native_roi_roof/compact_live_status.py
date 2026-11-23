from pathlib import Path
import json,datetime,subprocess,os
R=Path('/root/xian20261111_r16_roi_roof')
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None
def process(p):
 record=read(p)
 if not record:return None
 pid=record['pid'];record['process_exists']=Path(f'/proc/{pid}').exists()
 if record['process_exists']:record['actual_cmdline']=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
 return record
out=dict(cloud_time=datetime.datetime.now().astimezone().isoformat(),preparation_status=read(R/'output/STATUS.json'),training_status=read(R/'output/training/STATUS.json'),preparation_process=process(R/'output/PREPARE_PROCESS.json'),training_process=process(R/'output/training/PROCESS.json'),preparation_failure=read(R/'output/FAILURE_RECEIPT.json'),training_failure=read(R/'output/training/FAILURE_RECEIPT.json'),training_functional_receipt=read(R/'output/TRAINING_FUNCTIONAL_RECEIPT.json'),dataset_ready=(R/'data/DATASET_READY.json').exists(),training_complete=(R/'output/training/TRAINING_RECEIPT.json').exists())
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=15);out['GPU_current_sample']=gpu.stdout.strip();print(json.dumps(out))
