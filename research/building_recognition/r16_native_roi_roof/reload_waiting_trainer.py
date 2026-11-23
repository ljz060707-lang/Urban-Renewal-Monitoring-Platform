"""Reload only the parameter-update-free waiting driver after added preflight."""
from pathlib import Path
import json,time,os,signal,subprocess,hashlib
R=Path('/root/xian20261111_r16_roi_roof');T=R/'output/training';state=json.loads((T/'STATUS.json').read_text());process=json.loads((T/'PROCESS.json').read_text());pid=process['pid'];command=Path(f'/proc/{pid}/cmdline').read_bytes();assert state['event']=='WAIT_COMPLETE_FROZEN_NATIVE_ROI_FEATURES' and state['actual_optimizer_updates']==0;assert b'/root/xian20261111_r16_roi_roof/code/train_roi_decoder.py' in command;assert not list(T.glob('*/iter_*.pt'));history=T/'pretrain_reload_history';history.mkdir(exist_ok=True);record=dict(time=time.time(),old_process=process,status_before=state,reason='ADDED_ZERO_UPDATE_GENERIC_ROI_PARENT_INTERFACE_NATIVE_MAP_REPLAY;WAITING_DRIVER_ONLY;FEATURE_ENCODER_UNTOUCHED',new_code_sha256=hashlib.sha256((R/'code/train_roi_decoder.py').read_bytes()).hexdigest(),actual_optimizer_updates_before=0);(history/(str(pid)+'.json')).write_text(json.dumps(record,indent=2));os.kill(pid,signal.SIGTERM)
for _ in range(50):
 if not Path(f'/proc/{pid}').exists():break
 time.sleep(.1)
assert not Path(f'/proc/{pid}').exists();print('RELOADED_ONLY_WAITING_DRIVER_ZERO_UPDATES',pid);subprocess.run(['/root/xian20261109/venv/bin/python',str(R/'code/launch_training.py')],check=True)
