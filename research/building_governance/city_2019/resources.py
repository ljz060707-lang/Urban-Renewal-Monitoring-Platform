import subprocess,time,csv,os
from pathlib import Path
from datetime import datetime,timezone
import psutil
p=Path('/root/xian2019/output/resources.csv');end=datetime.fromisoformat('2026-10-29T09:25:00+08:00').timestamp()
with p.open('a',buffering=1) as f:
 w=csv.writer(f)
 if p.stat().st_size==0:w.writerow(['utc','gpu_util_percent','gpu_memory_used_mib','gpu_memory_total_mib','cpu_percent','ram_used_gib','ram_available_gib'])
 while time.time()<end:
  r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
  a=r.stdout.strip().split(',');mem=psutil.virtual_memory();w.writerow([datetime.now(timezone.utc).isoformat(),*a,psutil.cpu_percent(),round(mem.used/2**30,2),round(mem.available/2**30,2)]);time.sleep(30)
