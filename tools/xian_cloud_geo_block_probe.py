"""Fixed-window differential performance test; predictions only, no GT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sweep',action='store_true')
    p.add_argument('--row',type=int,default=0);p.add_argument('--col',type=int,default=0)
    p.add_argument('--blocks',type=int,nargs='+',default=[32,16,8,4]);args=p.parse_args()
    root=args.root.resolve();stage='load';started=time.monotonic()
    def emit(**values):
        print(json.dumps(dict(probe='FIXED_32_WINDOW_EXACT_PARTITION',stage=stage,
                              elapsed=time.monotonic()-started,**values)),flush=True)
    def stop(signum,frame):
        emit(status='TIMEOUT',GT_read=False);raise TimeoutError()
    signal.signal(signal.SIGALRM,stop);signal.alarm(50)
    try:
        import numpy as np
        from affine import Affine
        from shapely.geometry import shape,Polygon
        from shapely.ops import unary_union
        path=root/'payload/code/tools/xian_cloud_release_geo.py'
        spec=importlib.util.spec_from_file_location('_frozen_geo_partition_probe',path)
        g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
        packet_path=root/'RELEASE/INFERENCE/E0/instances.json';content=packet_path.read_bytes()
        expected=g.read_json(root/'RELEASE/INFERENCE/INFERENCE_MANIFEST.json')['artifacts']['E0/instances.json']['sha256']
        if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Frozen predictions changed')
        packet=json.loads(content);del content
        raw,candidate,risks=g.geometry_sets(packet['instances'])
        valid=unary_union([shape(r['exact_support']) for r in packet['sources']])
        api=g.transfer_api(root/'payload')
        if args.sweep:
            for row in range(0,256,32):
                for col in range(0,256,32):
                    stage='window_%d_%d'%(row,col);t=time.monotonic();emit(status='START')
                    aff=Affine(*g.TARGET_GRID['affine'])*Affine.translation(col,row)
                    sub=g.grid_with_affine(api,aff,32,32,'EPSG:32649')
                    area=Polygon([sub.pixel_to_map(c,r) for c,r in ((0,0),(32,0),(32,32),(0,32))])
                    support=valid.intersection(area)
                    local={'raw':raw.intersection(area),'model_candidate':candidate.intersection(area)}
                    value=api.aggregate(sub,local,support,events=True)
                    emit(status='DONE',seconds=time.monotonic()-t,conservation_error=value['area_conservation_error_m2'])
            return
        aff=Affine(*g.TARGET_GRID['affine'])*Affine.translation(args.col,args.row)
        grid=g.grid_with_affine(api,aff,32,32,'EPSG:32649')
        results={};timings={}
        for block in args.blocks:
            stage='block_'+str(block);emit(status='START');t=time.monotonic()
            value=g.exact_fractions(api,grid,{'raw':raw,'model_candidate':candidate},valid,block=block)
            timings[block]=time.monotonic()-t;results[block]=value
            error=float(np.max(np.abs(value['fractions']-results[args.blocks[0]]['fractions'])))
            coverage_error=float(np.max(np.abs(value['valid']-results[args.blocks[0]]['valid'])))
            emit(status='DONE',seconds=timings[block],fraction_max_difference=error,
                 coverage_max_difference=coverage_error,conservation_error=value['area_conservation_error_m2'])
            if error>1e-9 or coverage_error>1e-9:raise ValueError('Exact partition mismatch')
        emit(status='PASS_EXACT_PARTITION_EQUIVALENCE',timings=timings,GT_read=False,
             physical_accuracy_claim=False,active_worker_modified=False)
    finally:signal.alarm(0)


if __name__=='__main__':main()
