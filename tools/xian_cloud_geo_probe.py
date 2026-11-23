"""Bounded, GT-free performance repro of frozen geometry export calls.

This does not attach to or modify the active worker. It opens predictions only,
never model checkpoints, RGB, labels, manual reference or evaluation records.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=30)
    args = p.parse_args()
    root = args.root.resolve()
    stage = 'setup'
    started = time.monotonic()
    def emit(**values):
        print(json.dumps(dict(probe='FROZEN_GEO_PERFORMANCE_ONLY', stage=stage,
                              elapsed=time.monotonic()-started, **values)), flush=True)
    def timeout(signum, frame):
        emit(status='TIMEOUT_REPRODUCED', limit_seconds=args.seconds,
             stack=[dict(file=x.filename, line=x.lineno, function=x.name)
                    for x in traceback.extract_stack(frame)[-8:]], GT_read=False)
        raise TimeoutError('Bounded geometry probe stopped')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    try:
        path = root/'payload/code/tools/xian_cloud_release_geo.py'
        spec = importlib.util.spec_from_file_location('_frozen_geo_probe', path)
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        from affine import Affine
        from shapely.geometry import shape
        from shapely.ops import unary_union
        stage = 'load_frozen_E0_predictions'
        emit(status='START')
        packet_path = root/'RELEASE/INFERENCE/E0/instances.json'
        content = packet_path.read_bytes()
        expected = g.read_json(root/'RELEASE/INFERENCE/INFERENCE_MANIFEST.json')['artifacts']['E0/instances.json']['sha256']
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Frozen prediction hash mismatch')
        packet = json.loads(content)
        del content
        emit(status='DONE', observations=len(packet['instances']))
        stage = 'geometry_sets_original_call'
        emit(status='START')
        raw, candidate, risks = g.geometry_sets(packet['instances'])
        emit(status='DONE', excluded_invalid=len(risks))
        stage = 'exact_fractions_original_call'
        emit(status='START')
        valid = unary_union([shape(row['exact_support']) for row in packet['sources']])
        api = g.transfer_api(root/'payload')
        grid = g.grid_with_affine(api, Affine(*g.TARGET_GRID['affine']), 256, 256, 'EPSG:32649')
        result = g.exact_fractions(api, grid, {'raw': raw, 'model_candidate': candidate}, valid)
        emit(status='PASS_WITHIN_BOUND', conservation_error=result['area_conservation_error_m2'], GT_read=False)
    except TimeoutError:
        return 2
    finally:
        signal.alarm(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
