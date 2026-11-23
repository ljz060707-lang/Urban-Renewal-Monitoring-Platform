"""Strict F-drive model/numerical and geometry replay, without rereading GT."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--torch-python', required=True)
    parser.add_argument('--geo-python', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    returned = run / 'incoming/cloud'
    exports = read(run / 'CLOUD_EXPORT_FINAL_MANIFEST.json')
    # Exact transfer verification precedes all deserialization/execution.
    for relative, record in exports['files'].items():
        path = returned / relative
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Returned artifact mismatch: ' + relative)
    original = run / 'payload'
    view = run / 'F_GT_FREE_RESTORE_VIEW'
    view.mkdir(exist_ok=False)
    def link(source, relative):
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(source)
    for path in (returned / 'payload/code').rglob('*'):
        if path.is_file() and (path.suffix == '.py' or path.name in ('LICENSE', 'LICENSE.txt')):
            link(path, path.relative_to(returned / 'payload'))
    for name in ('B0_model.py', 'NATIVE_GT_FREE_INFERENCE_INPUTS.json'):
        link(returned / 'payload/config' / name, Path('config') / name)
    link(original / 'weights/B0.pth', Path('weights/B0.pth'))
    for row in read(view / 'config/NATIVE_GT_FREE_INFERENCE_INPUTS.json')['records']:
        for key in ('native_image', 'feature_input'):
            link(original / row[key], Path(row[key]))
    output = run / 'F_LOCAL_RESTORE'
    output.mkdir(exist_ok=False)
    inference = view / 'code/tools/xian_cloud_release_inference.py'
    geo = view / 'code/tools/xian_cloud_release_geo.py'
    release = returned / 'RELEASE'
    commands = [
        (args.torch_python, inference, ['--payload', str(view), '--release-dir', str(release),
            '--replay-directory', str(release / 'INFERENCE/E3'), '--output', str(output / 'TORCH')]),
        (args.geo_python, geo, ['--payload', str(view), '--predictions', str(release / 'INFERENCE'),
            '--output', str(output / 'GEO'), '--exports-only', '--expected-geo', str(returned / 'RELEASE_GEO')]),
    ]
    execution = []
    for index, (python, script, flags) in enumerate(commands):
        result = subprocess.run([python, '-B', '-X', 'utf8', str(script), *flags],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        (output / ('stage%d_stdout.txt' % index)).write_text(result.stdout, encoding='utf-8')
        (output / ('stage%d_stderr.txt' % index)).write_text(result.stderr, encoding='utf-8')
        execution.append(dict(stage=index, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError('Local restore stage failed: %d' % index)
    torch_report = read(output / 'TORCH/GT_FREE_REPLAY_RESULT.json')
    geo_report = read(output / 'GEO/EXPORT_REPLAY_COMPARISON.json')
    passed = torch_report['status'] == 'PASS' and geo_report['status'] == 'PASS'
    report = dict(status='PASS' if passed else 'FAIL_REPLAY', execution=execution,
                  torch=torch_report, geometry=geo_report, GT_read=False,
                  scientific_validation=False, production_admission=False)
    with (output / 'RESTORE_RECEIPT.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
