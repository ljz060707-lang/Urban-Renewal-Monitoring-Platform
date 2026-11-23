"""Append-only snapshot of this run's followup adapters and local restore job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    args = p.parse_args()
    run, repo = args.run.resolve(), args.repo.resolve()
    # Preserve any partial failed attempt rather than overwriting its evidence.
    stage = run / 'FOLLOWUP_SNAPSHOT_02'
    stage.mkdir(exist_ok=False)
    files = [
        'src/xian_r5/cloud_joint/alignment.py', 'src/xian_r5/label_transfer.py',
        'tools/xian_cloud_a_train.py', 'tools/xian_cloud_joint_phase.py',
        'tools/run_xian_cloud_followup.py', 'tools/xian_cloud_release_inference.py',
        'tools/xian_cloud_release_geo.py', 'tools/xian_cloud_public_final.py',
    ]
    new_members = []
    for relative in files:
        source = repo / relative
        target = stage / 'payload/code' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        new_members.append(target)
    for name in ('xian_cloud_export_manifest.py',):
        target = stage / 'incoming' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / 'tools' / name, target)
        new_members.append(target)
    protected = {
        'payload/code/tools/xian_cloud_s_train.py': '62bb74da06131dbba6161286ad253ce201a7cef1e95589d184cbb081539a1434',
        'payload/code/src/xian_r5/cloud_joint/segmentation.py': '3adbe2a525d74e5daa71cee2975616dddf6a8ed22370eba8894258ee6aa1447b',
    }
    for relative, expected in protected.items():
        if sha(run / relative) != expected:
            raise ValueError('Frozen live S source changed: ' + relative)
    hashes = {path.relative_to(stage).as_posix(): sha(path) for path in new_members}
    hashes.update(protected)
    for relative in ('payload/config/segmentation_data.json', 'payload/config/alignment_data.json',
                     'payload/config/NATIVE_GT_FREE_INFERENCE_INPUTS.json',
                     'payload/config/MANUAL_REFERENCE_GEOMETRY.json',
                     'payload/config/A_REFERENCE_FID_CROSSWALK.json'):
        local_source = run / relative
        if not local_source.is_file() and Path(relative).name in (
                'MANUAL_REFERENCE_GEOMETRY.json', 'A_REFERENCE_FID_CROSSWALK.json'):
            local_source = run / Path(relative).name
        hashes[relative] = sha(local_source)
    hashes['incoming/PHASED_TRAINING_PROTOCOL.json'] = sha(run / 'PHASED_TRAINING_PROTOCOL.json')
    manifest = dict(schema='XIAN_CLOUD_FOLLOWUP_FROZEN_V1', input_sha256=hashes,
        wait_for='S_PHASE_EXIT.json', no_overwrite_live_S_inputs=True,
        new_final_evaluation='ONCE_AFTER_MODEL_FREEZE_NO_SELECTION', production_admission=False)
    write(stage / 'FOLLOWUP_LAUNCH_MANIFEST.json', manifest)
    new_members.append(stage / 'FOLLOWUP_LAUNCH_MANIFEST.json')
    archive = run / 'cloud_followup_payload.tar'
    with tarfile.open(archive, 'x') as tar:
        for path in new_members:
            tar.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
    write(run / 'FOLLOWUP_DEPLOYMENT_MANIFEST.json', dict(archive_sha256=sha(archive),
        bytes=archive.stat().st_size, files=hashes, excludes_private_credentials=True))
    local = run / 'LOCAL_OPERATIONS'
    local.mkdir(exist_ok=False)
    for name in ('restore_xian_cloud_joint.py', 'watch_xian_cloud_joint_return.py'):
        shutil.copy2(repo / 'tools' / name, local / name)
    torch_python = r'C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe'
    geo_python = r'C:\Users\ljz52\.conda\envs\xian-r5-fixedcase-20260925\python.exe'
    argv = [geo_python, '-B', '-X', 'utf8', str(local / 'restore_xian_cloud_joint.py'),
            '--run', str(run), '--torch-python', torch_python, '--geo-python', geo_python]
    write(run / 'RESTORE_COMMAND.json', dict(argv=argv,
        code_sha256=sha(local / 'restore_xian_cloud_joint.py'),
        GT_free=True, strict_model_and_geometry_replay_required=True))
    print(json.dumps(dict(status='FROZEN_FOLLOWUP_STAGED', archive=str(archive),
                         sha256=sha(archive), input_files=len(hashes))))


if __name__ == '__main__':
    main()
