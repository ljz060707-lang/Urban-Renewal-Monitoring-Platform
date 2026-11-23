"""Isolated official SAM dependency acquisition; not RBF-SAM, no environment edits."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,shutil,sys,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json

COMMIT='dca509fe793f601edb92606367a655c15ac00fdf'
WEIGHTS_URL='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
SOURCE_URL='https://codeload.github.com/facebookresearch/segment-anything/zip/'+COMMIT


def download(url,path,maximum_bytes,expected_bytes=None):
    partial=path.with_suffix(path.suffix+'.part')
    if path.exists() or partial.exists():raise ValueError('No download overwrite')
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Xian-R5-candidate-runtime'}),timeout=30) as source,partial.open('xb') as output:
        if source.status!=200:raise ValueError('Unexpected download status')
        count=0;next_report=25*1024*1024
        while True:
            chunk=source.read(1024*1024)
            if not chunk:break
            count+=len(chunk)
            if count>maximum_bytes:raise ValueError('Download exceeds frozen limit')
            output.write(chunk)
            if count>=next_report:print(path.name+' downloaded '+str(count)+' bytes',flush=True);next_report+=25*1024*1024
    if expected_bytes is not None and count!=expected_bytes:raise ValueError('Downloaded size differs from official endpoint HEAD')
    partial.rename(path)
    return {'url':url,'bytes':count,'sha256':sha(path),'hash_is_local_download_fingerprint_not_published_upstream_attestation':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/22_roof_candidate_segmentation').resolve()
    if out.parent!=parent or out.exists():raise ValueError('Fresh isolated dependency directory required')
    if shutil.disk_usage(parent.parent).free<1500*1024*1024:raise ValueError('Insufficient dependency workspace free space')
    out.mkdir(parents=True)
    save_json(out/'FROZEN_ACQUISITION_PLAN.json',{'official_repository':'facebookresearch/segment-anything',
        'commit':COMMIT,'source_url':SOURCE_URL,'weights_url':WEIGHTS_URL,
        'expected_checkpoint_bytes_from_official_HEAD':375042383,'runtime_is_RBF_SAM':False,
        'device_policy':'CPU_ONLY_NO_GPU','existing_conda_environment_modified':False,
        'imagery_uploaded':False,'new_model_training':False,'new_code_sha256':{str(Path(__file__).resolve()):sha(__file__)}})
    source=download(SOURCE_URL,out/'official_sam_source.zip',150*1024*1024)
    package=out/'sam_source';package.mkdir()
    extracted=[]
    with zipfile.ZipFile(out/'official_sam_source.zip') as archive:
        if archive.testzip() is not None:raise ValueError('Official code ZIP CRC error')
        for member in archive.infolist():
            parts=PurePosixPath(member.filename).parts
            if len(parts)==1 and member.is_dir():continue
            if '..' in parts or member.filename.startswith('/') or len(parts)<2:raise ValueError('Unsafe source ZIP path')
            relative=PurePosixPath(*parts[1:])
            include=(relative.parts[0]=='segment_anything' and relative.suffix=='.py') or str(relative) in ('LICENSE','README.md','setup.py')
            if not include:continue
            target=(package/str(relative)).resolve()
            if package.resolve() not in target.parents:raise ValueError('Source extraction escaped runtime')
            if member.file_size>4*1024*1024:raise ValueError('Unexpected source file size')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(archive.read(member))
            extracted.append(target)
    if not (package/'segment_anything/predictor.py').is_file():raise ValueError('Official SAM predictor absent')
    print('Pinned official SAM Python source extracted; downloading ViT-B checkpoint.',flush=True)
    weights=download(WEIGHTS_URL,out/'sam_vit_b_01ec64.pth',400*1024*1024,375042383)
    if not zipfile.is_zipfile(out/'sam_vit_b_01ec64.pth'):raise ValueError('Unexpected checkpoint container')
    save_json(out/'OFFICIAL_SAM_RUNTIME_RECEIPT.json',{'source':source,'weights':weights,'commit':COMMIT,
        'source_files_sha256':{str(f.relative_to(package)):sha(f) for f in extracted},
        'execution_policy':'CPU; torch.load(weights_only=True); strict state_dict; eval; no training',
        'RBF_SAM_status':'NOT_RUN_OFFICIAL_RBF_CHECKPOINT_UNAVAILABLE','SAM_inference_status':'NOT_YET_RUN',
        'existing_model_environments_or_weights_modified':False,'provider_imagery_uploaded':False})
    print(json.dumps({'runtime':str(out),'checkpoint_bytes':weights['bytes'],'checkpoint_sha256':weights['sha256'],'inference':'NOT_YET_RUN'}))


if __name__=='__main__':main()
