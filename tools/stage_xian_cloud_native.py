"""Stage the existing16 native RGB packets for GT-free release inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(4194304),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    source=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment/SINGLE2019_FULLROI_ROOF_PROPOSALS_20261012_20/FULLROI_NATIVE_SOURCE_PACKETS.json')
    records=json.loads(source.read_text())['records'];target=run/'payload/data/native';target.mkdir(exist_ok=False)
    files={};out=[]
    for r in records:
        src=Path(r['image_path']);h=sha(src)
        if h!=r['image_sha256']:raise ValueError('Native source hash changed')
        name='data/native/'+r['region_id']+'.png';dst=run/'payload'/name;shutil.copyfile(src,dst)
        files[name]=dict(sha256=h,bytes=dst.stat().st_size)
        out.append({k:r[k] for k in ('region_id','source_window','source_raster_affine','source_raster_crs',
                                    'source_observed_date','exact_support','support_is_verified_camera_acquisition_unit')})
        out[-1].update(native_image=name,native_image_sha256=h,
                       feature_input='data/xian/'+next(x['split'] for x in json.loads((run/'payload/config/alignment_data.json').read_text())['records'] if x['id']==r['region_id'])+'/'+r['region_id']+'_input.npz')
    config=run/'payload/config/NATIVE_GT_FREE_INFERENCE_INPUTS.json'
    with config.open('x') as f:json.dump(dict(records=out,manual_GT_paths_present=False,GT_required=False,
          target_grid=dict(crs='EPSG:32649',affine=[10,0,308460,0,-10,3794750],width=256,height=256),
          input_registry_sha256=sha(source)),f,indent=2,allow_nan=False)
    files['config/'+config.name]=dict(sha256=sha(config),bytes=config.stat().st_size)
    archive=run/'cloud_native_payload.tar'
    with tarfile.open(archive,'x') as tar:
        for name in files:tar.add(run/'payload'/name,arcname=name,recursive=False)
    meta=dict(part='native',sha256=sha(archive),bytes=archive.stat().st_size,files=files,no_GT_labels=True,no_credentials=True)
    with (run/'CLOUD_NATIVE_PAYLOAD_MANIFEST.json').open('x') as f:json.dump(meta,f,indent=2)
    print(json.dumps({k:v for k,v in meta.items() if k!='files'}),flush=True)


if __name__=='__main__':main()
