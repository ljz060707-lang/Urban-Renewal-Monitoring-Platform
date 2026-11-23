from pathlib import Path
import csv,json,hashlib
import numpy as np,rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    arc=sha(R/'RETURN.tgz');assert arc==(R/'RETURN.sha256').read_text().split()[0]
    folds=[set(read(R/(role+'_IDS.json'))) for role in ('MAP_FIT','MAP_INNER')];assert not folds[0].intersection(folds[1])
    fig,ax=plt.subplots(figsize=(7.5,3.9))
    for arm in ('native','native_and_map'):
        folder=R/arm; info=read(folder/'TRAINED.json');assert sha(folder/'weights.pt')==info['weight_sha']
        with (folder/'CURVES.csv').open() as f:c=list(csv.DictReader(f))
        assert len(c)==750 and [int(r['step']) for r in c]==list(range(1,751))
        loss=np.array([float(r['loss']) for r in c]);assert np.isfinite(loss).all()
        ax.plot(np.arange(25,751),np.convolve(loss,np.ones(25)/25,'valid'),label=arm.replace('_',' '))
    ax.set(xlabel='Parameter update',ylabel='Pairwise loss, 25-step mean',title='Within-roof instance affinity');ax.legend();fig.tight_layout();fig.savefig(R/'TRAINING_CURVES.png',dpi=165);plt.close(fig)
    gate=R/'selective_split';assert sha(gate/'gate.joblib')==read(gate/'FIT.json')['weight_sha']
    snapshots=list(read(R/'TARGET_PREDICTIONS_BEFORE_GT.json').values())+[read(gate/'XIAN_FROZEN_BEFORE_GT.json')]+list(read(R/'external/MAP_AND_IMAGE_PREDICTIONS_FROZEN.json').values())+[read(gate/'limited_fragmentation/BEFORE_GT.json'),read(R/'candidate_diagnostic/PROPOSALS_BEFORE_GT.json')]
    checks=0
    for group in snapshots:
        assert len(group)==72
        for r in group.values():
            rel=r['path'].split('/root/xian20261111_r21_pair/')[1];p=R/rel;assert sha(p)==(r.get('sha256') or r.get('sha'))
            with np.load(p) as z:assert z[z.files[0]].shape==(1024,1024)
            checks+=1
    tif_count=change=0
    for fld,recs in [(R/'position',read(R/'position/DETAILS.json')),(gate/'limited_fragmentation/position',read(gate/'limited_fragmentation/POSITION_DETAILS.json'))]:
        for r in recs:
            p=fld/((r.get('id') or r['window'])+'.tif');assert sha(p)==r['sha256']
            with rasterio.open(p) as ds:
                a=ds.read();assert ds.crs.to_epsg()==32649 and ds.transform.a==1 and ds.transform.e==-1 and not (a[1:,a[0]==0]>0).any()
                change+=int(((a[2:]>0)!=(a[1]>0)).sum())
            tif_count+=1
    assert change==0 and tif_count==144
    residual=0
    fd=R/'public_position';frs=read(fd/'DETAILS.json')
    for item in frs:
        r=item['fraction'];p=fd/'fraction'/r['relative_file'];assert sha(p)==r['sha256']
        with rasterio.open(p) as ds:
            assert ds.count==6 and ds.crs.to_epsg()==32649 and ds.transform.a==10
            for n,band in zip(ds.descriptions,ds.read()):
                area=np.sum(band[band>=0],dtype='float64')*100;residual=max(residual,abs(area-r['union_area_m2'][n]))
    assert len(frs)==72 and residual<.01
    for r in read(R/'review/CASES.json'):assert sha(R/'review'/r['png'])==r['sha']
    receipt=dict(archive_sha256=arc,actual_weight_updates=1500,position_split_tifs_checked=tif_count,split_union_pixel_differences=change,
                 checked_native_target_instance_maps=checks,checked_fraction_tifs=len(frs),max_fraction_area_residual_m2=residual,
                 map_fid_overlap_between_new_training_and_inner=0,source_native_and_Xian_partial_map_development_only=True)
    (R/'LOCAL_READBACK.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(receipt)
if __name__=='__main__':main()

