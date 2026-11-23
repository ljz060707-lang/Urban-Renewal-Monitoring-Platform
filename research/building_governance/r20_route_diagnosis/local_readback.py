import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio,numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    four=read(R/'FOUR_ARMS_DETAIL.json')
    residual=0.
    for w in four:
        for arm,rec in w['fractions'].items():
            p=R/'four_arm_fraction'/arm/rec['relative_file']
            assert sha(p)==rec['sha256']
            with rasterio.open(p) as d:
                assert d.crs.to_epsg()==32649 and d.count==6 and d.transform.a==10
                z=d.read()
                for name,b in zip(d.descriptions,z):
                    residual=max(residual,abs(np.sum(b[b>=0],dtype='float64')*100-rec['union_area_m2'][name]))
    assert residual<.01
    model=R/'split_training'
    assert sha(model/'final.pt')==read(model/'TRAINING.json')['weight_sha']
    for r in read(model/'XIAN_IMAGE_ONLY_FROZEN.json').values():
        p=model/'xian'/Path(r['path']).name
        assert sha(p)==r['sha256']
        with np.load(p) as data:assert data['SPLIT'].shape==(1024,1024)
    with (model/'CURVES.csv').open() as f:curve=list(csv.DictReader(f))
    assert len(curve)==400 and [int(r['step']) for r in curve]==list(range(1,401))
    vals=np.array([float(r['loss']) for r in curve]);assert np.isfinite(vals).all()
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.plot(np.arange(1,401),vals,color='#afa5bf',lw=.8,label='Batch loss')
    ax.plot(np.arange(20,401),np.convolve(vals,np.ones(20)/20,'valid'),color='#572b77',label='20-update mean')
    ax.set(xlabel='Optimizer update',ylabel='Weighted BCE',title='Roof instance separation | source FIT');ax.legend();fig.tight_layout()
    fig.savefig(model/'TRAINING_CURVE.png',dpi=160);plt.close(fig)
    rec=read(R/'change_evidence'/'RECEIPT.json')
    for item in rec['records']:
        p=R/'change_evidence'/item['file'];assert sha(p)==item['sha256']
        with rasterio.open(p) as d:
            z=d.read(1);assert (z>=0).sum()==item['supported_px'] and (z>=.5).sum()==item['changed_px']
    assert len(four)==72 and len(rec['records'])==51
    receipt=dict(geometry_windows=len(four),fraction_tifs=len(four)*4,area_integral最大差_m2=residual,
        native_split_windows=72,learned_weight_SHA=sha(model/'final.pt'),training_steps=len(curve),
        checked_change_tifs=len(rec['records']),official_change_weight_SHA=sha(R/'MambaBCD_Tiny_WHU_verified.pth'))
    assert receipt['official_change_weight_SHA']==read(R/'change_evidence'/'LOAD.json')['sha256']
    (R/'LOCAL_READBACK.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(receipt)
if __name__=='__main__':main()

