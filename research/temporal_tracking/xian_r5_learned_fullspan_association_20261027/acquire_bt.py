"""Acquire only official TRAIN BT flow/change members; bounded HTTP ranges."""
import json, hashlib, struct, zlib, time
from pathlib import Path
from urllib.parse import urlencode
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('F:/xian/XIAN_R5_S4_CALIBRATION_FACADE_PREPARATION_20261019_01')
class Form(HTMLParser):
    def __init__(self): super().__init__(); self.action=None; self.values={}
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form': self.action=a.get('action')
        if tag=='input' and a.get('type')=='hidden': self.values[a['name']]=a['value']
def main():
    inv=json.loads((OLD/'BANDON_TRAIN_ZIP_MEMBER_INVENTORY.json').read_text())
    idx={x['name']:x for x in inv['members']}
    pairs=(OLD/'official_sources/lists/list_BANDON_train.txt').read_text().splitlines()
    names=sorted({'train/'+l.split()[j].lstrip('/') for l in pairs for j in (2,7)})
    spans=[]
    for n in sorted(names,key=lambda n:idx[n]['offset']):
        m=idx[n]; a=m['offset']; b=a+30+len(n.encode())+1024+m['compressed_bytes']-1
        if spans and a<=spans[-1]['end']+4096 and b-spans[-1]['start']<8*1024**2:
            spans[-1]['end']=b; spans[-1]['names'].append(n)
        else: spans.append(dict(start=a,end=b,names=[n]))
    plan=dict(members=len(names),spans=len(spans),max_bytes=sum(s['end']-s['start']+1 for s in spans),split='official_train',test_reads=0)
    assert plan['max_bytes']<128*1024**2
    (ROOT/'evidence/BANDON_BT_PLAN.json').write_text(json.dumps(plan,indent=2))
    f=Form(); f.feed((OLD/'TRAIN_RANGE_PROBE.bin').read_text(encoding='utf8'))
    assert f.action=='https://drive.usercontent.google.com/download'
    url=f.action+'?'+urlencode(f.values)
    dest=ROOT/'data/bandon_bt'; dest.mkdir(exist_ok=True)
    rec=[]; errors=[]
    def fetch(s):
        a,b=s['start'],s['end']; rp=dest/(str(a)+'_'+str(b)+'.range')
        if rp.exists(): raw=rp.read_bytes(); assert len(raw)==b-a+1
        else:
            with requests.get(url+'&range_start='+str(a),headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(20,60)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{inv["archive_bytes"]}'
                raw=b''.join(r.iter_content(1024**2)); assert len(raw)==b-a+1
            rp.write_bytes(raw)
        records=[]
        for n in s['names']:
            m=idx[n]; off=m['offset']-a; v=struct.unpack('<4s5H3I2H',raw[off:off+30]); assert v[0]==b'PK\x03\x04'
            p=off+30+v[-2]+v[-1]; payload=raw[p:p+m['compressed_bytes']]
            if v[3]==8: payload=zlib.decompress(payload,-15)
            else: assert v[3]==0
            assert len(payload)==m['bytes'] and zlib.crc32(payload)&0xffffffff==m['CRC']
            out=dest/n; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(payload)
            records.append(dict(member=n,path=str(out),sha256=hashlib.sha256(payload).hexdigest(),crc_verified=True,range=str(rp)))
        return records
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,s):s for s in spans}
        for i,fu in enumerate(as_completed(futures)):
            try: rec.extend(fu.result())
            except Exception as e: errors.append(dict(start=futures[fu]['start'],error=repr(e)))
            print(json.dumps(dict(done=i+1,total=len(spans),members=len(rec),errors=len(errors))),flush=True)
    (ROOT/'evidence/BANDON_BT_RECEIPT.json').write_text(json.dumps(dict(plan=plan,status='COMPLETE' if len(rec)==len(names) else 'INCOMPLETE',records=rec,errors=errors),indent=2))
if __name__=='__main__': main()
