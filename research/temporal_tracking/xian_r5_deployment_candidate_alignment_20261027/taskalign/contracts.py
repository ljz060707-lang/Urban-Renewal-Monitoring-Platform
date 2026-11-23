"""Geometry-only observations and explicit candidate/absence boundaries."""
from dataclasses import dataclass,asdict
from pathlib import Path
from datetime import date
import json,hashlib
from shapely.geometry import shape,mapping

@dataclass(frozen=True)
class Observation:
    uid:str
    epoch:str
    time_key:str
    metric_geometry:dict
    native_geometry:dict
    source:str
    image_path:str=''
    image_sha256:object=None
    proposal_kind:str='PROVIDED_GEOMETRY'
    alpha_support:object=None
    source_roof_observed:object=True
    actual_date:object=None
    date_precision:str="EPOCH_ORDER_KEY"
    coordinate_contract:str='NATIVE_X_RIGHT_Y_DOWN_AND_PROJECTED_METRES'
    def __post_init__(self):
        date.fromisoformat(self.time_key)
        if self.actual_date is not None and self.actual_date!=self.time_key:raise ValueError("Actual-day and ordering key disagree")
        for field in ['metric_geometry','native_geometry']:
            g=shape(getattr(self,field))
            if not g.is_valid or g.is_empty or g.area<=0:raise ValueError('Invalid observation geometry: '+self.uid)
        if not self.uid or not self.epoch:raise ValueError('Missing immutable observation UID/epoch')
    def to_dict(self):return asdict(self)

@dataclass(frozen=True)
class CandidatePolicy:
    radius_projected_m:float=64.
    max_candidates:int=32
    include_all_epoch_pairs:bool=True
    dustbin_semantics:str='NO_ACCEPTED_CORRESPONDENCE_NOT_PHYSICAL_ABSENCE'
    def __post_init__(self):
        if self.radius_projected_m<=0 or self.max_candidates<1:raise ValueError('Invalid frozen range policy')
    @property
    def sha256(self):return digest(asdict(self))

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def lines(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
