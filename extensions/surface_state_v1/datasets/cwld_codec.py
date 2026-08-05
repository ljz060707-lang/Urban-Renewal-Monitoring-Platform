"""Auditable CWLD RGBA-to-class codec; never assigns ambiguous pixels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image

ANCHORS=np.asarray([[0,0,0],[255,255,255],[0,0,255],[255,0,0]],dtype=np.float32)
CLASS_NAMES=("background","vacant_land_candidate","engineering_facility_candidate","construction_waste")
IGNORE_INDEX=255

@dataclass(frozen=True)
class CWLDCodec:
    max_anchor_distance: float=96.0
    min_distance_margin: float=16.0
    ignore_index: int=IGNORE_INDEX

    def encode_array(self, rgba: np.ndarray, *, audit: bool=False):
        if rgba.ndim!=3 or rgba.shape[-1] not in (3,4): raise ValueError(f"CWLD label must be RGB/RGBA, got {rgba.shape}")
        rgb=np.asarray(rgba[...,:3],dtype=np.float32)
        distances=np.sqrt(((rgb[...,None,:]-ANCHORS)**2).sum(-1))
        nearest=distances.argmin(-1); ordered=np.partition(distances,1,axis=-1)
        exact=(distances.min(-1)==0); certain=exact|((ordered[...,0]<=self.max_anchor_distance)&((ordered[...,1]-ordered[...,0])>=self.min_distance_margin))
        out=np.full(rgb.shape[:2],self.ignore_index,dtype=np.uint8); out[certain]=nearest[certain].astype(np.uint8)
        if not audit: return out
        return out,{"mapped_fraction":float(certain.mean()),"ignored_fraction":float((~certain).mean()),"exact_anchor_fraction":float(exact.mean()),"ambiguous_pixel_count":int((~certain).sum()),"per_class_pixel_count":{CLASS_NAMES[i]:int((out==i).sum()) for i in range(4)},"alpha_unique_values":np.unique(rgba[...,3]).astype(int).tolist() if rgba.shape[-1]==4 else []}

    def encode_file(self,path,*,audit=False):
        with Image.open(path) as image: array=np.asarray(image.convert("RGBA"))
        return self.encode_array(array,audit=audit)
