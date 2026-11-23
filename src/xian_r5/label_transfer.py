"""Shadow-only label transport: exact target-grid fractions, never ground GT.

Observed geometry is immutable. Missing transformations and observability are
explicit; physical footprints and formal alignment acceptance are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from affine import Affine
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.affinity import translate


@dataclass(frozen=True)
class Grid:
    affine: Affine
    width: int
    height: int
    crs: str

    def __post_init__(self):
        if (self.width < 1 or self.height < 1 or
                not all(math.isfinite(v) for v in tuple(self.affine)[:6]) or
                abs(self.affine.determinant) < 1e-12):
            raise ValueError('Invalid grid')

    def pixel_to_map(self, col, row):
        return self.affine * (col, row)

    def map_to_pixel(self, x, y):
        return ~self.affine * (x, y)

    def window(self, bounds):
        xmin, ymin, xmax, ymax = bounds
        if not all(math.isfinite(v) for v in bounds) or xmin >= xmax or ymin >= ymax:
            raise ValueError('Invalid bounds')
        corners = [self.map_to_pixel(x,y) for x,y in
                   ((xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax))]
        c0,r0 = (math.floor(min(p[i] for p in corners)+1e-8) for i in (0,1))
        c1,r1 = (math.ceil(max(p[i] for p in corners)-1e-8) for i in (0,1))
        if c0 < 0 or r0 < 0 or c1 > self.width or r1 > self.height:
            raise ValueError('Support overflow outside target grid')
        return Grid(self.affine * Affine.translation(c0,r0),c1-c0,r1-r0,self.crs)

    def nested(self, factor):
        if isinstance(factor,bool) or not isinstance(factor,int) or factor < 1:
            raise ValueError('Positive integer nesting factor required')
        return Grid(self.affine * Affine.scale(1/factor),
                    self.width*factor,self.height*factor,self.crs)

    def cell(self, col, row):
        if not 0 <= col < self.width or not 0 <= row < self.height:
            raise ValueError('Cell outside grid')
        return Polygon([self.pixel_to_map(c,r) for c,r in
                        ((col,row),(col+1,row),(col+1,row+1),(col,row+1))])


def aggregate(grid, classes, valid_geometry, *, events=False):
    """Exact polygon fractions of full cell area, including unknown denominator.

Unassigned valid area is returned separately, NOT inferred known background.
Overlapping exclusive classes fail closed; explicitly separate event channels
may overlap. Vector inputs must already be in this metric target frame.
"""
    from pyproj import CRS
    measured_crs = CRS.from_user_input(grid.crs)
    if not measured_crs.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in measured_crs.axis_info[:2]):
        raise ValueError('Exact area fractions require a metric projected grid')
    if not valid_geometry.is_valid or any(not g.is_valid for g in classes.values()):
        raise ValueError('Invalid input geometry; no silent buffer repair')
    clipped = [g.intersection(valid_geometry) for g in classes.values()]
    merged = unary_union(clipped)
    if not events and sum(g.area for g in clipped)-merged.area > 1e-6:
        raise ValueError('Exclusive classes overlap; conflict must be explicit')
    fractions = np.zeros((len(clipped),grid.height,grid.width),dtype=np.float64)
    valid = np.zeros((grid.height,grid.width),dtype=np.float64)
    unassigned = valid.copy()
    area = abs(grid.affine.determinant)
    for row in range(grid.height):
        for col in range(grid.width):
            cell = grid.cell(col,row)
            v = cell.intersection(valid_geometry)
            valid[row,col] = v.area / area
            if v.is_empty:
                continue
            for band,geometry in enumerate(clipped):
                fractions[band,row,col] = v.intersection(geometry).area / area
            unassigned[row,col] = max(0.,(v.area-v.intersection(merged).area)/area)
    footprint = Polygon([grid.pixel_to_map(c,r) for c,r in
                         ((0,0),(grid.width,0),(grid.width,grid.height),(0,grid.height))])
    expected = sum(g.intersection(footprint).area for g in clipped)
    return dict(fractions=fractions,valid=valid,unknown=1-valid,
                unassigned=unassigned,classes=list(classes),
                area_conservation_error_m2=abs(float(fractions.sum())*area-expected))


def transport(geometry, vector, support):
    """Forward roof-label translation, with an explicit authorized support."""
    if vector is None or len(vector)!=2 or not all(math.isfinite(v) for v in vector):
        raise ValueError('Unknown transform is not a zero transform')
    if not geometry.is_valid or not support.is_valid:
        raise ValueError('Invalid geometry')
    output=translate(geometry,xoff=vector[0],yoff=vector[1])
    if not support.covers(geometry) or not support.covers(output):
        raise ValueError('Support overflow; no extrapolation')
    return output


def deduplicate(rows):
    """One object per source/date/ID; incompatible duplicates must be resolved."""
    unique={}
    for key,geometry in rows:
        if key in unique and not geometry.equals(unique[key]):
            raise ValueError('Source/object conflict: '+str(key))
        unique[key]=geometry
    return unique


def coordinate_states(before, after, presence, *, dates, frames):
    """Coordinate already-qualified Presence, not infer absence from detections.

Stable geometry policy is frozen to earlier observation. Identity keys and
reliable zero evidence belong to the upstream producer. Separate event regions
may overlap (demolition then rebuilding); callers must retain both channels.
"""
    if (len(dates)!=2 or len(frames)!=2 or not all(isinstance(d,str) for d in dates)
            or frames[0]!=frames[1] or date.fromisoformat(dates[0])>=date.fromisoformat(dates[1])):
        raise ValueError('Incompatible frame or temporal interval')
    regions={k:[] for k in ('STABLE','ADDITION','DEMOLITION','UNRESOLVED')}
    for key in before.keys() | after.keys() | presence.keys():
        values=tuple(presence.get(key,('U','U')))
        if len(values)!=2 or any(v not in (0,1,'U') for v in values):
            raise ValueError('Presence must be 1/0/U, not state codes')
        if values==(1,1) and key in before and key in after:
            regions['STABLE'].append(before[key])
        elif values==(0,1) and key in after:
            regions['ADDITION'].append(after[key])
        elif values==(1,0) and key in before:
            regions['DEMOLITION'].append(before[key])
        else:
            regions['UNRESOLVED'].extend(g[key] for g in (before,after) if key in g)
    return {k:unary_union(v) for k,v in regions.items()}


def web_row(row, zoom, source_convention, target_convention):
    """Explicit web-grid conventions; GEID metadata is not a web XYZ grid."""
    if source_convention not in {'XYZ','TMS'} or target_convention not in {'XYZ','TMS'}:
        raise ValueError('Unknown convention; GEID must use MosaicTileGrid')
    if not isinstance(zoom,int) or not 0<=zoom<=30 or not isinstance(row,int) or not 0<=row<2**zoom:
        raise ValueError('Zero-based row outside web grid')
    return row if source_convention==target_convention else 2**zoom-1-row


def nearest_codes(array, source_corner_pixels):
    """Categorical sampling uses containing source cell, never numeric averages."""
    if array.ndim!=2 or array.dtype.kind not in 'iu':
        raise ValueError('Integer categorical source required')
    result=[]
    for col,row in source_corner_pixels:
        if not math.isfinite(col) or not math.isfinite(row):
            raise ValueError('Unknown categorical sample location')
        c,r=math.floor(col),math.floor(row)
        if not 0<=c<array.shape[1] or not 0<=r<array.shape[0]:
            raise ValueError('Categorical sampling outside support')
        result.append(int(array[r,c]))
    return result


def perturb(objects, region_vectors, object_vectors=None):
    """Deterministic scenarios, preserving source correlation and object shape."""
    object_vectors=object_vectors or {}
    output=[]
    for dx,dy in region_vectors:
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError('Invalid stress vector')
        output.append({key:translate(g,xoff=dx+object_vectors.get(key,(0,0))[0],
                                      yoff=dy+object_vectors.get(key,(0,0))[1])
                       for key,g in objects.items()})
    return output


def bounded_transport(objects, vector, source_support, target_support, object_vectors=None):
    """Clip source evidence BEFORE forward mapping and move its validity with it.

Halo objects contribute only their verified source fragments. The destination
remains inside frozen target support; unavailable source data cannot move in.
"""
    if vector is None or len(vector)!=2 or not all(math.isfinite(v) for v in vector):
        raise ValueError('Unknown bounded transform')
    fragments={key:g.intersection(source_support) for key,g in objects.items()}
    moved=perturb(fragments,[vector],object_vectors)[0]
    valid=translate(source_support,xoff=vector[0],yoff=vector[1]).intersection(target_support)
    return {key:g.intersection(valid) for key,g in moved.items()},valid


def training_mask(valid, bandwidth, *, envelope, temporal):
    if valid.shape!=bandwidth.shape or not np.isfinite(valid).all() or not np.isfinite(bandwidth).all():
        raise ValueError('Quality grid/values mismatch')
    if envelope is None or not temporal:
        return np.zeros(valid.shape,dtype=np.uint8)
    if not math.isfinite(envelope) or envelope<0:
        raise ValueError('Invalid admission envelope')
    return ((valid>=.95)&(bandwidth<=.10)).astype(np.uint8)


def validate_input_role(role):
    if role not in {'OBSERVED_CANDIDATE','MANUAL_ROOF_REFERENCE','SOURCE_COVERAGE'}:
        raise ValueError('Input role forbidden in label construction: '+role)


def supervision_eligibility(*, time_ok, geometry_ok, state_ok, interface_ok):
    """Audit explicit evidence masks without treating unknown as permission.

    Counts by cause overlap. Exclusive combinations partition the original
    pixel denominator; they must not be summed with the inclusive counts.
    Geometry permission must come from validated errors, never stress rasters.
    """
    masks = dict(TIME=time_ok, GEOMETRY=geometry_ok,
                 STATE_EVIDENCE=state_ok, INTERFACE=interface_ok)
    if any(np.ma.isMaskedArray(m) for m in masks.values()):
        raise ValueError('Masked unknowns require explicit evidence masks')
    shape = np.asarray(time_ok).shape
    if len(shape) != 2 or not all(shape):
        raise ValueError('Nonempty two-dimensional evidence masks required')
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.shape != shape or mask.dtype.kind != 'b':
            raise ValueError('Evidence masks must be explicit booleans on the same grid')
        masks[name] = mask
    failures = np.stack([~m for m in masks.values()])
    eligible = ~failures.any(axis=0)
    counts = {'denominator': eligible.size, 'eligible': int(eligible.sum()),
              'EXCLUDED_BY_MULTIPLE_CAUSES': int((failures.sum(axis=0)>1).sum())}
    counts.update({'EXCLUDED_BY_'+name: int((~m).sum()) for name,m in masks.items()})
    reasons = {}
    for pixel in failures.reshape(len(masks),-1).T:
        key = '+'.join(name for name,failed in zip(masks,pixel) if failed) or 'NONE'
        reasons[key] = reasons.get(key,0)+1
    return dict(eligible=eligible, counts=counts, exclusive_reason_counts=reasons)


def fraction_supervision(fractions, *, valid, unknown, eligible):
    """Soft-target dataloader seam, not a model/loss activation or GT certificate.

    Retain each fraction per entire pixel. Independent event channels may
    overlap; each channel must stay within known support. Eligibility comes
    from the separate evidence gate. No hard-label majority rule is needed.
    """
    if any(np.ma.isMaskedArray(v) for v in (fractions,valid,unknown,eligible)):
        raise ValueError('Masked unknowns must be represented explicitly')
    fractions = np.asarray(fractions,dtype=np.float64)
    valid = np.asarray(valid,dtype=np.float64)
    unknown = np.asarray(unknown,dtype=np.float64)
    eligible = np.asarray(eligible)
    if (fractions.ndim!=3 or fractions.shape[0]<1 or valid.ndim!=2
            or fractions.shape[1:]!=valid.shape or unknown.shape!=valid.shape
            or eligible.shape!=valid.shape or eligible.dtype.kind!='b'):
        raise ValueError('Soft target, validity and eligibility grids disagree')
    for values in (fractions,valid,unknown):
        if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
            raise ValueError('Fractions must be finite in [0,1]')
    if not np.allclose(valid+unknown,1,rtol=0,atol=1e-6):
        raise ValueError('Validity must retain the full pixel denominator')
    if (fractions>valid[None]+1e-6).any() or ((valid==0)&eligible).any():
        raise ValueError('Target/eligibility outside known support')
    return dict(target=fractions.copy(),valid=valid.copy(),unknown=unknown.copy(),
                weight=np.where(eligible,valid,0.),denominator='FULL_TARGET_PIXEL')


def export_raster(path, array, grid, bands, *, root, nodata=None):
    """Write and reopen a temporary TIFF before atomically registering it.

The caller supplies its independently authorized shadow root, never an input
path. Identical completed exports resume without rewrites; drift fails closed.
"""
    import rasterio
    path=Path(path).resolve(); root=Path(root).resolve()
    if not path.is_relative_to(root) or path==root or path.parent!=root:
        raise ValueError('Output outside authorized shadow directory')
    array=np.asarray(array)
    if array.shape!=(len(bands),grid.height,grid.width) or not np.isfinite(array).all():
        raise ValueError('Raster grid/bands or finite values invalid')
    if array.dtype.kind=='f': array=array.astype(np.float32)
    expected_shape=array.shape
    if path.exists():
        with rasterio.open(path) as src:
            if (src.transform!=grid.affine or str(src.crs)!=grid.crs or src.nodata!=nodata
                    or list(src.descriptions)!=bands or not np.array_equal(src.read(),array)):
                raise ValueError('Resume content mismatch')
        return
    fd,tmp=tempfile.mkstemp(prefix='.partial_',suffix='.tif',dir=root);os.close(fd)
    try:
        with rasterio.open(tmp,'w',driver='GTiff',width=grid.width,height=grid.height,
                           count=len(bands),crs=grid.crs,transform=grid.affine,
                           dtype=array.dtype,nodata=nodata,compress='deflate',
                           tiled=True,blockxsize=256,blockysize=256) as dst:
            dst.write(array)
            for index,name in enumerate(bands,1): dst.set_band_description(index,name)
            dst.update_tags(branch='GT10M_LABEL_TRANSFER_BRANCH',scope='SHADOW_ONLY',AREA_OR_POINT='Area')
        with rasterio.open(tmp) as src:
            if src.read().shape!=expected_shape or not np.array_equal(src.read(),array):
                raise RuntimeError('Written TIFF verification failed')
        # No overwrite: link registration is atomic on the local NTFS volume.
        os.link(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load_rasters(paths):
    """Dataloader seam: unmasked values, exact grid and shape agreement."""
    import rasterio
    arrays=[];grid=None;bands=[]
    for path in paths:
        with rasterio.open(path) as src:
            current=Grid(src.transform,src.width,src.height,str(src.crs))
            if grid is not None and current!=grid:
                raise ValueError('Training label/quality grid mismatch')
            grid=current;arrays.append(src.read(masked=False));bands.append(list(src.descriptions))
    if grid is None: raise ValueError('No rasters')
    return dict(arrays=arrays,grid=grid,bands=bands)
