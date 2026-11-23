"""Metric geometry adapters; classes, coverage and unknown remain separate."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize,shapes
from shapely.geometry import shape,mapping,box,Polygon,GeometryCollection
from shapely.ops import transform,unary_union
from shapely.affinity import affine_transform,translate
from pyproj import CRS,Transformer
from .label_transfer import Grid
from .application_contract import digest

EMPTY=GeometryCollection()

def map_geometry(g, affine):
    return affine_transform(g,[affine.a,affine.b,affine.d,affine.e,affine.c,affine.f])

def convert(g, source, target):
    if CRS.from_user_input(source)==CRS.from_user_input(target):return g
    return transform(Transformer.from_crs(source,target,always_xy=True).transform,g)

def grid_support(grid):
    return Polygon([grid.affine*(x,y) for x,y in [(0,0),(grid.width,0),(grid.width,grid.height),(0,grid.height)]])

def metric_grid(template, support, source_crs):
    with rasterio.open(template) as ds:
        crs=CRS.from_user_input(ds.crs)
        if not crs.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in crs.axis_info[:2]):
            raise ValueError('Target template must define a metric projected grid; prepare a local metric template')
        full=Grid(ds.transform,ds.width,ds.height,crs.to_string())
    g=convert(support,source_crs,full.crs).intersection(grid_support(full))
    if g.is_empty:raise ValueError('No overlap with target grid')
    inv=~full.affine;xy=[inv*(x,y) for x,y in box(*g.bounds).exterior.coords]
    c0=max(0,math.floor(min(x for x,y in xy)));r0=max(0,math.floor(min(y for x,y in xy)))
    c1=min(full.width,math.ceil(max(x for x,y in xy)));r1=min(full.height,math.ceil(max(y for x,y in xy)))
    return Grid(full.affine*Affine.translation(c0,r0),c1-c0,r1-r0,full.crs)

def write_tif(path, data, grid, names, product_id, nodata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=np.asarray(data);data=data if data.ndim==3 else data[None]
    if data.shape[1:]!=(grid.height,grid.width) or len(names)!=len(data):raise ValueError('Raster grid/band mismatch')
    with rasterio.open(path,'w',driver='GTiff',height=grid.height,width=grid.width,count=len(data),dtype=data.dtype,
                       crs=grid.crs,transform=grid.affine,nodata=nodata,compress='deflate',tiled=True,blockxsize=256,blockysize=256) as ds:
        ds.write(data);ds.update_tags(product_id=product_id,product_level='research_candidate',physical_area_recovered='NO_UNLESS_INSTANCE_LINEAGE_SAYS_SO')
        for i,n in enumerate(names,1):ds.set_band_description(i,n)
    with rasterio.open(path) as ds:
        if ds.shape!=data.shape[1:] or not np.array_equal(ds.read(),data,equal_nan=True):raise ValueError('Raster write/read mismatch')

def burn(geometries,grid,values=None,dtype='uint8'):
    geometries=list(geometries);values=[1]*len(geometries) if values is None else values
    gg=[(g,int(v)) for g,v in zip(geometries,values) if not g.is_empty]
    return rasterize(gg,out_shape=(grid.height,grid.width),transform=grid.affine,dtype=dtype) if gg else np.zeros((grid.height,grid.width),dtype)

def exact(grid,geometries,support):
    """Use existing exact fraction adapter after validating metre units."""
    from .label_transfer import aggregate
    return aggregate(grid,geometries,support,events=True)

def vector(path,rows,crs):
    import geopandas as gpd
    import pandas as pd
    # JSON properties keep complete provenance without SQLite mixed Python types.
    props=[dict(observation_id=str(r.get('observation_id',r.get('event_id',''))),
                record_json=__import__('json').dumps({k:v for k,v in r.items() if k!='geometry'},ensure_ascii=False,allow_nan=False)) for r in rows]
    frame=gpd.GeoDataFrame(pd.DataFrame(props,columns=['observation_id','record_json']),geometry=[shape(r['geometry']) for r in rows],crs=crs)
    frame.to_file(path,layer='objects',driver='GPKG',engine='pyogrio',geometry_type='Unknown')

def aspect(g,truncated,threshold):
    if truncated:return dict(ratio=None,status='UNKNOWN_TRUNCATED',linear_risk=None)
    r=g.minimum_rotated_rectangle
    if r.geom_type!='Polygon':return dict(ratio=None,status='UNKNOWN_DEGENERATE',linear_risk=None)
    coords=list(r.exterior.coords);lengths=[math.dist(coords[i],coords[i+1]) for i in range(4)];short=min(lengths)
    if short<=1e-9:return dict(ratio=None,status='UNKNOWN_DEGENERATE',linear_risk=None)
    ratio=max(lengths)/short
    return dict(ratio=ratio,status='METRIC_MRR_DIAGNOSTIC_UNCALIBRATED',linear_risk=ratio>threshold,long_axis_m=max(lengths),short_axis_m=short)

def physical(g,evidence):
    if not evidence:return None,'UNRESOLVED_NO_VIEW_GEOMETRY'
    # This adapter accepts explicit, source/target-bound planar geometry only.
    if (evidence.get('method')!='VERIFIED_PLANAR_AFFINE' or not evidence.get('evidence_ref')
        or evidence.get('already_rectified') is not False or evidence.get('source_frame')==evidence.get('target_frame')
        or not evidence.get('source_frame') or not evidence.get('target_frame')):
        return None,'UNRESOLVED_UNSUPPORTED_PHYSICAL_EVIDENCE'
    m=evidence.get('affine6')
    if not isinstance(m,list) or len(m)!=6 or not all(math.isfinite(x) for x in m):raise ValueError('Invalid physical affine')
    det=m[0]*m[3]-m[1]*m[2]
    if abs(det)<1e-12:raise ValueError('Singular physical transform')
    result=affine_transform(g,m)
    if abs(result.area-abs(det)*g.area)>max(1e-6,g.area*1e-8):raise ValueError('Physical area invariant')
    return result,'REFERENCE_PLANE_TRANSFORM_EVIDENCE_BOUND_NOT_AUTOMATICALLY_QUALIFIED'

def medoid(rows):
    return min(rows,key=lambda r:(sum(shape(r['geometry']).boundary.hausdorff_distance(shape(s['geometry']).boundary) for s in rows)/len(rows),
                  float(r['position_sigma_m']) if r.get('position_sigma_m') is not None else float('inf'),r['observation_id']))

def fuse_parent_inputs(parents, destination):
    """Fuse aligned RGB acquisitions BEFORE B0 extraction. Conflicting pixels are invalidated.

    Return a packet per actual acquisition/CRS/native affine family. Different dates
    or unaligned grids remain separate; the caller does not invent a common date.
    """
    from PIL import Image
    from .application_contract import save,sha
    out=[];groups={}
    for p in parents:
        with rasterio.open(p['hr_image']) as ds:
            a=ds.transform;key=(p.get('actual_date'),p.get('source_acquisition_id'),str(ds.crs),a.a,a.b,a.d,a.e)
        groups.setdefault(key,[]).append(p)
    for index,group in enumerate(groups.values()):
        ref=group[0]
        with rasterio.open(ref['hr_image']) as ds:aff=ds.transform;crs=str(ds.crs)
        inv=~aff;windows=[]
        for p in group:
            with rasterio.open(p['hr_image']) as ds:
                x,y=inv*(ds.transform.c,ds.transform.f)
                if abs(x-round(x))>1e-5 or abs(y-round(y))>1e-5:
                    if len(group)>1:raise ValueError('Parents not on a common native grid; pre-warp explicitly with provenance')
                windows.append((int(round(x)),int(round(y)),ds.width,ds.height,p))
        x0=min(q[0] for q in windows);y0=min(q[1] for q in windows);x1=max(q[0]+q[2] for q in windows);y1=max(q[1]+q[3] for q in windows)
        rgb=np.zeros((y1-y0,x1-x0,3),np.uint8);valid=np.zeros(rgb.shape[:2],np.uint8);conflict=valid.copy();refs=[]
        for x,y,w,h,p in windows:
            with rasterio.open(p['hr_image']) as ds:
                a=ds.read([1,2,3]).transpose(1,2,0);v=ds.dataset_mask()>0
                if a.dtype!=np.uint8:raise ValueError('RGB must be uint8; supply explicit source radiometric conversion')
                if p.get('source_valid_mask'):
                    with rasterio.open(p['source_valid_mask']) as vm:
                        if vm.shape!=ds.shape or vm.transform!=ds.transform or vm.crs!=ds.crs:raise ValueError('Source validity grid mismatch')
                        v &= (vm.read(1)>0)&(vm.dataset_mask()>0)
            xx,yy=x-x0,y-y0;sub=rgb[yy:yy+h,xx:xx+w];vv=valid[yy:yy+h,xx:xx+w];cc=conflict[yy:yy+h,xx:xx+w]
            cc[(vv>0)&v&np.any(sub!=a,axis=2)]=1;sub[(vv==0)&v]=a[(vv==0)&v];vv[v]=1
            refs.append(dict(parent_id=p['parent_id'],payload_sha256=sha(p['hr_image']),path=p['hr_image'],native_window=[xx,yy,w,h],actual_date=p.get('actual_date'),source_acquisition_id=p.get('source_acquisition_id'),source_registry_ref=p.get('source_registry_ref')))
        valid[conflict>0]=0
        d=Path(destination)/('packet_%02d'%index);d.mkdir(parents=True);Image.fromarray(rgb).save(d/'RGB.png');np.save(d/'valid.npy',valid);np.save(d/'source_conflict.npy',conflict)
        out.append(dict(packet_id='packet_%02d'%index,image=str(d/'RGB.png'),rgb_pixel_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest(),valid=str(d/'valid.npy'),shape=list(valid.shape),crs=crs,affine=list(aff*Affine.translation(x0,y0))[:6],parents=refs,
                        actual_date=ref.get('actual_date'),source_acquisition_id=ref.get('source_acquisition_id'),source_conflict_pixels=int(conflict.sum()),
                        hr_building_tif=ref.get('hr_building_tif'),building_tif_schema=ref.get('building_tif_schema')))
    save(Path(destination)/'PACKETS.json',out);return out
