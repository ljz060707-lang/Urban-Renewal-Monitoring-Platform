"""Keep original manual GT FIDs distinct from derived GPKG feature IDs.

Append-only metadata correction: packed A reference_FIDs values are layer FIDs,
not original shapefile FIDs. No image/label/geometry is changed or re-matched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import fiona
from shapely.geometry import shape


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda:s.read(4194304),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    graph=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/23_building_object_identity/XIAN_R5_IDENTITY_MINIMAL_CLOSURE_20261015/IDENTITY_GRAPH_MINIMAL.gpkg')
    source=Path('F:/xian/西安建筑轮廓数据/西安.shp');rows=[]
    with fiona.open(graph,layer='manual_roof_reference') as derived,fiona.open(source) as original:
        for f in derived:
            prop=dict(f.properties);fid=str(prop['source_fid']);raw=original[int(fid)]
            if raw is None or str(raw.id)!=fid:raise ValueError('Original FID lookup failed')
            rows.append(dict(gpkg_feature_id=str(f.id),GT_original_FID=fid,
                 GT_original_nonunique_Id=prop['original_id'],GT_source=str(source),
                 GT_original_geometry_sha256=hashlib.sha256(shape(raw.geometry).wkb).hexdigest(),
                 reference_metric_geometry_sha256=hashlib.sha256(shape(f.geometry).wkb).hexdigest(),
                 metric_geometry_semantics='CRS_TRANSFORM_ONLY_NO_POSITION_CORRECTION'))
    files={str(q):sha(q) for q in [graph,*[source.with_suffix(s) for s in ('.shp','.shx','.dbf','.prj') if source.with_suffix(s).exists()]]}
    data=dict(schema='A_MANUAL_GT_FID_CROSSWALK_V1',rows=rows,source_files_sha256=files,
         correction='PACKED_REFERENCE_FIDs_ARRAYS_ARE_DERIVED_GPKG_FEATURE_IDS_USE_THIS_CROSSWALK_FOR_ORIGINAL_FID',
         geometry_or_label_changed=False,identity_correspondences_created=0,
         manual_reference_semantics='2019_ROOF_REFERENCE_NOT_GROUND_FOOTPRINT',
         absolute_position_accuracy='UNKNOWN',GT_model_forward_input=False)
    target=a.run/'A_REFERENCE_FID_CROSSWALK.json'
    with target.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='PASS',reference_objects=len(rows),output_sha256=sha(target))),flush=True)


if __name__=='__main__':main()
