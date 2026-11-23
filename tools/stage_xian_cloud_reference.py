"""Training/evaluation-only reference polygons; never in GT-free input view."""
import argparse
import hashlib
import json
from pathlib import Path
import fiona
from shapely.geometry import shape,mapping


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    graph=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/23_building_object_identity/XIAN_R5_IDENTITY_MINIMAL_CLOSURE_20261015/IDENTITY_GRAPH_MINIMAL.gpkg')
    rows=[]
    with fiona.open(graph,layer='manual_roof_reference') as source:
        for feature in source:
            g=shape(feature.geometry);prop=dict(feature.properties)
            rows.append(dict(GT_original_FID=str(prop['source_fid']),derived_gpkg_FID=str(feature.id),
                  geometry=mapping(g),geometry_sha256=hashlib.sha256(g.wkb).hexdigest(),
                  crs='EPSG:32649',year=2019,actual_date=None))
    target=a.run/'MANUAL_REFERENCE_GEOMETRY.json'
    with target.open('x',encoding='utf-8') as f:json.dump(dict(records=rows,
        role='TRAINING_AND_EVALUATION_ONLY_NEVER_MODEL_FORWARD',
        semantics='REFERENCE_BUILDING_PLANIMETRY_MANUAL_ROOF',
        derived_geometry_operation='CRS_ONLY_NO_DISPLACEMENT',
        coverage_completeness='UNKNOWN_NOT_CERTIFIED_BACKGROUND',
        absolute_position_accuracy='UNKNOWN',no_new_identity_matches=True),f,ensure_ascii=False,allow_nan=False)
    print(json.dumps(dict(reference_objects=len(rows),sha256=hashlib.sha256(target.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':main()
