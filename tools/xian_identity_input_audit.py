"""Read-only identity input inventory; never repairs source databases."""
import json
from pathlib import Path
import fiona

BASE = Path('F:/xian')
paths = [
    *sorted((BASE / '西安_三期建筑追踪与三态最终正式候选版_20260916/02_三期建筑几何/remote_full_geometry').glob('*.gpkg')),
    BASE / 'XIAN_R5_MODEL_UPGRADE_20260923/18_reconstruction/B03_TEX04_PIPELINE_RECONSTRUCTION_FREEZE_FIXEDCASE_20260929_02/building_lineage.gpkg',
    BASE / 'XIAN_R5_MODEL_UPGRADE_20260923/18_reconstruction/B04-TQA10_posegraph05_aliked_qualitycodes_20260925_02/building_lineage.gpkg',
    BASE / '西安_三期建筑追踪与三态最终正式候选版_20260916/05_建筑追踪/tracking_buildings.gpkg',
    BASE / '西安_R4_城市级建筑变化最终候选版/03_建筑体_未筛选/2019-01-01/建筑体_未筛选.gpkg',
    BASE / 'XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment/GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg',
    BASE / '西安建筑轮廓数据/西安.shp',
]
for path in paths:
    print(str(path))
    try:
        for layer in fiona.listlayers(path):
            with fiona.open(path, layer=layer) as src:
                print(json.dumps({'layer': layer, 'schema':src.schema, 'crs':str(src.crs),
                    'count':len(src), 'sample':dict(next(iter(src))['properties'])},ensure_ascii=False))
    except Exception as exc:
        print(type(exc).__name__, str(exc))
