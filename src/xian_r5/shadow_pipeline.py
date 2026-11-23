"""Live RGB adapter over existing A/I/Presence/fraction seams, shadow-only.

Two existing runtimes exchange hash-bound files. No training, GT inputs,
second object store, numerical displacement fallback or model promotion.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import fiona
from PIL import Image
from affine import Affine
from pyproj import Transformer
from shapely.geometry import mapping, shape, box, Polygon
from shapely.ops import transform, unary_union
from .integration import (CONTRACT, verify_artifacts, sha256, read_json,
                          adapt_native_components, attach_single_date_links,
                          verify_source_dates, run_r5_state, build_gt10m_shadow)
from .alignment.instance_adaptive import predict_instance_alignment
from .label_transfer import Grid, export_raster, load_rasters
from .fraction_shadow import QUALITY_BANDS, NODATA

MODE = 'LIVE_B0_RGB_SHADOW_V1'
B0_SHA = 'f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7'
REPO = Path(__file__).resolve().parents[2]

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def freeze_live_manifest(path, root):
    """Fixed first existing region, unchanged full target grid, no score access."""
    import io
    import unittest
    sys.path.insert(0,str(REPO/'tests'))
    modules=['test_xian_live_shadow','test_xian_integration','test_xian_instance_adaptive','test_xian_fraction_shadow']
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    if not tests.wasSuccessful():raise RuntimeError('Live handoff regressions failed: '+log.getvalue())
    root = Path(root)
    old = root/'24_system_integration/XIAN_R5_INTEGRATION_CONTRACT_V1_20261015_04'
    prior = read_json(old/'FROZEN_INTEGRATION_MANIFEST.json')
    before = read_json(old/'FINAL_RECEIPT.json')
    names = ['i_receipt','i_graph','i_links','i_proofs','i_recovery','i_metrics',
             'components','source_registry','window_config','grid_template','actual_date_registry','b_packets']
    pins = {k:prior['inputs'][k] for k in names}
    for k,p in pins.items():
        if p['sha256'] != before['verified_input_sha256'][k]:
            raise ValueError('Historical integration input binding changed')
    pins.update({k:v for k,v in prior['inputs'].items() if k.startswith('tile_')})
    def pin(key,p,role,expected=None):
        p=Path(p).resolve();h=sha256(p)
        if expected is not None and h!=expected:raise ValueError('Pinned input mismatch: '+key)
        pins[key]={'path':str(p),'sha256':h,'role':role}
    packets=read_json(pins['b_packets']['path'])['records']
    packet=min(packets,key=lambda r:r['region_id'])
    pin('native_rgb',packet['image_path'],'MODULE_B_NATIVE_RGB',packet['image_sha256'])
    pin('historical_integration_receipt',old/'FINAL_RECEIPT.json','MODULE_B_RECEIPT')
    pin('a_instance_receipt',root/'21_source_partition_alignment/XIAN_R5_INSTANCE_ADAPTIVE_ALIGNMENT_20261019/FINAL_RECEIPT.json','MODULE_A_RECEIPT')
    pin('a_region_receipt',root/'21_source_partition_alignment/A_LOCAL_ALIGNMENT_NUMERICAL_ROBUSTNESS_20261013/FINAL_RECEIPT.json','MODULE_A_RECEIPT')
    s=Path('F:/xian/XIAN_R5_S2_SUPERVISED_ROOF_TRAINING_20261019')
    pin('roof_runtime',s/'roof_runtime.py','CODE')
    model_info=read_json(Path('F:/xian/XIAN_R5_ROOF_FACADE_UPGRADE_20261013/RUNTIME_MODEL_LOAD_AND_LOSS_CHECK.json'))
    checkpoint=Path(model_info['checkpoint_path'])
    pin('B0',checkpoint,'MODULE_B_WEIGHTS',B0_SHA)
    pin('B0_config',checkpoint.parent/'Swin_T_whu_windows_pilot5000.py','CODE')
    pin('legacy_roof_loss_adapter',Path('F:/xian/XIAN_R5_ROOF_FACADE_UPGRADE_20261013/roof_training_adapter.py'),'CODE')
    pin('S411_receipt',Path('F:/xian/XIAN_R5_S411_SOURCE_SUPERVISION_AUDIT_20261020/FINAL_RECEIPT.json'),'MODULE_B_RECEIPT')
    pin('S410_receipt',Path('F:/xian/XIAN_R5_S410_PAIRED_ATTRIBUTION_20261020/FINAL_RECEIPT.json'),'MODULE_B_RECEIPT')
    pin('S48_receipt',Path('F:/xian/XIAN_R5_S48_SUPERVISION_ADMISSION_20261020/FINAL_RECEIPT.json'),'MODULE_B_RECEIPT')
    pin('S48_coverage',Path('F:/xian/XIAN_R5_S48_SUPERVISION_ADMISSION_20261020/S4_8_USED_VS_UNUSED_COVERAGE.json'),'MODULE_B_RECEIPT')
    for p in sorted((REPO/'src/xian_r5').rglob('*.py')):
        pin('code_'+p.relative_to(REPO).as_posix(),p,'CODE')
    for p in [REPO/'tools/run_xian_r5_integration.py',REPO/'tools/xian_r5_live_worker.py',REPO/'tests/test_xian_live_shadow.py']:
        pin('code_'+p.name,p,'CODE')
    for name in modules:
        pin('test_'+name,REPO/'tests'/(name+'.py'),'CODE')
    verify_artifacts(pins)
    doc={'contract_version':CONTRACT,'execution_mode':MODE,'project':'XIAN-R5-GROUNDREG-FREEZE-0929',
         'model_role':'ORIGINAL_B0_ONLY','run_role':'SHADOW_CANDIDATE','manual_gt_required_for_inference':False,
         'scope_rule':'LEXICOGRAPHIC_FIRST_EXISTING_NATIVE_PACKET_NO_MODEL_SELECTION',
         'packet':packet,'inputs':pins,'runtime_python':r'C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe',
         'geo_python':str(Path(sys.executable).resolve()),'worker_path':str(REPO/'tools/xian_r5_live_worker.py'),
         'instance_checkpoint':None,'reviewed_alignment_fields':None,
         'regression':{'tests_run':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),'passed':tests.wasSuccessful(),'modules':modules,'actual_log':log.getvalue()},
         'expected':{'historical_observations':1662,'historical_single_date_links':53,'target_width':256,'target_height':256},
         'rgb_runtime_semantics':'VISIBLE_ROOF_IN_SOURCE_IMAGE_FRAME',
         'inference_protocol':'existing roof_runtime:256 window,stride192,weighted blending,B0 roof channel0 sigmoid>.5; frozen not searched'}
    save(path,doc)
    return len(pins)

def new_records(result, packet, tiles):
    """Keep runtime IDs and pixel polygons; CRS conversion is not alignment."""
    from shapely.strtree import STRtree
    aff=Affine(*packet['source_raster_affine'])
    c0,r0,w,h=packet['source_window']
    tf=Transformer.from_crs(packet['source_raster_crs'],'EPSG:32649',always_xy=True)
    def project(x,y,z=None):
        xx,yy=aff*(np.asarray(x)+c0,np.asarray(y)+r0)
        return tf.transform(xx,yy)
    geoms=[shape(t['support_geometry']) for t in tiles];tree=STRtree(geoms)
    records=[];invalid=[]
    for row in result['ROOF_INSTANCE_CANDIDATES']:
        pixel=shape(row['geometry_pixel_edge_xy']);g=transform(project,pixel)
        key=row['candidate_id']
        item={'contract_version':CONTRACT,'observation_id':key,'original_id':key,
              'source_object_id':None,'geometry':mapping(g),'geometry_sha256':hashlib.sha256(g.wkb).hexdigest(),
              'native_geometry_sha256':hashlib.sha256(pixel.wkb).hexdigest(),'native_pixel_geometry':mapping(pixel),
              'native_pixel_area':row['pixel_area'],'geometry_status':'NATIVE_ROOF_CANDIDATE_NOT_VERIFIED_INSTANCE',
              'native_resampling':'NONE','position_semantics':'VISIBLE_ROOF_IN_SOURCE_IMAGE_FRAME','crs':'EPSG:32649',
              'source_path':packet['image_path'],'source_layer':None,'source_fid':key,
              'source_edge_partial_risk':row['source_edge_partial_risk'],
              'model_probability_uncalibrated':row['mean_probability_uncalibrated'],
              'building_object_ids':[],'physical_id':None,'registered_geometry':None,
              'ground_footprint':None,'GT_USED_FOR_GEOMETRY':False,'GT_USED_FOR_IDENTITY':False}
        if not g.is_valid or g.is_empty or g.geom_type not in ('Polygon','MultiPolygon'):
            item.update(geometry_status='INVALID_NATIVE_POLYGON_UNRESOLVED_NO_REPAIR',actual_date=packet['source_observed_date'],source_support_complete=False)
            invalid.append(item);continue
        hits=[int(i) for i in tree.query(g,predicate='intersects') if g.intersection(geoms[int(i)]).area>1e-6]
        dates={tiles[i]['actual_date'] for i in hits}
        if dates and dates!={packet['source_observed_date']}:raise ValueError('Fresh observation actual-date conflict')
        support=unary_union([geoms[i] for i in hits]);outside=g.difference(support).area
        item.update(actual_date=packet['source_observed_date'] if hits else None,
                    date_status='REGISTRY_DATE_COHERENT' if hits else 'UNKNOWN',
                    source_support_complete=bool(hits) and outside<=1e-6,
                    source_support_uncovered_m2=outside,
                    source_tile_ids=[tiles[i]['download_tile_id'] for i in hits],
                    source_paths=[tiles[i]['source_JPG'] for i in hits],source_hashes=[tiles[i]['source_JPG_sha256'] for i in hits])
        records.append(item)
    if len({r['observation_id'] for r in records+invalid})!=len(records+invalid):raise ValueError('Duplicate fresh ID')
    return records,invalid

def validate_shadow_manifest(m):
    if (m.get('execution_mode')!=MODE or m.get('contract_version')!=CONTRACT or
        m.get('model_role')!='ORIGINAL_B0_ONLY' or m.get('run_role')!='SHADOW_CANDIDATE' or
        m.get('manual_gt_required_for_inference') is not False or
        m.get('instance_checkpoint') is not None or m.get('reviewed_alignment_fields') is not None or
        m['inputs']['B0']['sha256']!=B0_SHA):
        raise ValueError('Shadow mode cannot promote a development model or accept unreviewed transforms')

def run_shadow(manifest_path, output):
    m=read_json(manifest_path);validate_shadow_manifest(m)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('New append-only output required')
    if any(Path(p['path']).resolve().is_relative_to(output) for p in m['inputs'].values()):raise ValueError('Output/input overlap')
    before=verify_artifacts(m['inputs']);manifest_hash=sha256(manifest_path)
    output.mkdir(parents=True,exist_ok=False);save(output/'S4_12_END_TO_END_RUN_MANIFEST.json',m)
    phases={};started=time.perf_counter()
    def path(k):return Path(m['inputs'][k]['path'])
    def worker(stage):
        subprocess.run([m['runtime_python'],'-B','-X','utf8',m['worker_path'],stage,
                        '--manifest',str(manifest_path),'--output',str(output)],check=True)
    try:
        packet=m['packet'];rgb=np.array(Image.open(path('native_rgb')).convert('RGB'))
        if sha256(path('native_rgb'))!=packet['image_sha256'] or list(rgb.shape[:2])!=packet['local_image_shape']:
            raise ValueError('Native RGB packet mismatch')
        # Model subprocess only accepts RGB, source metadata, original weights.
        print('LIVE_B0_RGB_ONLY_INFERENCE',flush=True);worker('infer')
        phases['S']={'execution':'EXECUTED','semantics':'VISIBLE_ROOF_IN_SOURCE_IMAGE_FRAME','quality':'UNCALIBRATED_CANDIDATE'}
        result=read_json(output/'ROOF_CANDIDATES_NATIVE.json')
        reg=read_json(path('source_registry'));tiles=reg['tiles']
        verify_source_dates(tiles,path('actual_date_registry'))
        window=read_json(path('window_config'));aff=Affine(10,0,308460,0,-10,3794750)
        grid=Grid(aff,256,256,'EPSG:32649')
        # Same fixed target window as the prior integrated run, not a new ROI.
        import rasterio
        with rasterio.open(path('grid_template')) as ref:
            from rasterio.windows import Window
            old_window=window.get('reference_window') or window.get('window')
            if ref.crs.to_epsg()!=32649 or ref.transform.a!=10 or ref.transform.e!=-10:raise ValueError('Target grid template mismatch')
            col,row=(~ref.transform)*(grid.affine.c,grid.affine.f)
            if abs(col-round(col))>1e-6 or abs(row-round(row))>1e-6:raise ValueError('Target grid origin not on template')
            if round(col)<0 or round(row)<0 or round(col)+256>ref.width or round(row)+256>ref.height:raise ValueError('Target grid outside template')
            if ref.window_transform(Window(round(col),round(row),256,256))!=grid.affine:raise ValueError('Frozen target affine mismatch')
        records,invalid=new_records(result,packet,tiles)
        if not records:raise ValueError('No valid candidate polygons; no pretend downstream execution')
        save(output/'FRESH_OBSERVATIONS_RAW.json',{'records':records,'invalid_native_geometries_unmodified':invalid})
        inputs=[dict(observation_id=r['observation_id'],roof_polygon_raw=shape(r['geometry']),roof_instance_quality=r['geometry_status']) for r in records]
        sources={r['observation_id']:dict(source_component_id=None,actual_date=r['actual_date'],
                 native_image_frame={'packet_id':packet['region_id'],'source_window':packet['source_window'],'crs':packet['source_raster_crs'],'affine':packet['source_raster_affine']},
                 source_support_complete=r['source_support_complete'],source_geometry_hash=r['geometry_sha256'],
                 source_payload_hashes=r['source_hashes'],source_tile_ids=r['source_tile_ids'],photographic_seam_status='UNKNOWN') for r in records}
        print('A_INSTANCE_INTERFACE_AND_STRICT_APPLICATION_GATE',flush=True)
        a=predict_instance_alignment(rgb,inputs,sources,grid,checkpoint=None)
        if any(r['total_dx_m'] is not None or r['registered_geometry'] is not None for r in a):raise ValueError('Unqualified displacement promoted')
        phases['A']={'execution':'EXECUTED_UNRESOLVED_INTERFACE','numerical_predictor':'NOT_RUN_NO_QUALIFIED_CHECKPOINT','numeric_candidates':0,'accepted':0,'applied':0,'target_semantics':'OBSERVED_ROOF_TO_PLANIMETRIC_ROOF','reference':str(path('grid_template')),'independent_CHECK':'MISSING'}
        # Revalidate historical evidence only in its exact old observation scope.
        print('I_SCOPED_SOURCE_LINK_HANDOFF',flush=True)
        old_records=adapt_native_components(path('components'),tiles)
        links,closed=attach_single_date_links(old_records,path('i_links'),path('i_graph'),[read_json(path('i_proofs')),read_json(path('i_recovery'))])
        if closed!=53 or len(old_records)!=1662:raise ValueError('Historical I scope changed')
        if {r['observation_id'] for r in records}&{r['observation_id'] for r in old_records}:raise ValueError('Fresh ID overwrites old observation')
        identity={'execution':'EXECUTED','historical_single_date_source_links_revalidated':closed,'historical_observations':1662,
                  'historical_links_scope':'ORIGINAL_OBSERVATIONS_ONLY_NOT_FRESH_MODEL_COMPONENTS','source_ledger_path':str(path('i_links')),'source_ledger_sha256':sha256(path('i_links')),
                  'fresh_records':[{'observation_id':r['observation_id'],'original_id':r['original_id'],'source_object_id':None,'building_object_ids':[],
                                    'geometry_hash':r['geometry_sha256'],'source_RGB_hash':packet['image_sha256'],'physical_id':None,
                                    'source_relation':'MODEL_OUTPUT_DERIVED_FROM_NATIVE_RGB','historical_object_relation':'UNRESOLVED_NO_EXACT_SOURCE_DERIVATION_PROOF'} for r in records],
                  'fresh_traceable_building_object_links':0,'cross_date_physical_ids':0,'historical_tracking_not_physical_id':True,'manual_GT_geometry_read':False}
        save(output/'S4_12_SOURCE_IDENTITY_HANDOFF.json',identity)
        phases['I']={'execution':'EXECUTED','historical_links_preserved':53,'fresh_building_links':0,'physical_ids':0}
        states,review=run_r5_state(records,{'ACCEPTED_LOCAL_REGIONS':0,'APPLIED_SHADOW_LOCAL_REGIONS':0},[str(path('grid_template'))])
        if states.transitions or any(p['presence']!='U' for p in states.presence):raise ValueError('Single date candidate promoted to change')
        save(output/'PRESENCE_AND_STATE_SHADOW.json',{'presence':states.presence,'transitions':states.transitions,'actual_date_order':states.actual_date_order,'physical_ids':0,'state':'UNKNOWN_SINGLE_DATE','invalid_geometry_objects':'UNRESOLVED_NOT_PASSED_TO_STATE_ENGINE','invalid_count':len(invalid)})
        phases['R5']={'execution':'EXECUTED','Presence_U':len(records),'transitions':0,'single_date':True}
        native_file=output/'S4_12_INSTANCE_ALIGNMENT_CANDIDATES.gpkg'
        schema={'geometry':'Unknown','properties':{'observation_id':'str','original_id':'str','source_object_id':'str','physical_id':'str','actual_date':'str','geometry_hash':'str','target_semantics':'str','total_dx_m':'float','total_dy_m':'float','uncertainty_m':'float','region_dx_m':'float','region_dy_m':'float','residual_dx_m':'float','residual_dy_m':'float','prediction_status':'str','validation_status':'str','application_status':'str','presence':'str','geometry_status':'str'}}
        ap={r['observation_id']:r for r in a}
        with fiona.open(native_file,'w',driver='GPKG',layer='roof_alignment_candidates',crs='EPSG:32649',schema=schema) as dst:
            for r in records+invalid:
                q=ap.get(r['observation_id']);dst.write({'geometry':r['geometry'],'properties':dict(observation_id=r['observation_id'],original_id=r['original_id'],source_object_id=None,physical_id=None,actual_date=r['actual_date'],geometry_hash=r['geometry_sha256'],target_semantics='OBSERVED_ROOF_TO_PLANIMETRIC_ROOF',total_dx_m=None,total_dy_m=None,uncertainty_m=None,region_dx_m=None,region_dy_m=None,residual_dx_m=None,residual_dy_m=None,prediction_status=q['prediction_status'] if q else 'NOT_RUN_INVALID_NATIVE_GEOMETRY',validation_status='NOT_VALIDATED',application_status='UNRESOLVED',presence='U',geometry_status=r['geometry_status'])})
        with fiona.open(native_file,'w',driver='GPKG',layer='registered_roofs_qualified',crs='EPSG:32649',schema=schema):pass
        print('EXACT_GT10M_SHADOW_AGGREGATION',flush=True)
        fullsupport=unary_union([shape(t['support_geometry']) for t in tiles]);rgb_support=shape(packet['exact_support']).intersection(fullsupport)
        fraction,eligibility,supervision=build_gt10m_shadow(records,[packet],rgb_support,grid,states)
        raw=fraction['raw_fraction'];qualified=np.full(raw.shape,NODATA,np.float32)
        outpaths=[]
        def tif(n,array,bands,nodata=None):
            p=output/n;export_raster(p,array,grid,bands,root=output,nodata=nodata);outpaths.append(p)
        tif('S4_12_GT10M_FRACTION_SHADOW.tif',np.stack([raw,qualified]),['NATIVE_ROOF_CANDIDATE','VERIFIED_TRANSFORM_ROOF_ONLY'],NODATA)
        tif('S4_12_GT10M_QUALITY_SHADOW.tif',fraction['quality'],QUALITY_BANDS)
        tif('SOURCE_SUPPORT_FRACTION.tif',fraction['scope_fraction'][0:1],['SOURCE_SPATIAL_SUPPORT_NOT_SUPERVISION_VALID'])
        tif('GT10M_VALID_UNKNOWN.tif',np.stack([supervision['valid'],supervision['unknown']]),['SUPERVISION_VALID','SUPERVISION_UNKNOWN'])
        tif('GT10M_TRAINING_WEIGHT.tif',supervision['weight'][None],['TRAINING_WEIGHT'])
        tif('GT10M_STATE_IGNORE.tif',np.full((1,256,256),255,np.uint8),['STATE_IGNORE_255'])
        tif('GT10M_SUPERVISION_FRACTIONS.tif',supervision['target'],['Stable','Addition','Demolition'])
        back=load_rasters(outpaths)
        if back['grid']!=grid or np.any(supervision['weight']!=0):raise ValueError('Dataloader/qualification mismatch')
        phases['GT10m']={'execution':'EXECUTED','native_fraction':'SHADOW_CANDIDATE','verified_transform_fraction':'ALL_NODATA','qualified_pixels':int(eligibility['eligible'].sum()),'denominator':'FULL_100M2_TARGET_PIXEL','outside_inferred_RGB_support':'NODATA_NOT_BACKGROUND','native_zero':'NO_BACKGROUND_CERTIFICATION','geometry_not_full_instance_truth':True}
        el={'run_role':'SHADOW_CANDIDATE','qualified_role':'QUALIFIED_SUPERVISION','QUALIFIED_10M_PIXEL_COUNT':0,'spatial_transform':'UNKNOWN_NOT_ZERO','time_evidence':'SOURCE_2018-11-01_SINGLE_OBSERVATION_NOT_2019_CHANGE','roof_semantics':'UNCALIBRATED_B0_CANDIDATE','state_evidence':'UNKNOWN','source_supported_pixels':int((fraction['scope_fraction'][0]>0).sum()),'positive_native_fraction_pixels':int((raw>0).sum()),'valid':0,'unknown':65536,'ignore':65536,'training_weight_nonzero':0,'invalid_geometry_excluded_from_area_count':len(invalid),'training':'NOT_RUN','three_state_truth':'NOT_RUN','eligibility_is_not_source_support':True}
        save(output/'S4_12_GT10M_SUPERVISION_ELIGIBILITY.json',el)
        save(output/'S4_12_MODULE_HANDOFF_AUDIT.json',{'contract':CONTRACT,'stages':phases,'fresh_candidates':len(records)+len(invalid),'valid_geometry_candidates':len(records),'invalid_geometry_candidates':len(invalid),'raw_geometries_preserved':True,'fraction_area_conservation':fraction['area_conservation_error_m2'],'roles_not_promoted':True,'GT_layers_read':False,'student_predictions_used':False,'legacy_closed_link_count':53,'invalid_candidates_retained_not_silently_repaired':True})
        save(output/'FIGURE_INPUTS.json',{'packet':packet,'grid':{'affine':list(aff)[:6],'width':256,'height':256,'crs':'EPSG:32649'},'inferred_support':mapping(rgb_support),'valid_candidates':len(records),'invalid_candidates':len(invalid),'accepted_alignment':0,'label_read_paths':[str(p) for p in outpaths]})
        print('TORCH_DATALOADER_READBACK_AND_DATA_FIGURE',flush=True);worker('readback-figure')
        phases['dataloader']={'execution':'EXECUTED','existing_reader':'load_rasters','torch_batch':'EXECUTED_READBACK_ONLY'}
        after=verify_artifacts(m['inputs'])
        if after!=before or sha256(manifest_path)!=manifest_hash:raise ValueError('Frozen input drift')
        summary={'ENGINEERING_PASS':True,'engineering_scope':'LIVE_B0_FIXED_NATIVE_PACKET_TO_SHADOW_FRACTION_WITH_UNRESOLVED_A_AND_FRESH_I',
                 'GT_FREE_INFERENCE_PASS':True,'INSTANCE_ALIGNMENT_VALIDATED':False,'QUALIFIED_10M_PIXEL_COUNT':0,
                 'SCIENTIFIC_VALIDATION_PASS':False,'READY_FOR_CITYWIDE':False,'PRODUCTION_ADMISSION_PASS':False,
                 'run_id':output.name,'actual_date':packet['source_observed_date'],'model':'R0_B0_RETAINED',
                 'native_candidates':len(records)+len(invalid),'valid_geometry_candidates':len(records),'invalid_geometry_candidates':len(invalid),
                 'historical_source_links_preserved':53,'new_historical_links_inherited':0,'physical_ids':0,
                 'applied_displacements':0,'registered_roofs':0,'ground_footprints':0,'ABSOLUTE_POSITION_ACCURACY':'UNKNOWN',
                 'training_updates':0,'cloud_started':False,'protected_tests_read':False,'elapsed_seconds':time.perf_counter()-started}
        report=['# XIAN-R5-S4.12 真实RGB全链路Shadow','',json.dumps(summary,ensure_ascii=False,indent=2),'',
                '复用统一入口，增加LIVE_B0_RGB_SHADOW_V1分支；旧缓存复演分支不重写。一次冻结manifest可指定新的不存在目录复跑，输入、模型、代码哈希不一致则失败。',
                f'固定选择已有{packet["region_id"]}原生窗口，原始RGB {rgb.shape[1]}×{rgb.shape[0]}；未新划ROI。10m使用原有EPSG:32649、256×256格网，未推理范围保持NoData/Unknown。',
                'B0 RGB-only网络实际前向已执行，固定原预处理和概率>0.5，不使用新三类权重或阈值搜索。屋顶为源框架候选，不是地面足迹。',
                'A接口实际执行，但没有合格TARGET_B checkpoint或独立CHECK，数值位移及不确定性为null，严格应用门槛未放行；NO_ACCEPTED_ALIGNMENT，不画伪纠偏箭头。',
                'I重新核验53条历史单期来源关联，仅适用原1662观察。新B0组件只关联源RGB与几何来源，不继承历史building object或physical ID。',
                'Presence引擎实际运行，单期候选全部U，没有变化区间。几何无效的原始候选保留且单独报Unknown，不自动修复或进入面积积分。',
                'GT10m用原精确相交fraction builder；先源支持裁剪，不把零fraction当背景；有验证变换的fraction全NoData；监督valid=0、unknown=1、ignore255、weight=0。真实load_rasters和torch DataLoader均完成回读。',
                '数据图使用academic-plotting的数据绘图流程：原始RGB/概率/候选轮廓/10m面积分数/质量/Unknown，固定窗口含全部失败和未解决对象；无生成式或人工伪造几何。',
                '本轮工程通过仅指候选链及失败关闭路径贯通，不是位移预测有效、身份闭合或科学精度通过。代码按codebase-design最小适配复用A/I/R5/GT10m，未建立第二套数据库或任务系统。',
                '下一步缺口：合格源框架屋顶监督与独立空间CHECK；新观察到既有building object的可追溯派生证据；真实跨期存在/身份；日期兼容与像元训练准入证据。不得从当前候选自动产生正式GT。',
                '训练、云GPU、正式三态GT与正式GT10m均NOT_RUN；B0/S2/S3/S4历史权重、GT、模块A/I冻结资产和历史回执只读。']
        with (output/'FINAL_REPORT.md').open('x',encoding='utf-8') as f:f.write('\n\n'.join(report))
        save(output/'STAGE_EXECUTION_STATUS.json',phases)
        files={str(p.relative_to(output)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()}
        save(output/'FINAL_RECEIPT.json',{**summary,'completed_at':datetime.now().astimezone().isoformat(),'manifest_sha256':manifest_hash,'verified_inputs_sha256':after,'artifacts':files})
        with (output/'FINAL_RECEIPT.sha256').open('x',encoding='ascii') as f:f.write(sha256(output/'FINAL_RECEIPT.json'))
        print(json.dumps(summary),flush=True)
    except Exception:
        import traceback
        save(output/'FAILED_RECEIPT.json',{'ENGINEERING_PASS':False,'stages_completed':phases,'error':traceback.format_exc(),'training':'NOT_RUN'})
        raise
