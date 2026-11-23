"""Minimal sealed B0/V2 -> existing I interchange adapter; no model execution.

Observation, source-tile, graph and link schemas come from integration.py.
Research translations never enter registered_geometry or identity evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import traceback

import fiona
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import box, mapping, shape
from shapely.ops import transform, unary_union

from .integration import (CONTRACT, adapt_native_components, attach_single_date_links,
                          read_json, run_r5_state, sha256, verify_artifacts, verify_source_dates)
from .fraction_shadow import _areas_and_counts
from .label_transfer import Grid, fraction_supervision, load_rasters

MODE = 'CACHED_B0_V2_CANDIDATE_INTEGRATION_V1'
VERSION_B0 = 'B0_OBSERVATIONS_V1'
VERSION_RAW = 'V2_RAW_NUMERIC_V1'
VERSION_FULL = 'V2_COMPLETE_INSTANCE_V1'
PRODUCTS = {
    'native': 'B0_NATIVE_INSTANCES.json', 'b0_receipt': 'B0_RUN_RECEIPT.json',
    'area': 'INSTANCE_AREA_AND_SOURCE_LEDGER.json', 'predictions': 'V2_PARENT_PREDICTIONS.json',
    'quality': 'A_MATCH_QUALITY_LEDGER.json', 'governance': 'A_POSITION_GOVERNANCE_LEDGER.json',
    'raw': 'ROOF_CANDIDATES_EPSG32649.geojson',
    'complete': 'COMPLETE_INSTANCE_CANDIDATES_EPSG32649.geojson',
    'geo_receipt': 'GEOSPATIAL_RECEIPT.json', 'complete_receipt': 'COMPLETE_INSTANCE_RECEIPT.json',
    'gt_free_seal': 'GT_FREE_SEAL.json', 'complete_seal': 'GT_FREE_COMPLETE_INSTANCE_SEAL.json',
}
RASTERS = ['B0_ROOF_NATIVE.tif', 'V2_MIXED_AVAILABLE_ROOF_NATIVE.tif',
           *['FRACTION10M_' + s + '.tif' for s in (
               'B0', 'RETAIN_ORIGINAL', 'V2_MIXED_AVAILABLE', 'COMPLETE_INSTANCE_V2',
               'SOURCE_SUPPORT', 'UNKNOWN_POSITION', 'GEOMETRY_CONFLICT', 'COMPLETE_INSTANCE_CONFLICT')],
           'FORMAL_SUPERVISION_STATE10M.tif', 'FORMAL_TRAINING_WEIGHT10M.tif']


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def geometry_hash(g):
    return hashlib.sha256(g.wkb).hexdigest()


def publish_receipt(output, summary):
    """Publish SHA first and the complete success receipt last, on one volume."""
    output=Path(output)
    final=output/'FINAL_RECEIPT.json'; sidecar=output/'FINAL_RECEIPT.sha256'
    if final.exists() or sidecar.exists():raise FileExistsError('Receipt already published')
    pending=output/'.FINAL_RECEIPT.json.pending'; pending_sha=output/'.FINAL_RECEIPT.sha256.pending'
    with pending.open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    with pending_sha.open('x',encoding='ascii') as stream:
        stream.write(sha256(pending)+'  FINAL_RECEIPT.json\n');stream.flush();os.fsync(stream.fileno())
    # Windows rename refuses existing destinations. A sidecar alone is not success.
    os.rename(pending_sha,sidecar)
    os.rename(pending,final)  # Commit point; no fallible filesystem operation follows.


def unique(rows, key='candidate_id'):
    result = {}
    for row in rows:
        uid = row[key]
        if not isinstance(uid, str) or not uid or uid in result:
            raise ValueError('Duplicate/missing original observation ID: ' + str(uid))
        result[uid] = row
    return result


def polygon(value):
    g = shape(value)
    if (g.geom_type not in {'Polygon', 'MultiPolygon'} or g.is_empty or not g.is_valid
            or not np.isfinite(g.area) or g.area <= 0):
        raise ValueError('Invalid candidate geometry; automatic repair forbidden')
    return g


def project_native(affine):
    tr = Transformer.from_crs(4326, 32649, always_xy=True)
    def project(g):
        return transform(tr.transform, transform(
            lambda x, y, z=None: (affine.a*np.asarray(x)+affine.b*np.asarray(y)+affine.c,
                                  affine.d*np.asarray(x)+affine.e*np.asarray(y)+affine.f), g))
    return project


def source_tiles(packet, registry):
    """Use real native mosaic windows, not old ROI-clipped metric support."""
    tiles = list(unique(registry['tiles'], 'download_tile_id').values())
    if packet['source_raster_crs'] != 'EPSG:4326' or packet['native_resampling'] != 'NONE':
        raise ValueError('Unsupported native source frame')
    x, y, w, h = packet['source_window']; parent = box(x, y, x+w, y+h)
    project = project_native(Affine(*packet['source_raster_affine']))
    selected = []
    for tile in tiles:
        tx, ty, tw, th = tile['mosaic_window']; native = box(tx, ty, tx+tw, ty+th)
        if native.intersection(parent).area <= 0:
            continue
        if tile['actual_date'] != packet['source_observed_date']:
            raise ValueError('Parent/source-tile date conflict')
        # New interface view only. Keep the frozen old clipped support in provenance.
        row = dict(tile, upstream_clipped_support_geometry=tile['support_geometry'],
                   support_geometry=mapping(project(native)),
                   support_role='ACTUAL_DOWNLOAD_TILE_NATIVE_WINDOW_NOT_CAMERA_CERTIFICATION')
        selected.append(row)
    selected.sort(key=lambda t: t['download_tile_id'])
    if not selected:
        raise ValueError('Missing source tile coverage')
    coverage = unary_union([box(t['mosaic_window'][0], t['mosaic_window'][1],
                                t['mosaic_window'][0]+t['mosaic_window'][2],
                                t['mosaic_window'][1]+t['mosaic_window'][3]) for t in selected])
    if parent.difference(coverage).area > 0:
        raise ValueError('Missing native parent source coverage')
    return selected


def freeze_candidate_manifest(path, products, source_manifest):
    """Explicit historical product root and existing I manifest; never find latest."""
    path, products, source_manifest = map(lambda p: Path(p).resolve(), (path, products, source_manifest))
    if path.exists():
        raise FileExistsError('Append-only manifest exists')
    old = read_json(source_manifest); pins = {}
    for name in ('source_registry', 'actual_date_registry', 'b_packets', 'i_graph', 'i_links', 'i_receipt', 'components', 'grid_template'):
        pins[name] = old['inputs'][name]
    verify_artifacts(pins)
    def pin(key, file, role, expected=None):
        actual = sha256(file)
        if expected is not None and actual != expected:
            raise ValueError('Sealed product SHA mismatch: ' + key)
        pins[key] = dict(path=str(Path(file).resolve()), sha256=actual, role=role)
    pin('upstream_source_manifest', source_manifest, 'WINDOW_CONFIG')
    manifest_file = products/'DELIVERY_SHA256_MANIFEST_FINAL.json'
    expected = read_json(manifest_file)['files']
    pin('product_manifest', manifest_file, 'OBSERVED_CANDIDATE')
    for key, name in PRODUCTS.items():
        pin(key, products/name, 'OBSERVED_CANDIDATE', expected[name])
    for name in RASTERS:
        pin(name, products/name, 'OBSERVED_CANDIDATE', expected[name])
    receipt = read_json(products/PRODUCTS['b0_receipt'])
    pin('source_rgb', receipt['source'], 'MODULE_B_NATIVE_RGB', receipt['source_sha256'])
    packets = read_json(pins['b_packets']['path'])['records']
    matches = [p for p in packets if p['image_sha256'] == receipt['source_sha256']]
    if len(matches) != 1:
        raise ValueError('Unique frozen parent/source packet required')
    packet = matches[0]
    tiles = source_tiles(packet, read_json(pins['source_registry']['path']))
    for tile in tiles:
        pin('tile_'+tile['download_tile_id'], tile['source_JPG'], 'SOURCE_COVERAGE', tile['source_JPG_sha256'])
    repo = Path(__file__).resolve().parents[2]
    for file in sorted((repo/'src/xian_r5').rglob('*.py')):
        pin('code_'+file.relative_to(repo).as_posix(), file, 'CODE')
    for name in ['tools/run_xian_r5_integration.py', 'docs/XIAN_R5_CACHED_ROOF_PRODUCTS.md']:
        pin('code_'+name, repo/name, 'CODE')
    raw = read_json(products/PRODUCTS['raw'])['features']
    complete = read_json(products/PRODUCTS['complete'])['features']
    save(path, dict(contract_version=CONTRACT, execution_mode=MODE,
                    parent_ids=[packet['region_id']], source_tile_ids=[t['download_tile_id'] for t in tiles],
                    products_root=str(products), formal_position='UNRESOLVED',
                    manual_gt_required_for_inference=False, max_seconds=600,
                    expected=dict(observations=len(read_json(products/PRODUCTS['native'])['ROOF_INSTANCE_CANDIDATES']),
                                  raw_numeric=sum(f['properties']['object_state']=='V2_SHIFT_CANDIDATE' for f in raw),
                                  complete_numeric=sum(f['properties']['object_state']=='V2_SHIFT_CANDIDATE' for f in complete)),
                    inputs=pins))
    return len(pins)


def decode_products(documents, parent, native_affine):
    """Reject duplicates, mismatched sources, implicit zero vectors and promotion."""
    native = unique(documents['native']['ROOF_INSTANCE_CANDIDATES'])
    quality = unique(documents['quality']['records']); governance = unique(documents['governance']['records'])
    area = unique(documents['area']['records']); predictions = unique(documents['predictions']['records'])
    def features(document, allowed):
        result = {s: {} for s in allowed}
        if document['crs']['properties']['name'] != 'EPSG:32649':
            raise ValueError('Original metric candidate CRS required')
        for f in document['features']:
            p = f['properties']; state = p['object_state']; uid = p['candidate_id']
            if state not in result or uid in result[state]:
                raise ValueError('Duplicate candidate state or unsupported state')
            if p['formal_position_selection'] != 'UNRESOLVED' or p['formal_training_weight'] != 0:
                raise ValueError('Unqualified formal candidate state')
            if p.get('physical_identity_certified', False) is not False:
                raise ValueError('Candidate cannot certify physical identity')
            if (any(p.get(k) is not None for k in ('physical_id','registered_geometry','ground_footprint','applied_vector_m'))
                    or p.get('GT_USED_FOR_GEOMETRY',False) is not False):
                raise ValueError('Unqualified formal candidate fields')
            result[state][uid] = f
        return result
    raw = features(documents['raw'], ['NO_SHIFT_CANDIDATE', 'V2_SHIFT_CANDIDATE'])
    full = features(documents['complete'], ['NO_SHIFT_CANDIDATE', 'V2_SHIFT_CANDIDATE', 'UNRESOLVED'])
    ids = set(native)
    if (not ids or any(set(r) != ids for r in [quality, governance, area, raw['NO_SHIFT_CANDIDATE'],
                                               full['NO_SHIFT_CANDIDATE'], full['UNRESOLVED']])):
        raise ValueError('Missing original object, provenance or unresolved reference')
    if (set(raw['V2_SHIFT_CANDIDATE']) != {k for k, r in predictions.items() if r['shift_model_xy'] is not None}
            or not set(predictions) <= ids):
        raise ValueError('Numeric prediction/geometry binding differs')
    expected_full = {k for k in raw['V2_SHIFT_CANDIDATE'] if quality[k]['complete_instance_context']
                     and not quality[k]['source_edge_partial_risk']}
    if set(full['V2_SHIFT_CANDIDATE']) != expected_full:
        raise ValueError('Frozen complete-instance candidate version differs')
    project = project_native(native_affine); base = {}; shifted = {}; lineage = []
    tr = Transformer.from_crs(4326, 32649, always_xy=True)
    for uid in sorted(ids):
        n = native[uid]; q = quality[uid]; a = area[uid]; p = raw['NO_SHIFT_CANDIDATE'][uid]['properties']
        if (n['original_source_hash'] != parent['image_sha256'] or p['source_sha256'] != parent['image_sha256']
                or n['source_image_id'] != parent['region_id'] or p['source_image_id'] != parent['region_id']
                or n['source_actual_date'] != parent['source_observed_date']
                or p['source_actual_date'] != parent['source_observed_date']
                or q['source_sha256'] != parent['image_sha256'] or q['source_date'] != parent['source_observed_date']):
            raise ValueError('Missing/conflicting original source: ' + uid)
        for ledger in (q, governance[uid]):
            if (ledger['formal_position_selection'] != 'UNRESOLVED' or ledger['formal_training_weight'] != 0
                    or ledger['not_selected_V2_is_true_zero'] is not False):
                raise ValueError('Position or supervision promoted')
        ng = polygon(n['geometry_pixel_edge_xy']); g = polygon(raw['NO_SHIFT_CANDIDATE'][uid]['geometry'])
        pg = project(ng)
        height,width=parent['local_image_shape']
        if not box(0,0,width,height).covers(ng):
            raise ValueError('Original geometry outside native parent support')
        if (not pg.is_valid or pg.wkb != g.wkb or polygon(full['NO_SHIFT_CANDIDATE'][uid]['geometry']).wkb != g.wkb
                or abs(ng.area-n['pixel_area']) > 1e-6 or abs(g.area-a['B0_area_m2'])>1e-6):
            raise ValueError('Original geometry/hash/frame differs: ' + uid)
        if (full['UNRESOLVED'][uid]['geometry'] is not None
                or full['UNRESOLVED'][uid]['properties']['candidate_geometry_reference'] != uid):
            raise ValueError('Unresolved must reference original, not zero displacement')
        base[uid] = g; vector = a['shift_east_north_m']; vg = None
        if uid in raw['V2_SHIFT_CANDIDATE']:
            vg = polygon(raw['V2_SHIFT_CANDIDATE'][uid]['geometry'])
            d = predictions[uid]['shift_model_xy']; c = ng.centroid
            xy0 = tr.transform(*(native_affine*(c.x,c.y))); xy1 = tr.transform(*(native_affine*(c.x+2*d[0],c.y+2*d[1])))
            if (not np.array_equal(np.subtract(xy1,xy0), vector)
                    or predictions[uid]['source_crop_xy'] != q['origin_xy']
                    or vg.wkb != translate(g,*vector).wkb or abs(g.area-vg.area) > 1e-6
                    or abs(vg.area-a['V2_area_m2'])>1e-6
                    or raw['V2_SHIFT_CANDIDATE'][uid]['properties']['translation_east_north_m'] != vector):
                raise ValueError('Displacement provenance or pure translation differs: ' + uid)
            shifted[uid] = vg
        elif vector is not None or a['V2_area_m2'] is not None:
            raise ValueError('Missing prediction cannot become zero shift')
        if uid in expected_full and polygon(full['V2_SHIFT_CANDIDATE'][uid]['geometry']).wkb != vg.wkb:
            raise ValueError('Complete version changed displacement')
        lineage.append(dict(candidate_id=uid, geometry_sha256=geometry_hash(g), native_geometry_sha256=geometry_hash(ng),
                            source_sha256=parent['image_sha256'], source_actual_date=parent['source_observed_date'],
                            V2_geometry_sha256=None if vg is None else geometry_hash(vg),
                            V2_vector_m=vector, V2_checkpoint_sha256=documents['quality']['checkpoint_sha256'],
                            V2_input_origin_xy=q['origin_xy'], source_edge_partial_risk=q['source_edge_partial_risk'],
                            complete_instance_context=q['complete_instance_context'],
                            V2_raw_available=vg is not None, V2_complete_available=uid in expected_full,
                            formal_position_selection='UNRESOLVED', physical_id=None, formal_training_weight=0))
    return native, base, shifted, expected_full, lineage


def validate_fraction_bundle(paths, base, shifted, full):
    metric = [n for n in RASTERS if not n.endswith('ROOF_NATIVE.tif')]
    loaded = load_rasters([paths[n] for n in metric]); grid = loaded['grid']
    if grid.crs != 'EPSG:32649' or tuple(grid.affine)[:2] != (10.,0.) or grid.affine.d != 0 or grid.affine.e != -10:
        raise ValueError('Original exact 10m metric grid required')
    arrays = {n:a[0] for n,a in zip(metric, loaded['arrays'])}
    state=arrays['FORMAL_SUPERVISION_STATE10M.tif']; weight=arrays['FORMAL_TRAINING_WEIGHT10M.tif']
    with rasterio.open(paths['FORMAL_SUPERVISION_STATE10M.tif']) as src:
        if src.nodata != 255 or not np.all(src.read_masks(1)==0):
            raise ValueError('Formal unknown mask changed')
    if not np.all(state==255) or not np.all(weight==0):
        raise ValueError('Unqualified formal supervision input')
    for n,a in arrays.items():
        if n.startswith('FRACTION') and (not np.isfinite(a).all() or (a<0).any() or (a>1).any()):
            raise ValueError('Invalid cached fraction: ' + n)
    if not np.array_equal(arrays['FRACTION10M_SOURCE_SUPPORT.tif'], arrays['FRACTION10M_UNKNOWN_POSITION.tif']):
        raise ValueError('Supported-position Unknown was changed')
    cells=np.array([grid.cell(c,r) for r in range(grid.height) for c in range(grid.width)],dtype=object)
    variants={VERSION_B0:base, VERSION_RAW:{k:shifted.get(k,g) for k,g in base.items()},
              VERSION_FULL:{k:shifted[k] if k in full else g for k,g in base.items()}}
    names={VERSION_B0:'B0',VERSION_RAW:'V2_MIXED_AVAILABLE',VERSION_FULL:'COMPLETE_INSTANCE_V2'};errors={}
    for version, objects in variants.items():
        areas,_,_=_areas_and_counts(objects.values(),cells)
        actual=arrays['FRACTION10M_'+names[version]+'.tif']
        difference=float(np.max(np.abs(areas.reshape(actual.shape)/100-actual)))
        mass=abs(float(actual.astype(float).sum()*100)-unary_union(list(objects.values())).area)
        if difference>1e-6 or mass>.01:
            raise ValueError('Exact full-union cached fraction differs: ' + version)
        errors[version]=dict(max_fraction_difference=difference,float32_mass_error_m2=mass)
    if not np.array_equal(arrays['FRACTION10M_RETAIN_ORIGINAL.tif'],arrays['FRACTION10M_B0.tif']):
        raise ValueError('Retain-original candidate drift')
    zero=np.zeros((grid.height,grid.width)); formal=fraction_supervision(
        np.zeros((3,*zero.shape)),valid=zero,unknown=np.ones_like(zero),eligible=np.zeros_like(zero,dtype=bool))
    if np.any(formal['weight']):raise ValueError('Research fraction became supervision')
    return grid,errors


def write_gpkg(path, base, native, lineage, tiles, parent, shifted, full):
    by_id={r['candidate_id']:r for r in lineage}; temp=path.with_name('.partial_'+path.name)
    component_schema=dict(geometry='Unknown',properties=dict(component_key='str',native_sha256='str',source_actual_date='str'))
    with fiona.open(temp,'w',driver='GPKG',layer='native_positive_components',crs='EPSG:32649',schema=component_schema) as dst:
        for uid,g in sorted(base.items()):
            dst.write(dict(geometry=mapping(g),properties=dict(component_key=uid,native_sha256=by_id[uid]['native_geometry_sha256'],source_actual_date=parent['source_observed_date'])))
    graph_schema=dict(geometry='Unknown',properties=dict(node_id='str',original_id='str',physical_id='str'))
    with fiona.open(temp,'w',driver='GPKG',layer='roof_instance_reference',crs='EPSG:32649',schema=graph_schema) as dst:
        for uid,g in sorted(base.items()):dst.write(dict(geometry=mapping(g),properties=dict(node_id=uid,original_id=uid,physical_id=None)))
    # There is no existing I proof for these observation IDs. Do not copy old object identity.
    with fiona.open(temp,'w',driver='GPKG',layer='building_object_reference',crs='EPSG:32649',schema=graph_schema):pass
    schema=dict(geometry='Unknown',properties=dict(product_version='str',candidate_id='str',object_state='str',geometry_sha256='str',dx_m='float',dy_m='float',formal_position='str',formal_weight='float'))
    with fiona.open(temp,'w',driver='GPKG',layer='roof_position_candidates',crs='EPSG:32649',schema=schema) as dst:
        for version in (VERSION_B0,VERSION_RAW,VERSION_FULL):
            for uid,g in sorted(base.items()):
                states=[('NO_SHIFT_CANDIDATE',g,None),('UNRESOLVED',None,None)]
                if (version==VERSION_RAW and uid in shifted) or (version==VERSION_FULL and uid in full):
                    states.append(('V2_SHIFT_CANDIDATE',shifted[uid],by_id[uid]['V2_vector_m']))
                for state,geometry,vector in states:
                    dst.write(dict(geometry=None if geometry is None else mapping(geometry),properties=dict(product_version=version,candidate_id=uid,object_state=state,geometry_sha256=None if geometry is None else geometry_hash(geometry),dx_m=None if vector is None else float(vector[0]),dy_m=None if vector is None else float(vector[1]),formal_position='UNRESOLVED',formal_weight=0.0)))
    if path.exists():raise FileExistsError('Append-only GPKG exists')
    os.replace(temp,path)


def read_candidate_view(path, base, shifted, full, lineage):
    """Deterministic GPKG content acceptance, excluding container timestamps."""
    expected={}
    vectors={r['candidate_id']:r['V2_vector_m'] for r in lineage}
    for version in (VERSION_B0,VERSION_RAW,VERSION_FULL):
        for uid,g in base.items():
            expected[(version,uid,'NO_SHIFT_CANDIDATE')]=(g,None)
            expected[(version,uid,'UNRESOLVED')]=(None,None)
            if (version==VERSION_RAW and uid in shifted) or (version==VERSION_FULL and uid in full):
                expected[(version,uid,'V2_SHIFT_CANDIDATE')]=(shifted[uid],vectors[uid])
    result={}
    with fiona.open(path,layer='roof_position_candidates') as src:
        if src.crs.to_epsg()!=32649:raise ValueError('Candidate GPKG CRS drift')
        for feature in src:
            p=dict(feature.properties);key=(p['product_version'],p['candidate_id'],p['object_state'])
            if key not in expected or key in result:raise ValueError('Duplicate/unexpected candidate GPKG state')
            geometry,vector=expected[key]
            if geometry is None:
                if feature.geometry is not None or p['geometry_sha256'] is not None:raise ValueError('Unresolved GPKG geometry promoted')
            elif polygon(feature.geometry).wkb!=geometry.wkb or p['geometry_sha256']!=geometry_hash(geometry):
                raise ValueError('Candidate GPKG geometry/hash changed')
            if ([p['dx_m'],p['dy_m']]!=([None,None] if vector is None else vector)
                    or p['formal_position']!='UNRESOLVED' or p['formal_weight']!=0):
                raise ValueError('Candidate GPKG transform/admission changed')
            result[key]=p
    if set(result)!=set(expected):raise ValueError('Incomplete candidate GPKG export')
    return [result[key] for key in sorted(result)]


def replay_candidate_products(manifest_path, output):
    started=time.monotonic(); manifest_path=Path(manifest_path).resolve();output=Path(output).resolve()
    manifest=read_json(manifest_path); manifest_hash=sha256(manifest_path)
    limit=manifest.get('max_seconds')
    if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not np.isfinite(limit) or not 0<limit<=600:
        raise ValueError('Finite replay limit must be in (0,600] seconds')
    if (manifest['contract_version']!=CONTRACT or manifest['execution_mode']!=MODE
            or manifest['manual_gt_required_for_inference'] is not False or len(manifest['parent_ids'])!=1
            or manifest['formal_position']!='UNRESOLVED'):
        raise ValueError('Only explicit single-parent unresolved cached product handoff supported')
    if output.exists():raise FileExistsError('Append-only output must be new')
    pins=manifest['inputs']
    if any(output.is_relative_to(Path(p['path']).resolve().parent) or Path(p['path']).resolve().is_relative_to(output) for p in pins.values()):
        raise ValueError('Output overlaps historical input directory')
    before=verify_artifacts(pins)
    def checkpoint():
        if time.monotonic()-started>min(float(manifest['max_seconds']),600):raise TimeoutError('Finite replay limit')
    def doc(key):return read_json(pins[key]['path'])
    try:
        seal=doc('product_manifest')['files']
        for key in [*PRODUCTS,*RASTERS]:
            name=PRODUCTS.get(key,key)
            if pins[key]['sha256']!=seal[name]:raise ValueError('Historical product pin differs: '+key)
        receipt=doc('b0_receipt'); packets=doc('b_packets')['records']
        packets=[p for p in packets if p['region_id']==manifest['parent_ids'][0] and p['image_sha256']==pins['source_rgb']['sha256']]
        if len(packets)!=1 or receipt['source_sha256']!=pins['source_rgb']['sha256']:raise ValueError('Parent/source SHA binding missing')
        parent=packets[0];tiles=source_tiles(parent,doc('source_registry'))
        if [t['download_tile_id'] for t in tiles]!=manifest['source_tile_ids']:raise ValueError('Source tile set drift')
        for tile in tiles:
            key='tile_'+tile['download_tile_id']
            if pins[key]['path']!=str(Path(tile['source_JPG']).resolve()) or pins[key]['sha256']!=tile['source_JPG_sha256']:raise ValueError('Missing source tile SHA pin')
        date_rows=verify_source_dates(tiles,Path(pins['actual_date_registry']['path']))
        x,y,w,h=parent['source_window'];aff=Affine(*parent['source_raster_affine'])*Affine.translation(x,y)
        with rasterio.open(pins['B0_ROOF_NATIVE.tif']['path']) as src:
            if src.transform!=aff or list(src.shape)!=parent['local_image_shape'] or str(src.crs)!='EPSG:4326':raise ValueError('Parent native raster/frame drift')
        documents={k:doc(k) for k in PRODUCTS}
        native,base,shifted,full,lineage=decode_products(documents,parent,aff)
        expected=manifest['expected']
        if (len(base),len(shifted),len(full))!=(expected['observations'],expected['raw_numeric'],expected['complete_numeric']):raise ValueError('Frozen product count differs')
        # Old I links are retained as provenance only: no overlap or spatial identity guessing.
        with Path(pins['i_links']['path']).open(encoding='utf-8-sig',newline='') as stream:upstream_links=list(csv.DictReader(stream))
        if any(r['observation_id'] in base for r in upstream_links):raise ValueError('Current-ID I links need explicit source adjudication; not silently replaced')
        with sqlite3.connect(Path(pins['i_graph']['path']).as_uri()+'?mode=ro',uri=True) as db:
            old_ids={r[0] for r in db.execute('SELECT node_id FROM roof_instance_reference')}
        if old_ids.intersection(base):raise ValueError('Existing current-ID graph needs explicit reconciliation')
        paths={n:Path(pins[n]['path']) for n in RASTERS};grid,errors=validate_fraction_bundle(paths,base,shifted,full);checkpoint()
        output.mkdir(parents=True);save(output/'FROZEN_INTEGRATION_MANIFEST.json',manifest)
        save(output/'SOURCE_COMPONENT_LEDGER.json',dict(tiles=tiles,tile_count=len(tiles),
             basis='PINNED_NATIVE_MOSAIC_WINDOWS; upstream ROI-clipped supports retained as provenance',
             upstream_registry_sha256=pins['source_registry']['sha256'],true_camera_source_components_recovered=False))
        graph=output/'CANDIDATE_I_HANDOFF.gpkg';write_gpkg(graph,base,native,lineage,tiles,parent,shifted,full)
        records=adapt_native_components(graph,tiles);by_id={r['observation_id']:r for r in records}
        source_rows=[]
        for item in lineage:
            uid=item['candidate_id'];ng=polygon(native[uid]['geometry_pixel_edge_xy']);global_geometry=translate(ng,x,y)
            hits=[]
            for tile in tiles:
                tx,ty,tw,th=tile['mosaic_window']
                if global_geometry.intersection(box(tx,ty,tx+tw,ty+th)).area>0:hits.append(tile)
            if not hits:raise ValueError('Missing exact native object/source binding: '+uid)
            row=by_id[uid];old_metric_hits=row['source_tile_ids']
            # Exact pixel provenance is authoritative; shifted geometry is never used for source lookup.
            row.update(source_tile_ids=[t['download_tile_id'] for t in hits],source_hashes=[t['source_JPG_sha256'] for t in hits],source_paths=[t['source_JPG'] for t in hits],
                       candidate_state='MODEL_PROPOSED',formal_position_selection='UNRESOLVED',formal_training_weight=0,
                       original_product_path=pins['raw']['path'],original_native_product_path=pins['native']['path'],
                       source_binding_basis='EXACT_ORIGINAL_NATIVE_PIXEL_WINDOW',candidate_products=item)
            if row['geometry_sha256']!=item['geometry_sha256'] or row['native_geometry_sha256']!=item['native_geometry_sha256']:raise ValueError('GPKG geometry roundtrip changed hash')
            source_rows.append(dict(observation_id=uid,source_tile_ids=row['source_tile_ids'],source_hashes=row['source_hashes'],source_paths=row['source_paths'],actual_date=row['actual_date'],metric_support_hit_ids=old_metric_hits,source_binding_basis=row['source_binding_basis'],native_source_coverage_complete=True))
        links=[dict(observation_id=uid,roof_obs_id=uid,building_object_id='',physical_id='',historical_candidate_track_id='',source_actual_date=parent['source_observed_date'],link_level='CANDIDATE_ONLY',relationship_type='INSUFFICIENT_EVIDENCE',source_evidence='NO_EXISTING_CURRENT_ID_SOURCE_OBJECT_PROOF',GT_USED_FOR_GEOMETRY='False') for uid in sorted(base)]
        ledger=output/'SOURCE_OBJECT_LINK_LEDGER.csv'
        with ledger.open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(links[0]));writer.writeheader();writer.writerows(links)
        links,closed=attach_single_date_links(records,ledger,graph,[])
        with sqlite3.connect(graph) as db:
            db.execute('CREATE TABLE source_object_links ('+','.join('"'+k+'" TEXT' for k in links[0])+')')
            db.executemany('INSERT INTO source_object_links VALUES ('+','.join('?' for _ in links[0])+')',[[str(v) if v is not None else None for v in r.values()] for r in links])
            db.execute('CREATE UNIQUE INDEX links_by_observation ON source_object_links(observation_id)')
            db.execute('CREATE UNIQUE INDEX candidate_version_state ON roof_position_candidates(product_version,candidate_id,object_state)')
        state,_=run_r5_state(records,dict(ACCEPTED_LOCAL_REGIONS=0,APPLIED_SHADOW_LOCAL_REGIONS=0),[pins['grid_template']['path']])
        if closed or state.transitions or any(r['presence']!='U' or r['registered_geometry'] is not None or r['applied_vector_m'] is not None or r['physical_id'] is not None for r in records):raise ValueError('Unresolved candidate promoted')
        for name,rows in [('UNIFIED_ROOF_OBSERVATIONS.jsonl',records),('OBJECT_LINKS.jsonl',links)]:
            with (output/name).open('x',encoding='utf-8') as stream:
                for row in rows:stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        save(output/'SOURCE_BINDINGS.json',dict(records=source_rows,original_source_sha256=parent['image_sha256']))
        save(output/'CANDIDATE_PRODUCT_VERSIONS.json',dict(versions={VERSION_B0:dict(observations=len(base),numeric_shifts=0),VERSION_RAW:dict(observations=len(base),numeric_shifts=len(shifted)),VERSION_FULL:dict(observations=len(base),numeric_shifts=len(full),retain_original=len(base)-len(full))},records=lineage,
            native_raster_binding={'B0_ROOF_NATIVE.tif':VERSION_B0,'V2_MIXED_AVAILABLE_ROOF_NATIVE.tif':VERSION_RAW},complete_version_native_TIF='NOT_PRESENT_NOT_GENERATED',formal_choice='UNRESOLVED'))
        save(output/'R5_STATE_HANDOFF.json',dict(contract_version=CONTRACT,actual_date_order=state.actual_date_order,presence=state.presence,transitions=state.transitions,temporal_interval_available=False))
        bundle=output/'PRODUCTS';bundle.mkdir();copied={}
        for key in [*PRODUCTS,*RASTERS]:
            name=PRODUCTS.get(key,key);dest=bundle/name;tmp=bundle/(name+'.partial')
            shutil.copyfile(pins[key]['path'],tmp);os.replace(tmp,dest)
            if sha256(dest)!=pins[key]['sha256']:raise ValueError('Incomplete byte-preserving product copy')
            copied[name]=pins[key]['sha256']
        shutil.copyfile(pins['i_links']['path'],output/'UPSTREAM_I_LINK_LEDGER_UNAPPLIED.csv')
        # Compare semantic contents, not SQLite timestamps or output-directory strings.
        canonical=[]
        for row in records:
            normalized=dict(row,source_path='$OUTPUT/CANDIDATE_I_HANDOFF.gpkg');canonical.append(normalized)
        candidate_view=read_candidate_view(graph,base,shifted,full,lineage)
        fingerprint=dict(unified_observations=digest_value(canonical),object_links=digest_value(links),source_bindings=digest_value(source_rows),candidate_lineage=digest_value(lineage),candidate_gpkg_view=digest_value(candidate_view),copied_product_sha256=copied)
        save(output/'CONTENT_FINGERPRINT.json',fingerprint);checkpoint()
        after=verify_artifacts(pins)
        if after!=before or sha256(manifest_path)!=manifest_hash:raise ValueError('Frozen input/code drift during replay')
        summary=dict(contract_version=CONTRACT,execution_mode=MODE,ENGINEERING_PASS=True,
                     scope='ONE_SEALED_PARENT_CACHED_B0_V2_TO_EXISTING_I_ZERO_ADMISSION',
                     parent_ids=manifest['parent_ids'],observations=len(base),V2_raw_numeric=len(shifted),V2_complete_numeric=len(full),
                     original_ids_and_geometry_hashes_preserved=True,source_tiles=len(tiles),date_registry_rows_checked=date_rows,
                     original_date_log_recheck='NOT_RUN_REGISTRY_REFERENCES_ONLY_NO_LOG_SUBSTITUTION',
                     missing_original_log_paths=sorted({t['date_evidence_source_log'] for t in tiles if not Path(t['date_evidence_source_log']).is_file()}),
                     upstream_I_observations=len(old_ids),
                     upstream_I_link_rows_preserved=len(upstream_links),inherited_upstream_I_links=0,source_object_links_closed=closed,
                     current_unresolved_link_rows=len(links),formal_position='UNRESOLVED',physical_ids=0,registered_geometry=0,
                     training_eligible_pixels=0,GT10m_training_weight=0,production='NO',SCIENTIFIC_VALIDATION_PASS=False,
                     cross_parent_batch='UNVERIFIED_NO_SECOND_QUALIFIED_PARENT',model_forwards=0,optimizer_updates=0,
                     GT_reads=0,cloud_operations=0,fraction_grid=dict(crs=grid.crs,affine=list(grid.affine)[:6],width=grid.width,height=grid.height),
                     fraction_readback=errors,copied_raster_bytes_unchanged=True,elapsed_seconds=time.monotonic()-started,
                     input_manifest_sha256=manifest_hash,verified_input_sha256=after,
                     output_sha256={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
        publish_receipt(output,summary)
        return summary
    except Exception as error:
        if output.exists():save(output/'FAILED_RECEIPT.json',dict(ENGINEERING_PASS=False,error=repr(error),traceback=traceback.format_exc(),inputs=before))
        raise
