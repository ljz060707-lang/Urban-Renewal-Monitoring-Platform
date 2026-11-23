"""Meaningful fail-closed tests for the cached B0/V2 I seam; synthetic data only."""
import copy
import tempfile
import unittest
import sqlite3
from unittest.mock import patch
from pathlib import Path

import numpy as np
import rasterio
import fiona
from affine import Affine
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import box,mapping,shape

from xian_r5.cached_roof_products import (decode_products,source_tiles,project_native,
    validate_fraction_bundle,unique,geometry_hash,write_gpkg,read_candidate_view,publish_receipt)
from xian_r5.integration import adapt_native_components,verify_artifacts


def documents():
    aff=Affine(2.682209014888908e-6,0,108.92,0,-2.6822090148921567e-6,34.27)
    parent=dict(region_id='fixture_parent',image_sha256='a'*64,source_observed_date='2018-11-01',
                source_window=[0,0,100,100],local_image_shape=[100,100],source_raster_affine=list(aff)[:6],
                source_raster_crs='EPSG:4326',native_resampling='NONE')
    source=[];quality=[];gov=[];area=[];pred=[];raw=[];full=[];project=project_native(aff)
    tr=Transformer.from_crs(4326,32649,always_xy=True)
    for i in range(3):
        uid='original_'+str(i);ng=box(10+20*i,10,20+20*i,20);g=project(ng)
        source.append(dict(candidate_id=uid,geometry_pixel_edge_xy=mapping(ng),pixel_area=ng.area,
            original_source_hash='a'*64,source_image_id='fixture_parent',source_actual_date='2018-11-01'))
        q=dict(candidate_id=uid,source_sha256='a'*64,source_date='2018-11-01',formal_position_selection='UNRESOLVED',
               formal_training_weight=0,not_selected_V2_is_true_zero=False,origin_xy=[0,0],
               complete_instance_context=True,source_edge_partial_risk=i==1)
        quality.append(q);gov.append(dict(q))
        props=dict(candidate_id=uid,source_sha256='a'*64,source_image_id='fixture_parent',source_actual_date='2018-11-01',
                   object_state='NO_SHIFT_CANDIDATE',formal_position_selection='UNRESOLVED',formal_training_weight=0)
        f=dict(type='Feature',properties=props,geometry=mapping(g));raw.append(f);full.append(copy.deepcopy(f))
        if i<2:
            d=[1.,2.];c=ng.centroid;p0=tr.transform(*(aff*(c.x,c.y)));p1=tr.transform(*(aff*(c.x+2*d[0],c.y+2*d[1])))
            vector=list(np.subtract(p1,p0));vg=translate(g,*vector)
            sf=dict(type='Feature',properties=dict(props,object_state='V2_SHIFT_CANDIDATE',translation_east_north_m=vector),geometry=mapping(vg))
            raw.append(sf)
            if i==0:full.append(copy.deepcopy(sf))
        else:d=vector=vg=None
        pred.append(dict(candidate_id=uid,shift_model_xy=d,source_crop_xy=[0,0]))
        area.append(dict(candidate_id=uid,shift_east_north_m=vector,B0_area_m2=g.area,V2_area_m2=None if vg is None else vg.area))
        full.append(dict(type='Feature',properties=dict(props,object_state='UNRESOLVED',candidate_geometry_reference=uid),geometry=None))
    geo=lambda fs:dict(type='FeatureCollection',crs=dict(properties=dict(name='EPSG:32649')),features=fs)
    return dict(native=dict(ROOF_INSTANCE_CANDIDATES=source),quality=dict(records=quality,checkpoint_sha256='c'*64),
                governance=dict(records=gov),area=dict(records=area),predictions=dict(records=pred),raw=geo(raw),complete=geo(full)),parent,aff


class CachedCandidateTests(unittest.TestCase):
    def test_three_versions_preserve_null_prediction_and_original_ids(self):
        d,p,a=documents();n,b,s,f,l=decode_products(d,p,a)
        self.assertEqual((len(n),len(b),len(s),len(f)),(3,3,2,1))
        self.assertEqual(set(b),{'original_0','original_1','original_2'})
        self.assertIsNone(l[2]['V2_vector_m']);self.assertEqual(l[0]['formal_position_selection'],'UNRESOLVED')

    def test_duplicate_observation_and_state_are_not_silently_deduplicated(self):
        d,p,a=documents();d['native']['ROOF_INSTANCE_CANDIDATES'].append(copy.deepcopy(d['native']['ROOF_INSTANCE_CANDIDATES'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):decode_products(d,p,a)
        d,p,a=documents();d['raw']['features'].append(copy.deepcopy(d['raw']['features'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):decode_products(d,p,a)

    def test_missing_source_and_wrong_source_hash_fail(self):
        for value in ('', 'b'*64):
            d,p,a=documents();d['native']['ROOF_INSTANCE_CANDIDATES'][0]['original_source_hash']=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'source'):decode_products(d,p,a)

    def test_invalid_geometry_fails_without_buffer_repair(self):
        d,p,a=documents();d['native']['ROOF_INSTANCE_CANDIDATES'][0]['geometry_pixel_edge_xy']={'type':'Polygon','coordinates':[[[0,0],[1,1],[1,0],[0,1],[0,0]]]}
        with self.assertRaisesRegex(ValueError,'Invalid candidate geometry'):decode_products(d,p,a)

    def test_wrong_vector_source_and_geometry_fail(self):
        for change in ('vector','origin'):
            d,p,a=documents()
            if change=='vector':d['area']['records'][0]['shift_east_north_m'][0]+=1
            else:d['predictions']['records'][0]['source_crop_xy']=[1,0]
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'provenance'):decode_products(d,p,a)

    def test_unresolved_geometry_or_formal_weight_cannot_be_promoted(self):
        d,p,a=documents();d['complete']['features'][-1]['geometry']=d['raw']['features'][0]['geometry']
        with self.assertRaisesRegex(ValueError,'Unresolved'):decode_products(d,p,a)
        d,p,a=documents();d['quality']['records'][0]['formal_training_weight']=1
        with self.assertRaisesRegex(ValueError,'promoted'):decode_products(d,p,a)

    def test_preattached_formal_geometry_and_identity_are_rejected(self):
        for key,value in [('physical_id','invented'),('registered_geometry',mapping(box(0,0,1,1))),('applied_vector_m',[0,0]),('GT_USED_FOR_GEOMETRY',True)]:
            d,p,a=documents();d['raw']['features'][0]['properties'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Unqualified formal'):decode_products(d,p,a)

    def test_success_receipt_not_published_when_sidecar_or_commit_fails(self):
        for fail_on in (1,2):
            with self.subTest(fail_on=fail_on),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=root/'original.txt';source.write_text('preserve')
                import os
                rename=os.rename;calls=[]
                def failing_rename(left,right):
                    calls.append(str(right))
                    if len(calls)==fail_on:raise OSError('synthetic receipt publication failure')
                    return rename(left,right)
                with patch('xian_r5.cached_roof_products.os.rename',side_effect=failing_rename):
                    with self.assertRaisesRegex(OSError,'synthetic'):publish_receipt(root,dict(ENGINEERING_PASS=True))
                self.assertFalse((root/'FINAL_RECEIPT.json').exists())
                self.assertEqual(source.read_text(),'preserve')

    def test_success_receipt_commits_with_matching_hash_sidecar(self):
        from xian_r5.integration import sha256
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);publish_receipt(root,dict(ENGINEERING_PASS=True))
            self.assertEqual((root/'FINAL_RECEIPT.sha256').read_text().split()[0],sha256(root/'FINAL_RECEIPT.json'))
            self.assertFalse(list(root.glob('*.pending')))

    def test_wrong_complete_version_is_rejected(self):
        d,p,a=documents();d['complete']['features']=[f for f in d['complete']['features'] if f['properties']['object_state']!='V2_SHIFT_CANDIDATE']
        with self.assertRaisesRegex(ValueError,'version'):decode_products(d,p,a)

    def test_native_tile_binding_does_not_use_old_clipped_ROI_support(self):
        d,p,a=documents();tile=dict(download_tile_id='1:2:20',mosaic_window=[0,0,100,100],actual_date=p['source_observed_date'],support_geometry=mapping(box(0,0,1,1)))
        selected=source_tiles(p,dict(tiles=[tile]))
        self.assertEqual(selected[0]['upstream_clipped_support_geometry'],tile['support_geometry'])
        self.assertTrue(shape(selected[0]['support_geometry']).covers(project_native(a)(box(10,10,20,20))))
        with self.assertRaisesRegex(ValueError,'coverage'):source_tiles(p,dict(tiles=[]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):source_tiles(p,dict(tiles=[tile,tile]))

    def test_missing_native_tile_strip_and_conflicting_date_fail(self):
        _,p,_=documents();tile=dict(download_tile_id='1:2:20',mosaic_window=[0,0,99,100],actual_date=p['source_observed_date'],support_geometry=mapping(box(0,0,1,1)))
        with self.assertRaisesRegex(ValueError,'coverage'):source_tiles(p,dict(tiles=[tile]))
        tile.update(mosaic_window=[0,0,100,100],actual_date='2019-01-01')
        with self.assertRaisesRegex(ValueError,'date conflict'):source_tiles(p,dict(tiles=[tile]))

    def test_gpkg_existing_component_adapter_preserves_exact_WKB(self):
        d,p,a=documents();n,b,s,f,l=decode_products(d,p,a)
        tiles=source_tiles(p,dict(tiles=[dict(download_tile_id='1:2:20',mosaic_window=[0,0,100,100],actual_date=p['source_observed_date'],source_JPG='source.jpg',source_JPG_sha256='f'*64,support_geometry=mapping(box(0,0,1,1)))]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'handoff.gpkg';write_gpkg(path,b,n,l,tiles,p,s,f)
            records=adapt_native_components(path,tiles)
            self.assertEqual(len(records),3)
            for r in records:
                self.assertEqual(r['geometry_sha256'],geometry_hash(b[r['original_id']]))
                self.assertIsNone(r['registered_geometry']);self.assertIsNone(r['physical_id'])

    def test_gpkg_candidate_semantic_readback_rejects_wrong_hash(self):
        d,p,a=documents();n,b,s,f,l=decode_products(d,p,a)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'handoff.gpkg';write_gpkg(path,b,n,l,[],p,s,f)
            rows=read_candidate_view(path,b,s,f,l)
            self.assertEqual(len(rows),3*3*2+2+1)
            corrupt=Path(folder)/'corrupt_output.gpkg'
            with fiona.open(path,layer='roof_position_candidates') as src:
                with fiona.open(corrupt,'w',driver='GPKG',layer='roof_position_candidates',crs=src.crs,schema=src.schema) as dst:
                    for feature in src:
                        properties=dict(feature.properties)
                        if properties['object_state']=='NO_SHIFT_CANDIDATE':properties['geometry_sha256']='0'*64
                        dst.write(dict(geometry=feature.geometry,properties=properties))
            with self.assertRaisesRegex(ValueError,'hash changed'):read_candidate_view(corrupt,b,s,f,l)

    def test_source_file_bad_sha_or_missing_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.jpg';path.write_bytes(b'synthetic source')
            pin=dict(path=str(path.resolve()),role='SOURCE_COVERAGE',sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'Artifact changed'):verify_artifacts({'source_tile':pin})
            path.unlink()
            with self.assertRaises(FileNotFoundError):verify_artifacts({'source_tile':pin})

    def test_cached_fraction_and_formal_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);aff=Affine(10,0,300000,0,-10,3790000)
            g=box(300001,3789991,300006,3789996);v=translate(g,2,0);paths={}
            arrays={name:np.zeros((2,2),np.float32) for name in ('B0','RETAIN_ORIGINAL','V2_MIXED_AVAILABLE','COMPLETE_INSTANCE_V2','GEOMETRY_CONFLICT','COMPLETE_INSTANCE_CONFLICT')}
            for name in ('B0','RETAIN_ORIGINAL'):arrays[name][0,0]=.25
            for name in ('V2_MIXED_AVAILABLE','COMPLETE_INSTANCE_V2'):arrays[name][0,0]=.25
            arrays.update(SOURCE_SUPPORT=np.ones((2,2),np.float32),UNKNOWN_POSITION=np.ones((2,2),np.float32))
            for name,array in arrays.items():
                file='FRACTION10M_'+name+'.tif';paths[file]=root/file
                with rasterio.open(paths[file],'w',driver='GTiff',height=2,width=2,count=1,crs='EPSG:32649',transform=aff,dtype='float32') as dst:dst.write(array,1)
            for file,array,nodata in [('FORMAL_SUPERVISION_STATE10M.tif',np.full((2,2),255,np.uint8),255),('FORMAL_TRAINING_WEIGHT10M.tif',np.zeros((2,2),np.float32),None)]:
                paths[file]=root/file
                with rasterio.open(paths[file],'w',driver='GTiff',height=2,width=2,count=1,crs='EPSG:32649',transform=aff,dtype=array.dtype,nodata=nodata) as dst:dst.write(array,1)
            grid,errors=validate_fraction_bundle(paths,{'id':g},{'id':v},{'id'})
            self.assertEqual(grid.width,2);self.assertEqual(len(errors),3)
            with rasterio.open(paths['FORMAL_TRAINING_WEIGHT10M.tif'],'r+') as dst:dst.write(np.ones((2,2),np.float32),1)
            with self.assertRaisesRegex(ValueError,'Unqualified formal'):validate_fraction_bundle(paths,{'id':g},{'id':v},{'id'})


if __name__=='__main__':unittest.main()
