"""Small live adapter invariants; synthetic data, no test imagery or models."""
import copy
import hashlib
import unittest
import numpy as np
from shapely.geometry import box,mapping,shape,Polygon
from xian_r5.shadow_pipeline import validate_shadow_manifest,MODE,B0_SHA,new_records
from xian_r5.integration import CONTRACT

class LiveShadowTests(unittest.TestCase):
    def manifest(self):
        return dict(execution_mode=MODE,contract_version=CONTRACT,model_role='ORIGINAL_B0_ONLY',run_role='SHADOW_CANDIDATE',manual_gt_required_for_inference=False,instance_checkpoint=None,reviewed_alignment_fields=None,inputs={'B0':{'sha256':B0_SHA}})
    def test_original_baseline_admitted(self):validate_shadow_manifest(self.manifest())
    def test_development_model_not_default(self):
        m=self.manifest();m['inputs']['B0']['sha256']='a'*64
        with self.assertRaises(ValueError):validate_shadow_manifest(m)
    def test_unknown_is_not_zero_checkpoint(self):
        m=self.manifest();m['instance_checkpoint']='old_failed_vector'
        with self.assertRaises(ValueError):validate_shadow_manifest(m)
    def test_qualified_role_cannot_be_renamed(self):
        m=self.manifest();m['run_role']='QUALIFIED_SUPERVISION'
        with self.assertRaises(ValueError):validate_shadow_manifest(m)
    def test_no_GT_input_requirement(self):
        m=self.manifest();m['manual_gt_required_for_inference']=True
        with self.assertRaises(ValueError):validate_shadow_manifest(m)
    def candidate(self,g):
        return {'candidate_id':'runtime-original-id','geometry_pixel_edge_xy':mapping(g),'pixel_area':1,'source_edge_partial_risk':False,'mean_probability_uncalibrated':.8}
    def packet(self):
        return dict(source_raster_affine=[1,0,0,0,1,0],source_window=[0,0,10,10],source_raster_crs='EPSG:32649',source_observed_date='2018-11-01',image_path='native.png')
    def test_pixel_geometry_preserved_and_ID_not_reassigned(self):
        g=box(1,1,2,2);src={'ROOF_INSTANCE_CANDIDATES':[self.candidate(g)]}
        tile=dict(support_geometry=mapping(box(0,0,10,10)),actual_date='2018-11-01',download_tile_id='original',source_JPG='tile.jpg',source_JPG_sha256='a'*64)
        rows,bad=new_records(src,self.packet(),[tile]);r=rows[0]
        self.assertEqual(r['observation_id'],'runtime-original-id');self.assertIsNone(r['physical_id']);self.assertIsNone(r['source_object_id'])
        self.assertEqual(shape(r['geometry']).wkb,g.wkb);self.assertEqual(shape(r['native_pixel_geometry']).wkb,g.wkb);self.assertEqual(bad,[])
    def test_invalid_native_geometry_not_silently_repaired(self):
        g=Polygon([(0,0),(2,2),(0,2),(2,0),(0,0)])
        rows,bad=new_records({'ROOF_INSTANCE_CANDIDATES':[self.candidate(g)]},self.packet(),[dict(support_geometry=mapping(box(0,0,10,10)))])
        self.assertEqual(rows,[]);self.assertEqual(len(bad),1);self.assertEqual(shape(bad[0]['geometry']).wkb,g.wkb)
if __name__=='__main__':unittest.main()
