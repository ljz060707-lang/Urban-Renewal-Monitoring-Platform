"""No-checkpoint safety contract, not learned predictor accuracy tests."""
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
from affine import Affine
from shapely.geometry import box
from xian_r5.label_transfer import Grid
from xian_r5.alignment.instance_adaptive import predict_instance_alignment, TARGET_B


class InstanceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.g=box(0,0,2,2);self.rgb=np.zeros((16,16,3),dtype='uint8')
        self.grid=Grid(Affine(10,0,0,0,-10,160),16,16,'EPSG:32649')
        self.rows=[dict(observation_id='original_1',roof_polygon_raw=self.g,roof_instance_quality='CANDIDATE')]
        self.registry={'original_1':dict(source_component_id=None,actual_date='2018-11-01',
            native_image_frame={'crs':'EPSG:4326'},source_support_complete=False,
            source_geometry_hash=hashlib.sha256(self.g.wkb).hexdigest(),source_payload_hashes=[],source_tile_ids=[],
            photographic_seam_status='UNKNOWN')}

    def predict(self,**kw):
        args=dict(ge_rgb=self.rgb,roof_instances=self.rows,source_registry=self.registry,target_grid=self.grid,checkpoint=None)
        args.update(kw);return predict_instance_alignment(**args)

    def test_missing_checkpoint_not_zero(self):
        r=self.predict()[0]
        for k in ['total_dx_m','total_dy_m','region_dx_m','region_dy_m',
                  'instance_residual_dx_m','instance_residual_dy_m','prediction_uncertainty','physical_id','registered_geometry']:
            self.assertIsNone(r[k])
        self.assertEqual(r['application_status'],'UNRESOLVED')
        self.assertEqual(r['target_semantics'],TARGET_B)

    def test_no_gt_or_filesystem(self):
        def deny(*a,**kw):raise AssertionError('File read forbidden')
        with patch('builtins.open',deny),patch('io.open',deny),patch('os.open',deny):
            self.assertEqual(len(self.predict()),1)

    def test_geometry_and_original_id_preserved(self):
        r=self.predict()[0]
        self.assertEqual(r['observation_id'],'original_1')
        self.assertEqual(r['roof_polygon_raw'].wkb,self.g.wkb)

    def test_no_cached_vectors_or_manual_matches(self):
        for k in ('cached_offset','GT','manual_pair','region_dx_m','student_prediction'):
            registry={'original_1':dict(self.registry['original_1'],**{k:None})}
            with self.assertRaises(ValueError):self.predict(source_registry=registry)

    def test_unsupported_checkpoint_rejected(self):
        for c in ({'target_semantics':'ROOF_TO_GROUND_FOOTPRINT'}, {'offset':[0,-10]}, {'algorithm':'POINT_RGB'}):
            with self.assertRaises(ValueError):self.predict(checkpoint=c)

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):self.predict(roof_instances=self.rows*2)

    def test_geometry_provenance_mismatch(self):
        self.registry['original_1']['source_geometry_hash']='0'*64
        with self.assertRaises(ValueError):self.predict()

    def test_nonfinite_rgb_rejected(self):
        rgb=self.rgb.astype('float64');rgb[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.predict(ge_rgb=rgb)

    def test_wrong_crs_rejected(self):
        with self.assertRaises(ValueError):self.predict(target_grid=Grid(self.grid.affine,16,16,'EPSG:4326'))

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):self.predict(source_registry={})


if __name__=='__main__':unittest.main()
