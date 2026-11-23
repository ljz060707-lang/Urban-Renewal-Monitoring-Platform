import unittest
from copy import deepcopy
from shapely.geometry import box
from pyproj import Transformer
from shapely.ops import transform
from xian_r5.roof_observation import source_roof_observation


class RoofObservationTests(unittest.TestCase):
    def fixture(self):
        packet={'source_parent_component':'source','image_path':'raw.png','image_sha256':'a'*64,
            'native_rgb_pixel_sha256':'b'*64,'source_window':[0,0,100,100],
            'source_raster_affine':[.00001,0,109,0,-.00001,34.3],'source_raster_crs':'EPSG:4326',
            'local_image_shape':[100,100],'native_resampling':'NONE','source_observed_date':'2018-11-01','experiment_year':2019}
        annotation={'observation_id':'MODEL_ROOF_SURFACE_1','method':'VISION_MODEL_RAW_SOURCE_DIGITIZATION',
            'vertices':[[10,10],[20,10],[20,20],[10,20]],'coordinate_convention':'LOCAL_PIXEL_EDGE_XY',
            'complete_instance':False,'uses_target_or_reference_position':False,
            'feature_basis':'Visible planar roof material; excludes facade and courtyard; boundary is a model candidate.'}
        t=Transformer.from_crs(4326,32649,always_xy=True)
        scope=transform(t.transform,box(108.999,34.298,109.002,34.301))
        return packet,annotation,scope,scope

    def test_pixel_conversion_not_correction(self):
        r,g=source_roof_observation(*self.fixture());self.assertGreater(g.area,0)
        self.assertIsNone(r['candidate_displacement_m']);self.assertFalse(r['target_reference_position_used'])

    def test_never_full_footprint_label_or_physical_id(self):
        r,_=source_roof_observation(*self.fixture());self.assertFalse(r['may_generate_fraction_labels'])
        self.assertFalse(r['ground_footprint']);self.assertIsNone(r['physical_building_id'])

    def test_no_uncertainty_invented(self):
        r,_=source_roof_observation(*self.fixture());self.assertIsNone(r['localization_bound_m'])
        self.assertFalse(r['formal_review'])

    def test_original_annotation_not_modified(self):
        args=self.fixture();before=deepcopy(args[1]);source_roof_observation(*args);self.assertEqual(before,args[1])

    def test_target_guidance_rejected(self):
        args=self.fixture();args[1]['uses_target_or_reference_position']=True
        with self.assertRaises(ValueError):source_roof_observation(*args)

    def test_complete_instance_claim_rejected(self):
        args=self.fixture();args[1]['complete_instance']=True
        with self.assertRaises(ValueError):source_roof_observation(*args)

    def test_outside_native_packet_rejected(self):
        args=self.fixture();args[1]['vertices'][0]=[-1,0]
        with self.assertRaises(ValueError):source_roof_observation(*args)

    def test_invalid_polygon_not_fixed(self):
        args=self.fixture();args[1]['vertices']=[[10,10],[20,20],[10,20],[20,10]]
        with self.assertRaises(ValueError):source_roof_observation(*args)

    def test_missing_image_hash_rejected(self):
        args=self.fixture();args[0]['image_sha256']='confidence_high'
        with self.assertRaises(ValueError):source_roof_observation(*args)

    def test_resized_image_not_native_position_source(self):
        args=self.fixture();args[0]['native_resampling']='BILINEAR'
        with self.assertRaises(ValueError):source_roof_observation(*args)

    def test_support_not_silently_clipped(self):
        args=list(self.fixture());args[3]=box(0,0,1,1)
        with self.assertRaises(ValueError):source_roof_observation(*args)

    def test_not_parent_snapped(self):
        args=list(self.fixture());args[2]=box(0,0,1,1)
        with self.assertRaises(ValueError):source_roof_observation(*args)


if __name__=='__main__':unittest.main()
