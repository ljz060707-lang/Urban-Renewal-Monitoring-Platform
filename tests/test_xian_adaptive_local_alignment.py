"""Public fit/predict seams explicitly requested by the R5-A specification."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xian_r5.alignment.adaptive_local import fit_local_alignment
import numpy as np
from affine import Affine
from shapely.geometry import box,mapping
from xian_r5.label_transfer import Grid
from xian_r5.source_partition import SourceRegion

class AdaptiveFitTests(unittest.TestCase):
    def test_exposed_unconfirmed_pairs_do_not_train_a_model(self):
        data=[{'pair_id':'p1','identity_status':'SHAPE_HYPOTHESIS',
            'time_compatible':None,'reference_quality':'USER_CONFIRMED_ROOF_ROLE',
            'training_split':'DEVELOPMENT_EXPOSED','source_component_id':'empirical1',
            'actual_date':'2018-11-01'}]
        result=fit_local_alignment(data,{},[],{})
        self.assertEqual(result['supervision_eligible_pairs'],0)
        self.assertFalse(result['model_trained'])
        self.assertEqual(result['status'],'INSUFFICIENT_QUALIFIED_TRAINING_PAIRS')
        self.assertNotIn('reference_roofs',result['checkpoint'])

    def test_overlapping_image_patches_cannot_cross_training_and_validation(self):
        common={'identity_status':'PROVEN_SAME_ROOF','time_compatible':True,'reference_quality':True,
            'label_semantics':'ROOF_OBSERVATION_TO_PLANIMETRIC_ROOF_REFERENCE','label_dx_m':1.,'label_dy_m':0.,
            'actual_date':'2018-11-01','empirical_alignment_region':'one','candidate_geometry':mapping(box(1,1,4,4))}
        rows=[{**common,'pair_id':'p1','roof_observation_id':'o1','roof_gt_id':'g1','training_split':'TRAIN',
            'acquisition_geographic_group':'group1','image_patch':{'raster_path':'same.tif','native_pixel_window':[0,0,20,20]}},
            {**common,'pair_id':'p2','roof_observation_id':'o2','roof_gt_id':'g2','training_split':'VALIDATION',
            'acquisition_geographic_group':'group2','image_patch':{'raster_path':'same.tif','native_pixel_window':[10,0,20,20]}}]
        with self.assertRaisesRegex(ValueError,'overlapping'):
            fit_local_alignment(rows,{'g1':box(0,0,5,5),'g2':box(10,0,15,5)},
                [SourceRegion('one','2018-11-01',box(0,0,100,100))],{})

    def test_gt_free_prediction_keeps_unknown_instead_of_borrowing_a_vector(self):
        from xian_r5.alignment.adaptive_local import predict_local_alignment
        from unittest.mock import patch
        image=np.zeros((64,64,3),np.uint8)
        grid=Grid(Affine(10,0,0,0,-10,640),64,64,'EPSG:32649')
        regions=[SourceRegion('one','2018-11-01',box(0,0,640,640),(0.,-10.),'a'*64)]
        cp=fit_local_alignment([],{},regions,{})['checkpoint']
        target={'grid':grid,'reference_rgb':image,'reference_valid':np.ones((64,64),bool)}
        with patch('builtins.open',side_effect=AssertionError('Inference must not open GT files')):
            result=predict_local_alignment(image,{},regions,target,cp)
        self.assertEqual(result['qualified_regions'],[])
        self.assertTrue(all(r['selected_vector_m'] is None for r in result['regions']))
        self.assertEqual(result['position_semantics'],'RELATIVE_IMAGE_GRID_ONLY')
        self.assertFalse(result['roof_plane_correction_validated'])

    def test_nan_displacement_cannot_be_admitted_even_with_positive_flags(self):
        row={'pair_id':'p1','identity_status':'PROVEN_SAME_ROOF','time_compatible':True,
            'reference_quality':True,'roof_gt_id':'g1','training_split':'TRAIN',
            'label_semantics':'ROOF_OBSERVATION_TO_PLANIMETRIC_ROOF_REFERENCE',
            'label_dx_m':float('nan'),'label_dy_m':0.,'actual_date':'2018-11-01'}
        result=fit_local_alignment([row],{'g1':box(0,0,5,5)},[],{})
        self.assertEqual(result['supervision_eligible_pairs'],0)

    def test_positive_flags_without_source_date_support_do_not_admit_supervision(self):
        row={'pair_id':'p1','identity_status':'PROVEN_SAME_ROOF','time_compatible':True,
            'reference_quality':True,'roof_gt_id':'g1','training_split':'TRAIN',
            'label_semantics':'ROOF_OBSERVATION_TO_PLANIMETRIC_ROOF_REFERENCE',
            'label_dx_m':1.,'label_dy_m':0.,'actual_date':'2018-11-01',
            'empirical_alignment_region':'missing','roof_observation_id':'o1'}
        result=fit_local_alignment([row],{'g1':box(0,0,5,5)},[],{})
        self.assertEqual(result['supervision_eligible_pairs'],0)

    def test_same_roof_cannot_cross_calibration_and_holdout(self):
        common={'identity_status':'SHAPE_HYPOTHESIS','roof_observation_id':'same_roof','roof_gt_id':'same_gt'}
        rows=[{**common,'pair_id':'p1','training_split':'CALIBRATION'},
            {**common,'pair_id':'p2','training_split':'FINAL_HOLDOUT'}]
        with self.assertRaisesRegex(ValueError,'leakage'):fit_local_alignment(rows,{},[],{})

    def test_inference_rejects_mixed_actual_acquisitions(self):
        from xian_r5.alignment.adaptive_local import predict_local_alignment
        image=np.ones((64,64,3),np.uint8);grid=Grid(Affine(10,0,0,0,-10,640),64,64,'EPSG:32649')
        regs=[SourceRegion('a','2018-11-01',box(0,0,320,640)),SourceRegion('b','2019-12-02',box(320,0,640,640))]
        cp=fit_local_alignment([],{},[],{})['checkpoint']
        with self.assertRaisesRegex(ValueError,'acquisition'):
            predict_local_alignment(image,{},regs,{'grid':grid,'reference_rgb':image,'reference_valid':np.ones((64,64),bool)},cp)

    def test_inference_rejects_gt_payload_and_does_not_claim_camera_source(self):
        from xian_r5.alignment.adaptive_local import predict_local_alignment
        image=np.ones((64,64,3),np.uint8);grid=Grid(Affine(10,0,0,0,-10,640),64,64,'EPSG:32649')
        regs=[SourceRegion('a','2018-11-01',box(0,0,640,640))]
        cp=fit_local_alignment([],{},[],{})['checkpoint']
        target={'grid':grid,'reference_rgb':image,'reference_valid':np.ones((64,64),bool)}
        with self.assertRaisesRegex(ValueError,'Only explicit'):
            predict_local_alignment(image,{},regs,{**target,'reference_roofs':{}},cp)
        result=predict_local_alignment(image,{},regs,target,cp)
        self.assertTrue(all(r['same_viewing_angle'] is None for r in result['regions']))

    def test_subpixel_transport_preserves_supplied_vector_without_10m_rounding(self):
        from xian_r5.source_partition import source_partition_transport
        result=source_partition_transport({'roof':box(10,10,20,20)},
            [SourceRegion('a','2018-11-01',box(0,0,100,100),(1.25,-.75),'a'*64)],box(0,0,100,100))
        self.assertEqual(result['candidate_geometries']['roof'].centroid.x,16.25)
        self.assertEqual(result['candidate_geometries']['roof'].centroid.y,14.25)

    def test_crossing_roof_prevents_quartering_adoption(self):
        from xian_r5.alignment.adaptive_local import predict_local_alignment
        image=np.ones((64,64,3),np.uint8);grid=Grid(Affine(10,0,0,0,-10,640),64,64,'EPSG:32649')
        regs=[SourceRegion('a','2018-11-01',box(0,0,640,640))];cp=fit_local_alignment([],{},[],{})['checkpoint']
        result=predict_local_alignment(image,{'crossing':box(300,300,340,340)},regs,
            {'grid':grid,'reference_rgb':image,'reference_valid':np.ones((64,64),bool)},cp)
        self.assertEqual(len(result['regions']),1)
        self.assertEqual(result['regions'][0]['subdivision_status'],'DECLINED_WHOLE_ROOF_CROSSING')


if __name__=='__main__':unittest.main()
