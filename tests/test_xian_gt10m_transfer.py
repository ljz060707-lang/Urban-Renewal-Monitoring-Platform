"""Synthetic contract tests; not actual geographic validation."""
from pathlib import Path
import unittest
import tempfile
import numpy as np

from affine import Affine
from xian_r5.alignment.mosaic_grid import MosaicTileGrid
from xian_r5.label_transfer import Grid
from shapely.geometry import box
import xian_r5.label_transfer as transfer


class CoordinateContract(unittest.TestCase):
    def test_gdal_tuple_order_is_explicit_including_rotation(self):
        grid = Grid(Affine(8,6,100,6,-8,200),2,2,'EPSG:32649')
        self.assertEqual(grid.affine.to_gdal(), (100,8,6,200,6,-8))
        recovered = Grid(Affine.from_gdal(100,8,6,200,6,-8),2,2,'EPSG:32649')
        self.assertEqual(recovered.pixel_to_map(.5,.5), (107,199))

    def test_geid_northward_y_is_not_xyz_and_has_inclusive_113_row_offset(self):
        grid = MosaicTileGrid(Path('task'), Path('raster'), Path('tiles'), 'synthetic',
            '2019-01-01', 20, 420534, 311895, 421138, 312289, 256, 256,
            'EPSG:4326', 154880, 101120, (1,0,0,0,-1,0), (0,0,1,1),0,0,0,0)
        self.assertEqual(grid.window_for_tile(420797,312176).pixel_window,
                         [67328,28928,256,256])
        with self.assertRaises(ValueError):
            grid.window_for_tile(0,312176)

    def test_target_window_preserves_affine_and_half_pixel_roundtrip(self):
        grid = Grid(Affine(10,0,293300,0,-10,3812400),3885,3086,'EPSG:32649')
        pilot = grid.window((310400,3803200,310600,3803400))
        self.assertEqual((pilot.width,pilot.height),(20,20))
        self.assertEqual(pilot.pixel_to_map(.5,.5),(310405,3803395))
        self.assertEqual(pilot.map_to_pixel(310405,3803395),(.5,.5))
        self.assertEqual(pilot.nested(10).pixel_to_map(5,5),(310405,3803395))
        with self.assertRaises(ValueError):
            grid.window((0,0,1,1))

    def test_xyz_tms_row_sign_and_zero_based_edges_are_explicit(self):
        # Web XYZ row2 at z3 is TMS row5; GEID's proprietary grid must not reuse it.
        self.assertEqual(transfer.web_row(2,3,'XYZ','TMS'),5)
        self.assertEqual(transfer.web_row(5,3,'TMS','XYZ'),2)
        with self.assertRaises(ValueError): transfer.web_row(2,3,'GEID','XYZ')

    def test_crop_resize_inverse_and_degree_axis_order(self):
        from pyproj import Transformer
        source=Affine(.0001,0,108,0,-.0001,35)
        resized=source @ Affine.translation(100,200) @ Affine.scale(.5)
        self.assertAlmostEqual((resized @ (20,40))[0],108.011,places=8)
        self.assertAlmostEqual((resized @ (20,40))[1],34.978,places=8)
        self.assertAlmostEqual((~resized @ (108.011,34.978))[0],20,places=6)
        transformer=Transformer.from_crs(4326,32649,always_xy=True)
        x,y=transformer.transform(111,0)
        self.assertAlmostEqual(x,500000,places=5)
        self.assertAlmostEqual(y,0,places=5)

    def test_rotated_target_grid_nested_area_is_one_hundredth(self):
        g=Grid(Affine(8,6,100,6,-8,200),2,2,'EPSG:32649')
        self.assertAlmostEqual(g.cell(0,0).area,100)
        self.assertAlmostEqual(g.nested(10).cell(0,0).area,1)


class SupervisionContract(unittest.TestCase):
    def test_completed_resume_stops_when_training_input_inventory_changes(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pilot',Path(__file__).resolve().parents[1]/'tools/run_xian_gt10m_pilot.py')
        pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scope=root/'search';scope.mkdir();run=root/'synthetic';run.mkdir()
            snapshot=pilot.binding_scope_snapshot(scope)
            pilot.write_json(run/'TRAINING_INPUT_AND_USAGE_AUDIT.json',{
                'binding_scope_snapshots':[snapshot],
                'underlying_source_lookup':{'expected_path':str(scope/'Xian_2019.tif'),'exists':False}})
            pilot.write_json(run/'OUTPUT_SHA256_MANIFEST.json',{'outputs':{},'code_sha256':{},'protected_sha256':{}})
            pilot.write_json(run/'IMPLEMENTATION_RECEIPT.json',{
                'manifest_sha256':pilot.sha(run/'OUTPUT_SHA256_MANIFEST.json'),'terminal':'SYNTHETIC'})
            pilot.write_json(scope/'training_inputs.csv',{'synthetic':True})
            with self.assertRaises(RuntimeError): pilot.eligibility(run)

    def test_masked_unknown_cannot_be_coerced_to_true_or_a_target(self):
        yes=np.ones((1,1),dtype=bool)
        masked=np.ma.array(yes,mask=True)
        with self.assertRaises(ValueError):
            transfer.supervision_eligibility(time_ok=masked,geometry_ok=yes,state_ok=yes,interface_ok=yes)
        with self.assertRaises(ValueError):
            transfer.fraction_supervision(np.ma.array([[[.2]]],mask=True),
                valid=np.ones((1,1)),unknown=np.zeros((1,1)),eligible=yes)

    def test_unknown_or_mismatched_masks_cannot_grant_admission(self):
        yes=np.ones((1,2),dtype=bool)
        for invalid in (np.array([[None,None]]),np.array([[1,1]]),np.ones((2,1),dtype=bool)):
            with self.assertRaises(ValueError):
                transfer.supervision_eligibility(time_ok=invalid,geometry_ok=yes,state_ok=yes,interface_ok=yes)

    def test_unknown_pixels_are_not_normalized_or_given_training_weight(self):
        with self.assertRaises(ValueError):
            transfer.fraction_supervision(np.array([[[.8]]]),valid=np.array([[.5]]),
                unknown=np.array([[.5]]),eligible=np.array([[False]]))
        with self.assertRaises(ValueError):
            transfer.fraction_supervision(np.array([[[0.]]]),valid=np.array([[0.]]),
                unknown=np.array([[1.]]),eligible=np.array([[True]]))
        with self.assertRaises(ValueError):
            transfer.fraction_supervision(np.array([[[float('nan')]]]),valid=np.array([[1.]]),
                unknown=np.array([[0.]]),eligible=np.array([[False]]))

    def test_soft_batch_preserves_full_pixel_denominator_and_eligibility_weights(self):
        batch = transfer.fraction_supervision(
            np.array([[[.2,.3]]]), valid=np.array([[.5,1.]]),
            unknown=np.array([[.5,0.]]), eligible=np.array([[False,True]]))
        np.testing.assert_array_equal(batch['target'],np.array([[[.2,.3]]]))
        np.testing.assert_array_equal(batch['weight'],np.array([[0.,1.]]))
        self.assertEqual(batch['denominator'],'FULL_TARGET_PIXEL')

    def test_exclusions_are_inclusive_and_disjoint_reasons_partition_denominator(self):
        report = transfer.supervision_eligibility(
            time_ok=np.array([[True,False,False,True]]),
            geometry_ok=np.array([[True,True,False,True]]),
            state_ok=np.array([[True,True,False,False]]),
            interface_ok=np.ones((1,4),dtype=bool))
        self.assertEqual(report['counts']['eligible'],1)
        self.assertEqual(report['counts']['EXCLUDED_BY_TIME'],2)
        self.assertEqual(report['counts']['EXCLUDED_BY_GEOMETRY'],1)
        self.assertEqual(report['counts']['EXCLUDED_BY_STATE_EVIDENCE'],2)
        self.assertEqual(report['counts']['EXCLUDED_BY_MULTIPLE_CAUSES'],1)
        self.assertEqual(report['exclusive_reason_counts'],{'NONE':1,'TIME':1,'TIME+GEOMETRY+STATE_EVIDENCE':1,'STATE_EVIDENCE':1})

class AreaContract(unittest.TestCase):
    def test_partial_valid_area_stays_absolute_and_unknown_not_background(self):
        grid = Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649')
        out = transfer.aggregate(grid, {'ROOF':box(0,0,2,10)}, box(0,0,5,10))
        self.assertAlmostEqual(out['fractions'][0,0,0],.2)
        self.assertAlmostEqual(out['valid'][0,0],.5)
        self.assertAlmostEqual(out['unknown'][0,0],.5)
        self.assertAlmostEqual(out['unassigned'][0,0],.3)
        self.assertEqual(out['area_conservation_error_m2'],0)

    def test_exclusive_overlap_fails_but_same_pixel_dual_events_survive(self):
        grid = Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649')
        events = {'ADDITION':box(0,0,2,10),'DEMOLITION':box(0,0,2,10)}
        with self.assertRaises(ValueError):
            transfer.aggregate(grid,events,box(0,0,10,10))
        out=transfer.aggregate(grid,events,box(0,0,10,10),events=True)
        self.assertEqual(out['fractions'][:,0,0].tolist(),[.2,.2])

    def test_unknown_uncovered_area_not_renormalized_to_full_cell(self):
        grid=Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649')
        out=transfer.aggregate(grid,{'ROOF':box(0,0,1,1)},box(0,0,1,1))
        self.assertEqual(out['fractions'][0,0,0],.01)
        self.assertEqual(out['unknown'][0,0],.99)

    def test_small_roofs_and_known_background_partition_exactly(self):
        g=Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649'); valid=box(0,0,10,10)
        roof=box(0,0,1,1)
        out=transfer.aggregate(g,{'ROOF':roof,'KNOWN_BACKGROUND':valid.difference(roof)},valid)
        self.assertEqual(out['fractions'][:,0,0].tolist(),[.01,.99])
        self.assertEqual(out['unassigned'][0,0],0)


class TransportContract(unittest.TestCase):
    def test_outside_source_roof_cannot_move_into_target_supported_label(self):
        rows,valid=transfer.bounded_transport({'outside':box(-3,1,-1,3),'inside':box(1,1,3,3)},
            (3,0),box(0,0,10,10),box(0,0,10,10))
        self.assertEqual(rows['outside'].area,0)
        self.assertEqual(rows['inside'].bounds,(4,1,6,3))
        self.assertEqual(valid.area,70)

    def test_adjacent_chunk_halo_replay_is_equal_to_single_grid_without_ghost(self):
        g=Grid(Affine(10,0,0,0,-10,10),2,1,'EPSG:32649')
        roof=box(8,0,12,5);support=box(0,0,20,10)
        rows=transfer.deduplicate([('same',roof),('same',roof)])
        full=transfer.aggregate(g,{'ROOF':rows['same']},support)
        left=transfer.aggregate(g.window((0,0,10,10)),{'ROOF':rows['same']},support)
        right=transfer.aggregate(g.window((10,0,20,10)),{'ROOF':rows['same']},support)
        self.assertEqual(full['fractions'][0,0].tolist(),[.1,.1])
        self.assertEqual(np.concatenate([left['fractions'],right['fractions']],axis=2).tolist(),full['fractions'].tolist())
        # An actual source-seam disagreement is a conflict, not a feathered category.
        with self.assertRaises(ValueError):
            transfer.aggregate(g,{'STABLE':roof,'DEMOLITION':roof},support)

    def test_json_list_presence_has_same_state_as_tuple(self):
        out=transfer.coordinate_states({'id':box(0,0,2,2)},{'id':box(0,0,2,2)},
            {'id':[1,1]},dates=('2018-04-29','2019-12-02'),frames=('ROOF','ROOF'))
        self.assertEqual(out['STABLE'].area,4)

    def test_null_transform_cannot_pass_as_zero_or_leave_support(self):
        g=box(2,2,4,4); support=box(0,0,10,10)
        with self.assertRaises(ValueError): transfer.transport(g,None,support)
        with self.assertRaises(ValueError): transfer.transport(g,(10,0),support)
        self.assertTrue(transfer.transport(g,(1,2),support).equals(box(3,4,5,6)))
        self.assertTrue(transfer.transport(box(3,4,5,6),(-1,-2),support).equals(g))

    def test_identity_not_contour_xor_preserves_stable_and_real_events(self):
        before={'stable':box(1,1,3,3),'old':box(6,1,8,3)}
        after={'stable':box(2,1,4,3),'new':box(6,1,8,3)}
        out=transfer.coordinate_states(before,after,{'stable':(1,1),'old':(1,0),'new':(0,1)},
            dates=('2018-04-29','2019-12-02'),frames=('ROOF','ROOF'))
        self.assertTrue(out['STABLE'].equals(before['stable']))
        self.assertTrue(out['ADDITION'].equals(after['new']))
        self.assertTrue(out['DEMOLITION'].equals(before['old']))
        self.assertEqual(out['UNRESOLVED'].area,0)

    def test_missing_observation_not_presence_zero_and_frame_mismatch_fails(self):
        with self.assertRaises(ValueError):
            transfer.coordinate_states({}, {},{},dates=('2018-04-29','2019-12-02'),frames=('ROOF','GROUND'))
        out=transfer.coordinate_states({'id':box(0,0,2,2)}, {},{'id':(1,'U')},
            dates=('2018-04-29','2019-12-02'),frames=('ROOF','ROOF'))
        self.assertEqual(out['DEMOLITION'].area,0)
        self.assertEqual(out['UNRESOLVED'].area,4)

    def test_chunk_duplicates_and_conflicting_source_objects_are_not_blended(self):
        rows=[('a',box(0,0,2,2)),('a',box(0,0,2,2)),('b',box(2,0,3,2))]
        self.assertEqual(len(transfer.deduplicate(rows)),2)
        with self.assertRaises(ValueError):
            transfer.deduplicate([('a',box(0,0,2,2)),('a',box(1,0,3,2))])

    def test_actual_translation_before_state_does_not_hide_replacement_events(self):
        support=box(0,0,20,20)
        raw=box(2,2,4,4);later=box(4,3,6,5)
        fixed=transfer.transport(later,(-2,-1),support)
        out=transfer.coordinate_states({'same':raw},{'same':fixed},{'same':(1,1)},
            dates=('2018-04-29','2019-12-02'),frames=('ROOF','ROOF'))
        self.assertEqual((out['ADDITION'].area,out['DEMOLITION'].area),(0,0))

    def test_category_codes_are_selected_not_numerically_interpolated(self):
        self.assertEqual(transfer.nearest_codes(np.array([[1,255],[3,0]],dtype=np.uint8),
                                             [(0.,0.),(1.,0.),(0.,1.)]),[1,255,3])
        with self.assertRaises(ValueError):
            transfer.nearest_codes(np.array([[1]],dtype=np.uint8),[(2,0)])

    def test_null_dates_and_duplicate_dates_cannot_make_change_interval(self):
        for dates in [(None,'2019-12-02'),('2019-12-02','2019-12-02')]:
            with self.assertRaises(ValueError):
                transfer.coordinate_states({}, {},{},dates=dates,frames=('ROOF','ROOF'))


class QualityAndDeliveryContract(unittest.TestCase):
    def test_orphan_staging_is_not_registered_on_close_retry(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pilot',Path(__file__).resolve().parents[1]/'tools/run_xian_gt10m_pilot.py')
        pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'real.json').write_text('{}',encoding='utf8')
            baseline=pilot.output_inventory(root)
            (root/'.partial_orphan').write_text('interrupted staging',encoding='utf8')
            self.assertEqual(pilot.output_inventory(root),baseline)

    def test_completed_receipt_and_code_hash_binding_detect_tampering(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pilot',Path(__file__).resolve().parents[1]/'tools/run_xian_gt10m_pilot.py')
        pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);code=root/'code.py';code.write_text('frozen code',encoding='utf8')
            pilot.write_json(root/'OUTPUT_SHA256_MANIFEST.json',{'outputs':{},'code_sha256':{str(code):pilot.sha(code)}})
            pilot.write_json(root/'IMPLEMENTATION_RECEIPT.json',{'manifest_sha256':pilot.sha(root/'OUTPUT_SHA256_MANIFEST.json')})
            pilot.verify_completed(root)
            code.write_text('changed code',encoding='utf8')
            with self.assertRaises(RuntimeError): pilot.verify_completed(root)

    def test_partial_geopackage_fails_closed_instead_of_being_skipped(self):
        import importlib.util
        import fiona
        from shapely.geometry import mapping
        spec=importlib.util.spec_from_file_location('pilot',Path(__file__).resolve().parents[1]/'tools/run_xian_gt10m_pilot.py')
        pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'interrupted.gpkg'
            with fiona.open(path,'w',driver='GPKG',layer='source_regions',crs='EPSG:32649',
                schema={'geometry':'Polygon','properties':{'id':'str'}}) as dst:
                dst.write({'geometry':mapping(box(0,0,1,1)),'properties':{'id':'synthetic'}})
            with self.assertRaises(ValueError): pilot.verify_ledger(path,{'objects':[]},[])

    def test_atomic_receipt_nan_does_not_leave_partial_file_and_equal_resume(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pilot',Path(__file__).resolve().parents[1]/'tools/run_xian_gt10m_pilot.py')
        pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'receipt.json'
            with self.assertRaises(ValueError): pilot.write_json(path,{'value':float('nan')})
            self.assertFalse(path.exists())
            pilot.write_json(path,{'value':1})
            before=path.read_bytes();pilot.write_json(path,{'value':1})
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(ValueError): pilot.write_json(path,{'value':2})

    def test_correlated_region_scenarios_move_whole_objects_and_not_each_pixel(self):
        rows={'a':box(2,2,4,4),'b':box(6,2,8,4)}
        scenarios=transfer.perturb(rows,[(0,0),(1,2)],{'a':(0,1)})
        self.assertEqual(scenarios[1]['a'].bounds,(3,5,5,7))
        self.assertEqual(scenarios[1]['b'].bounds,(7,4,9,6))

    def test_unknown_envelope_never_admits_despite_zero_stress_bandwidth(self):
        valid=np.ones((2,2)); band=np.zeros((2,2))
        self.assertEqual(transfer.training_mask(valid,band,envelope=None,temporal=True).sum(),0)
        self.assertEqual(transfer.training_mask(valid,band,envelope=2.,temporal=False).sum(),0)
        self.assertEqual(transfer.training_mask(valid,band,envelope=2.,temporal=True).sum(),4)

    def test_shadow_export_load_and_idempotence_cannot_write_protected_inputs(self):
        grid=Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'synthetic_shadow';root.mkdir()
            out=root/'test.tif'
            transfer.export_raster(out,np.array([[[.2]],[[.8]]]),grid,['roof','unknown'],root=root)
            self.assertEqual(transfer.load_rasters([out])['arrays'][0][:,0,0].tolist(),
                             [np.float32(.2),np.float32(.8)])
            before=out.read_bytes()
            transfer.export_raster(out,np.array([[[.2]],[[.8]]]),grid,['roof','unknown'],root=root)
            self.assertEqual(before,out.read_bytes())
            with self.assertRaises(ValueError):
                transfer.export_raster(out,np.array([[[.3]],[[.7]]]),grid,['roof','unknown'],root=root)
            with self.assertRaises(ValueError):
                transfer.export_raster(Path(temp)/'protected.tif',np.zeros((1,1,1)),grid,['roof'],root=root)
            with self.assertRaises(ValueError):
                transfer.validate_input_role('STUDENT_PREDICTION')

    def test_loader_rejects_misaligned_quality_even_with_same_epsg(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);g=Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649')
            transfer.export_raster(root/'a.tif',np.ones((1,1,1)),g,['a'],root=root)
            h=Grid(Affine(10,0,5,0,-10,10),1,1,'EPSG:32649')
            transfer.export_raster(root/'b.tif',np.ones((1,1,1)),h,['b'],root=root)
            with self.assertRaises(ValueError): transfer.load_rasters([root/'a.tif',root/'b.tif'])

    def test_raw_gt_bytes_remain_unchanged_and_wrong_band_shape_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            protected=Path(temp)/'manual_gt.bin';protected.write_bytes(b'protected reference')
            root=Path(temp)/'synthetic_shadow';root.mkdir()
            g=Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649')
            with self.assertRaises(ValueError):
                transfer.export_raster(protected,np.ones((1,1,1)),g,['a'],root=root)
            self.assertEqual(protected.read_bytes(),b'protected reference')
            with self.assertRaises(ValueError):
                transfer.export_raster(root/'bad.tif',np.ones((1,2,1)),g,['a'],root=root)


if __name__ == '__main__':
    unittest.main()
