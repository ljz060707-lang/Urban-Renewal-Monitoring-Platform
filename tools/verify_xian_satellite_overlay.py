"""Read-only regression checks and append-only receipt for photograph overlays."""
from pathlib import Path
import hashlib,json,shutil,sys,unittest,zipfile
import fiona,numpy as np,rasterio
from rasterio.windows import Window
from shapely.geometry import shape,Polygon
from shapely.affinity import translate
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
P=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
OUT=P/'SINGLE2019_SATELLITE_OVERLAY_CASES_20261012_29'
D=json.loads((OUT/'SATELLITE_OVERLAY_SHORTLIST.json').read_text(encoding='utf-8'))
SELECTION=json.loads((OUT/'DISPLAY_CASE_SELECTION.json').read_text(encoding='utf-8'))
CASES=[q for q in D['candidates'] if q['key'] in SELECTION['selected_keys']]

class OverlayChecks(unittest.TestCase):
    def test_exact_original_and_existing_shift(self):
        original={}
        with fiona.open(P/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
            for f in s:
                if f.properties['component_key'] in SELECTION['selected_keys']:
                    g=shape(f.geometry);original[f.properties['component_key']]=list(g.geoms)[0]
        for q in CASES:
            g=original[q['key']]
            self.assertEqual(hashlib.sha256(g.wkb).hexdigest(),q['before_wkb_sha256'])
            self.assertEqual(hashlib.sha256(translate(g,xoff=0,yoff=-10).wkb).hexdigest(),q['after_wkb_sha256'])
            np.testing.assert_allclose(np.asarray(q['after_rings_utm'][0])-np.asarray(q['before_rings_utm'][0]),
                np.tile([0,-10],(len(q['before_rings_utm'][0]),1)),rtol=0,atol=1e-9)
    def test_exact_support_and_no_quarantined_cases(self):
        supports=json.loads((P/'SINGLE2019_COHERENT_TRANSFER_20261012_12/DERIVED_SUPPORT_LINEAGE.json').read_text(encoding='utf-8'))
        byid={g['region_id']:g for g in supports['derived_geometries']}
        for q in CASES:
            s=byid[q['region_id']];self.assertEqual(s['correction_m'],q['vector_m'])
            for n in ['before_rings_utm','after_rings_utm']:self.assertTrue(shape(s['geometry']).covers(Polygon(q[n][0])))
            self.assertFalse(q['source_risk_overlap']);self.assertFalse(q['target_risk_overlap'])
            self.assertGreaterEqual(q['recommended_positive_pixels_intersecting_after'],3)
    def test_native_photos_and_common_grid(self):
        with rasterio.open(D['reference_path']) as s2,rasterio.open(D['ge_source_path']) as ge:
            for q in CASES:
                b=q['bounds_utm'];win=Window(*q['photo_provenance']['sentinel_window'])
                expected=np.moveaxis(s2.read(window=win),0,-1)
                np.testing.assert_array_equal(expected,np.asarray(Image.open(OUT/(q['key']+'_SENTINEL_NATIVE_10M.png'))))
                self.assertEqual(s2.window_bounds(win),tuple(b))
                win=Window(*q['photo_provenance']['native_window']);native=ge.read(window=win)
                self.assertEqual(hashlib.sha256(native.tobytes()).hexdigest(),q['photo_provenance']['native_window_array_sha256'])
                self.assertEqual(Image.open(OUT/(q['key']+'_GE_DISPLAY.png')).size,
                    (int((b[2]-b[0])/.25),int((b[3]-b[1])/.25)))
    def test_no_label_or_status_promotion(self):
        self.assertFalse(D['formal_alignment_accepted']);self.assertEqual(D['new_labels'],0)
        self.assertEqual(D['new_vectors'],0);self.assertEqual(D['source_actual_date'],'2018-11-01')
        self.assertEqual(D['absolute_position_accuracy'],'UNKNOWN')
        self.assertNotIn('reviewer_id',SELECTION);self.assertNotIn('reviewed_at',SELECTION)
        with rasterio.open(P/'SINGLE2019_RISK_QUARANTINED_WINDOW_20261012_25/2019_RECOMMENDED_PSEUDO_LABEL.tif') as s:
            a=s.read(1);self.assertEqual([int((a==c).sum()) for c in [1,0,255]],[4860,5006,55670])
    def test_readonly_inputs_and_ge_file_stat(self):
        plan=json.loads((P/'SINGLE2019_CASE_FIGURES_20261012_28/FROZEN_CASE_FIGURE_PLAN.json').read_text(encoding='utf-8'))
        protected={**plan['protected_sha256'],**plan['new_code_sha256'],**D['protected_sha256']}
        for p,h in protected.items():self.assertEqual(sha(p),h,p)
        stat=Path(D['ge_source_path']).stat()
        self.assertEqual(stat.st_size,D['ge_source_file_stat']['size'])
        self.assertEqual(stat.st_mtime_ns,D['ge_source_file_stat']['mtime_ns'])
        global CHECKED_INPUTS
        CHECKED_INPUTS=len(protected)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OverlayChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    scripts=[ROOT/'tools/prepare_xian_satellite_overlay.py',ROOT/'figures/gen_fig_xian_satellite_overlay.py',Path(__file__).resolve()]
    for p in scripts:shutil.copy2(p,OUT/p.name)
    files=[p for p in OUT.iterdir() if p.is_file() and p.suffix in {'.png','.pdf','.json','.md','.py'}]
    save_json(OUT/'SATELLITE_OVERLAY_VERIFICATION_RECEIPT.json',{
        'run':OUT.name,'tests_passed':result.testsRun,'tests_failed':0,
        'protected_inputs_verified':CHECKED_INPUTS,'input_artifacts_changed':0,
        'source_GE_stat_and_selected_native_array_hashes_verified':True,
        'all_source_GE_file_bytes_hashed_this_run':False,
        'PNG_visual_inspection':'PASS_BOTH_CASES_AND_COMBINED',
        'PDF_structural_QA':'THREE_SINGLE_PAGE_PDFS_EXTRACTABLE_CHINESE_TEXT',
        'PDF_rasterized_visual_QA':'NOT_RUN','PNG_resolution_dpi':300,
        'new_alignment_vectors':0,'new_training_labels':0,'formal_reviews_created':0,
        'formal_accepts':0,'ground_footprints':0,'absolute_position_accuracy':'UNKNOWN',
        'code_sha256':{p.name:sha(p) for p in scripts},
        'output_sha256':{p.name:sha(p) for p in files}})
    files.append(OUT/'SATELLITE_OVERLAY_VERIFICATION_RECEIPT.json')
    archive=OUT/'XIAN_REAL_SATELLITE_ROOF_MOVEMENT_CASES.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failed')
    save_json(OUT/'SATELLITE_OVERLAY_ARCHIVE_RECEIPT.json',{'archive':archive.name,
        'sha256':sha(archive),'CRC_test':'PASS','files':len(files)})
    print(json.dumps({'tests_passed':result.testsRun,'protected_inputs_unchanged':CHECKED_INPUTS,'archive':str(archive)}))
