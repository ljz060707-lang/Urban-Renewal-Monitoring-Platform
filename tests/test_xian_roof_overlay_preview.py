"""Pixel-display transport safety, not roof accuracy validation."""
import unittest
import numpy as np
from affine import Affine
from shapely.geometry import box
from xian_r5.label_transfer import Grid
from xian_r5.source_partition import SourceRegion
from xian_r5.alignment.roof_overlay_preview import render_roof_overlay_preview

class RoofOverlayPreviewTests(unittest.TestCase):
    def setup_grid(self):return Grid(Affine(10,0,0,0,-10,80),8,8,'EPSG:32649')
    def region(self,key,bounds,v=None):
        return SourceRegion(key,'2018-11-01',box(*bounds),v,'a'*64 if v is not None else None)
    def test_east_shift_has_correct_sign(self):
        m=np.zeros((8,8),bool);m[3,2]=True
        r=render_roof_overlay_preview(m,self.setup_grid(),[self.region('a',(0,0,80,80),(10,0))],seam_width_px=0)
        self.assertTrue(r['candidate_foreground'][3,3]);self.assertFalse(r['candidate_foreground'][3,2])
        self.assertFalse(r['formal_correction_accepted'])
    def test_source_clipped_before_translation(self):
        m=np.zeros((8,8),bool);m[3,5]=True
        r=render_roof_overlay_preview(m,self.setup_grid(),[self.region('a',(0,0,40,80),(-20,0))],seam_width_px=0)
        self.assertFalse(r['candidate_foreground'].any())
    def test_unknown_has_no_alpha_and_no_filled_vector(self):
        r=render_roof_overlay_preview(np.ones((8,8),bool),self.setup_grid(),[self.region('a',(0,0,80,80))])
        self.assertFalse(r['candidate_rgba'][3].any())
        self.assertTrue((r['quality']==2).all())
    def test_zero_shift_is_still_unverified(self):
        m=np.zeros((8,8),bool);m[3,3]=True
        r=render_roof_overlay_preview(m,self.setup_grid(),[self.region('a',(0,0,80,80),(0,0))],seam_width_px=0)
        self.assertTrue(np.array_equal(m,r['candidate_foreground']))
        self.assertTrue((r['quality']==1).all());self.assertFalse(r['training_use_allowed'])
    def test_no_wraparound_at_destination_edge(self):
        m=np.zeros((8,8),bool);m[3,7]=True
        r=render_roof_overlay_preview(m,self.setup_grid(),[self.region('a',(0,0,80,80),(10,0))],seam_width_px=0)
        self.assertFalse(r['candidate_foreground'].any());self.assertFalse(r['candidate_rgba'][3,:,0].any())
    def test_cell_seams_are_hidden_and_flagged(self):
        r=render_roof_overlay_preview(np.ones((8,8),bool),self.setup_grid(),[self.region('a',(0,0,80,80),(10,0))])
        self.assertTrue((r['quality'][:,0]==3).all());self.assertFalse(r['candidate_rgba'][3,:,0].any())
    def test_overlapping_or_duplicate_regions_fail_closed(self):
        for rs in ([self.region('a',(0,0,50,80)),self.region('z',(30,0,80,80))],
                   [self.region('a',(0,0,40,80)),self.region('a',(40,0,80,80))]):
            with self.assertRaises(ValueError):render_roof_overlay_preview(np.zeros((8,8),bool),self.setup_grid(),rs)
    def test_subpixel_vector_not_rounded_into_label_preview(self):
        with self.assertRaises(ValueError):render_roof_overlay_preview(np.zeros((8,8),bool),self.setup_grid(),[self.region('a',(0,0,80,80),(2.5,0))])
    def test_masked_unknowns_or_nonboolean_foreground_rejected(self):
        for m in (np.ma.array(np.zeros((8,8),bool)),np.zeros((8,8),np.uint8)):
            with self.assertRaises(ValueError):render_roof_overlay_preview(m,self.setup_grid(),[])
    def test_foreground_is_not_background_or_training_GT(self):
        r=render_roof_overlay_preview(np.zeros((8,8),bool),self.setup_grid(),[self.region('a',(0,0,80,80),(0,0))])
        self.assertEqual(r['zero_semantics'],'NO_POSITIVE_ENVELOPE_NOT_KNOWN_BACKGROUND')
        self.assertEqual(r['absolute_position_accuracy'],'UNKNOWN')

if __name__=='__main__':unittest.main()
