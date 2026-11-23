import unittest
import numpy as np
from xian_r5.alignment.image_block_replay import replay_check_blocks

class BlockReplayTests(unittest.TestCase):
    def fixture(self):
        a=np.random.default_rng(14).random((64,64,3))*255
        b=np.roll(a,(1,-1),axis=(0,1));m=np.zeros((64,64),bool);m[5:-5,5:-5]=True
        return a,b,m
    def test_correct_offset_multiblock(self):
        a,b,m=self.fixture();r=replay_check_blocks(a,b,m,m,np.ones(m.shape,bool),(-1,1))
        self.assertGreaterEqual(r['positive_blocks'],3)
        self.assertTrue(r['consistent_internal_experiment'])
    def test_zero_not_correction_evidence(self):
        a,_,m=self.fixture();r=replay_check_blocks(a,a,m,m,np.ones(m.shape,bool),(0,0))
        self.assertEqual(r['positive_blocks'],0)
        self.assertFalse(r['consistent_internal_experiment'])
    def test_wrong_offset_not_consistent(self):
        a,_,m=self.fixture();r=replay_check_blocks(a,a,m,m,np.ones(m.shape,bool),(1,1))
        self.assertFalse(r['consistent_internal_experiment'])
    def test_blocks_not_independent_buildings(self):
        a,b,m=self.fixture();r=replay_check_blocks(a,b,m,m,np.ones(m.shape,bool),(-1,1))
        self.assertFalse(r['independent_building_groups']);self.assertFalse(r['formal_fit_holdout_review'])
    def test_float_offset_rejected(self):
        a,b,m=self.fixture()
        with self.assertRaises(ValueError):replay_check_blocks(a,b,m,m,np.ones(m.shape,bool),(.5,1))

if __name__=='__main__':unittest.main()
