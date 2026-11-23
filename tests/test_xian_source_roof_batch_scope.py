import unittest
from shapely.geometry import box
from xian_r5.source_roof_batch_scope import source_roof_batch_scope


class BatchScopeTests(unittest.TestCase):
    def fixture(self):
        regions={f'r{r}c{c}':box(c*100,r*100,(c+1)*100,(r+1)*100) for r in range(4) for c in range(4)}
        objects={key:box(g.bounds[0]+10,g.bounds[1]+10,g.bounds[0]+30,g.bounds[1]+30) for key,g in regions.items()}
        return objects,regions

    def test_complete_sixteen_not_subset(self):
        r=source_roof_batch_scope(*self.fixture());self.assertEqual(len(r['targets']),16);self.assertEqual(len(r['per_unit_targets']),16)

    def test_all_eligible_not_topk(self):
        args=self.fixture();args[0]['extra']=box(40,40,60,60)
        self.assertEqual(len(source_roof_batch_scope(*args)['targets']),17)

    def test_missing_unit_fail_closed(self):
        args=self.fixture();del args[0]['r0c0']
        with self.assertRaises(ValueError):source_roof_batch_scope(*args)

    def test_cross_unit_component_not_whole_instance(self):
        args=self.fixture();args[0]['cross']=box(90,10,110,30)
        r=source_roof_batch_scope(*args);self.assertEqual(r['excluded_components'][0]['reason'],'SOURCE_SEAM_OR_PARTIAL')

    def test_support_count_not_expanded(self):
        args=self.fixture();del args[1]['r0c0']
        with self.assertRaises(ValueError):source_roof_batch_scope(*args)

    def test_small_component_disclosed(self):
        args=self.fixture();args[0]['small']=box(2,2,5,5)
        r=source_roof_batch_scope(*args);self.assertEqual(r['excluded_components'][0]['reason'],'BELOW_SOURCE_PARENT_AREA')

    def test_weak_prior_not_truth_or_ground_tie(self):
        r=source_roof_batch_scope(*self.fixture())
        self.assertFalse(r['targets'][0]['source_model_mask_is_ground_truth']);self.assertEqual(r['registration_ties_generated'],0)

    def test_source_shapes_immutable(self):
        args=self.fixture();before=args[0]['r0c0'].wkb;source_roof_batch_scope(*args);self.assertEqual(before,args[0]['r0c0'].wkb)


if __name__=='__main__':unittest.main()
