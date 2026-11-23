import unittest
from xian_r5.single_date_identity import adjudicate,validate_links,REQUIRED_SOURCE_GATES


class SingleDateSourceLinkTests(unittest.TestCase):
    def evidence(self):
        return {**dict.fromkeys(REQUIRED_SOURCE_GATES,True),'native_exact_object_count':1,
                'multiple_contributors':False,'source_conflicts':[],'geometric_plausible':True}

    def test_complete_source_chain_only_closes_single_date(self):
        r=adjudicate(self.evidence())
        self.assertEqual(r['link_level'],'TRACEABLE_SINGLE_DATE_LINK')
        self.assertIsNone(r['physical_id'])
        self.assertFalse(r['cross_date_physical_identity_confirmed'])

    def test_every_required_source_gate_fails_closed(self):
        for gate in REQUIRED_SOURCE_GATES:
            with self.subTest(gate=gate):
                e=self.evidence();e[gate]=False
                self.assertNotEqual(adjudicate(e)['link_level'],'TRACEABLE_SINGLE_DATE_LINK')

    def test_high_geometric_overlap_is_not_proof(self):
        r=adjudicate({'geometric_plausible':True,'native_pixels_exact':True,'native_exact_object_count':1})
        self.assertEqual(r['adjudication'],'PLAUSIBLE_GEOMETRIC_LINK')

    def test_date_conflict_prevents_link(self):
        e=self.evidence();e['source_conflicts']=['DATE_CONFLICT']
        self.assertEqual(adjudicate(e)['adjudication'],'SOURCE_CONFLICT')

    def test_competitors_not_merged_even_if_booleans_inconsistent(self):
        e=self.evidence();e['multiple_contributors']=True
        self.assertEqual(adjudicate(e)['adjudication'],'COMPETING_LINKS')

    def test_two_exact_objects_are_ambiguous(self):
        e=self.evidence();e['native_exact_object_count']=2
        self.assertEqual(adjudicate(e)['adjudication'],'COMPETING_LINKS')

    def test_missing_exact_count_is_not_one(self):
        e=self.evidence();del e['native_exact_object_count']
        self.assertNotEqual(adjudicate(e)['link_level'],'TRACEABLE_SINGLE_DATE_LINK')

    def test_string_true_is_not_evidence(self):
        e=self.evidence();e['date_and_source_hash_verified']='True'
        self.assertNotEqual(adjudicate(e)['link_level'],'TRACEABLE_SINGLE_DATE_LINK')

    def test_old_track_id_cannot_fill_physical_id(self):
        row={'observation_id':'O','building_object_id':'B','physical_id':'TRK_R4X_01'}
        with self.assertRaises(ValueError):validate_links([row],{'O'},{'B'})

    def test_original_ids_only(self):
        row={'observation_id':'NEW_O','building_object_id':'B','physical_id':None}
        with self.assertRaises(ValueError):validate_links([row],{'O'},{'B'})

    def test_no_source_proof_no_formal_link(self):
        row={'observation_id':'O','building_object_id':'B','physical_id':None,'link_level':'TRACEABLE_SINGLE_DATE_LINK','evidence':{}}
        with self.assertRaises(ValueError):validate_links([row],{'O'},{'B'})

    def test_one_to_many_rows_preserved(self):
        rows=[{'observation_id':'O','building_object_id':b,'physical_id':None,'link_level':'CANDIDATE_ONLY'} for b in ('B1','B2')]
        validate_links(rows,{'O'},{'B1','B2'})
        self.assertEqual([r['building_object_id'] for r in rows],['B1','B2'])

    def test_many_to_one_rows_preserved(self):
        rows=[{'observation_id':o,'building_object_id':'B','physical_id':None} for o in ('O1','O2')]
        validate_links(rows,{'O1','O2'},{'B'})
        self.assertEqual(len(rows),2)

    def test_gt_geometry_never_substituted(self):
        row={'observation_id':'O','building_object_id':'B','physical_id':None,'GT_USED_FOR_GEOMETRY':True}
        with self.assertRaises(ValueError):validate_links([row],{'O'},{'B'})


if __name__=='__main__':unittest.main()
