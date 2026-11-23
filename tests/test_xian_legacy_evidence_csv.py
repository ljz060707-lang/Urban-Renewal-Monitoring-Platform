import tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_xian_instance_evidence import selected_csv


class LegacyCsvTests(unittest.TestCase):
    def parse(self,text,fields=('uid','value')):
        with tempfile.TemporaryDirectory(prefix='xian_legacy_csv_test_') as directory:
            path=Path(directory)/'fixture.csv';path.write_text(text,encoding='utf-8')
            return selected_csv(path,fields,lambda row:row['uid']=='wanted')

    def test_complete_select_only_explicit_uid(self):
        r=self.parse('uid,value\nwanted,1\nother,2\n')
        self.assertEqual(r['rows_scanned'],2);self.assertEqual(r['selected_rows'],[{'uid':'wanted','value':'1'}])
        self.assertEqual(r['csv_structural_status'],'PASS')

    def test_last_truncated_row_disclosed_not_completion(self):
        r=self.parse('uid,value\nwanted,1\nother\n')
        self.assertEqual(len(r['invalid_rows']),1);self.assertEqual(r['invalid_rows'][0]['line'],3)
        self.assertEqual(r['csv_structural_status'],'PARTIAL_INVALID_ROWS_DISCLOSED')

    def test_extra_field_not_silent(self):
        r=self.parse('uid,value\nwanted,1,extra\n')
        self.assertEqual(r['selected_rows'],[]);self.assertEqual(len(r['invalid_rows']),1)

    def test_missing_schema_fail_closed(self):
        with self.assertRaises(ValueError):self.parse('uid\nwanted\n')

    def test_duplicate_header_fail_closed(self):
        with self.assertRaises(ValueError):self.parse('uid,value,value\nwanted,1,2\n')

    def test_valid_observed_zero_not_unknown_rows(self):
        r=self.parse('uid,value\nother,1\n')
        self.assertEqual(r['selected_rows'],[]);self.assertEqual(r['rows_scanned'],1)


if __name__=='__main__':unittest.main()
