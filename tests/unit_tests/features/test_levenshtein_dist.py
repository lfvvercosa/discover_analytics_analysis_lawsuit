import unittest
from features.LogFeatures import log_edit_distance
from pm4py.objects.log.importer.xes import importer as xes_importer


class TestLevenshteinDist(unittest.TestCase):
    def test1(self):
        log_path = 'xes_files/test/test_levenshtein.xes'
        log = xes_importer.apply(log_path)
        v = log_edit_distance(log)

        self.assertTrue(v == 1)


    def test2(self):
        log_path = 'xes_files/test/test_levenshtein2.xes'
        log = xes_importer.apply(log_path)
        v = log_edit_distance(log)

        self.assertTrue(v == 0.6)


    def test3(self):
        log_path = 'xes_files/test/test_levenshtein3.xes'
        log = xes_importer.apply(log_path)
        v = log_edit_distance(log)

        self.assertTrue(v == 1.2857)

    
    def test4(self):
        log_path = 'xes_files/5/Sepsis Cases - Event Log.xes.gz'
        log = xes_importer.apply(log_path)
        v = log_edit_distance(log)
        v = round(v,2)

        print('v: ' + str(v))


if __name__ == '__main__':
    unittest.main()