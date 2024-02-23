from pm4py.objects.log.importer.xes import importer as xes_importer
from features.aux_feat import activities_occurrence_2
import unittest


class TestActOccur(unittest.TestCase):

    def test1(self):
        log_path = 'xes_files/test/test_feat_markov_ks_diff6_5.xes'
        log = xes_importer.apply(log_path)

        occur = activities_occurrence_2(log)

        # print(occur)
    
    def test2(self):
        log_path = 'xes_files/test/test_levenshtein3.xes'
        log = xes_importer.apply(log_path)

        occur = activities_occurrence_2(log)

        print(occur)

    
if __name__ == '__main__':
    unittest.main()
