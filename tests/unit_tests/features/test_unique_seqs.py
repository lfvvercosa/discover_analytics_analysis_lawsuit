import unittest
from features.LogFeatures import number_unique_seqs, \
                              percent_unique_seqs, \
                              number_sequences
from pm4py.objects.log.importer.xes import importer as xes_importer


class TestUniqueSequences(unittest.TestCase):
    def test1(self):
        log_path = 'xes_files/test/test_levenshtein.xes'
        log = xes_importer.apply(log_path)
        v = number_unique_seqs(log)
        v2 = percent_unique_seqs(log)

        self.assertEqual(v,2)
        self.assertEqual(v2,100)
    

    def test2(self):
        log_path = 'xes_files/5/Sepsis Cases - Event Log.xes.gz'
        log = xes_importer.apply(log_path)
        v = number_unique_seqs(log)
        p = percent_unique_seqs(log)

        self.assertEqual(v, 846)
        self.assertEqual(round(p,1), 80.6)


    def test3(self):
        log_path = 'xes_files/3/BPI_Challenge_2012.xes.gz'
        log = xes_importer.apply(log_path)
        v = number_unique_seqs(log)
        p = percent_unique_seqs(log)

        print(v)
        print(p)


if __name__ == '__main__':
    unittest.main()
