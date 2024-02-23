import unittest
import networkx as nx
from simul_qual_metr.features.features_aux import activities_occurrence
from pm4py.objects.log.importer.xes import importer as xes_importer


class TestActivityOccurrence(unittest.TestCase):
    def test1(self):
        log_path = 'simul_qual_metr/tests/dfg_precision/tests/log1.xes'
        log = xes_importer.apply(log_path)
        dic = activities_occurrence(log)
        expected = {'a':5, 'b':5, 'd':5} 
        self.assertEqual(dic,expected)


    def test2(self):
        log_path = 'simul_qual_metr/tests/dfg_precision/tests/log5.xes'
        log = xes_importer.apply(log_path)
        dic = activities_occurrence(log)
        expected = {'a':12, 'b':5, 'd':10, 'c':2} 
        self.assertEqual(dic,expected)


if __name__ == '__main__':
    unittest.main()