import unittest
from utils.converter.markov.create_markov_log_2 import create_mk_abstraction_log_2
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics

class TestMarkovModel(unittest.TestCase):
    def test1(self):
        log_path = 'xes_files/test/test_markov.xes'
        log = xes_importer.apply(log_path)

        Gm = create_mk_abstraction_log_2(log, k=1)

        self.assertEqual(Gm.edges['-', "['A']"]['weight'], 5)
        self.assertEqual(Gm.edges["['D']", "-"]['weight'], 5)
        self.assertEqual(Gm.edges["['B']", "['C']"]['weight'], 3)

        Gm = create_mk_abstraction_log_2(log, k=2)

        self.assertEqual(Gm.edges['-', "['-', 'A']"]['weight'], 5)
        self.assertEqual(Gm.edges["['D', '-']", "-"]['weight'], 5)
        self.assertEqual(Gm.edges["['-', 'A']", "['A', 'C']"]['weight'], 2)
        self.assertEqual(Gm.edges["['B', 'C']", "['C', 'D']"]['weight'], 3)

        Gm = create_mk_abstraction_log_2(log, k=3)

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', weight: ' + str(Gm.edges[e]['weight']))

        self.assertEqual(Gm.edges["['-', '-', 'A']", "['-', 'A', 'B']"]['weight'], 3)
        self.assertEqual(Gm.edges["['D', '-', '-']", "-"]['weight'], 5)
        self.assertEqual(Gm.edges["['C', 'D', '-']", "['D', '-', '-']"]['weight'], 5)
        self.assertEqual(Gm.edges["['-', 'A', 'C']", "['A', 'C', 'D']"]['weight'], 2)


    def test2(self):
        log_path = 'xes_files/test/test_markov2.xes'
        log = xes_importer.apply(log_path)

        Gm = create_mk_abstraction_log_2(log, k=3)

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', weight: ' + str(Gm.edges[e]['weight']))

        self.assertEqual(Gm.edges["['-', '-', 'A']", "['-', 'A', 'B']"]['weight'], 6)
        self.assertEqual(Gm.edges["['-', 'A', 'B']", "['A', 'B', '-']"]['weight'], 3)
        self.assertEqual(Gm.edges["['B', 'D', 'D']", "['D', 'D', 'A']"]['weight'], 2)
        self.assertEqual(Gm.edges["['B', '-', '-']", "-"]['weight'], 3)


    def test3(self):    
        log_path = 'xes_files/1/Production_Data.xes.gz'
        log = xes_importer.apply(log_path)

        Gm = create_mk_abstraction_log_2(log, k=2)

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', weight: ' + str(Gm.edges[e]['weight']))

        self.assertTrue(Gm.has_edge(
            "['Flat Grinding - Machine 11', 'Flat Grinding - Machine 11']",
            "['Flat Grinding - Machine 11', 'Laser Marking - Machine 7']"
        ))
        self.assertTrue(Gm.has_edge(
            "['-', 'Turning & Milling - Machine 6']",
            "['Turning & Milling - Machine 6', 'Turning & Milling - Machine 6']"
        ))
        self.assertTrue(Gm.has_edge(
            "['Packing', 'Packing']",
            "['Packing', '-']"
        ))
        self.assertTrue(Gm.has_edge(
            "['Packing', '-']",
            "-"
        ))
        

if __name__ == '__main__':
    unittest.main()
