import unittest
from utils.converter.markov.create_markov_log import create_mk_abstraction_log
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics

class TestMarkovModel(unittest.TestCase):
    def test1(self):
        log_path = 'xes_files/test/test_markov.xes'
        log = xes_importer.apply(log_path)

        # variants_count = case_statistics.get_variant_statistics(log)
        
        Gm = create_mk_abstraction_log(log, k=2)

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', weight: ' + str(Gm.edges[e]['weight']))

        self.assertEqual(Gm.edges["['cc', 'dd']",'-']['weight'], 3)
        self.assertEqual(Gm.edges['-', "['aa', 'bb']"]['weight'], 7)
        self.assertEqual(Gm.edges["['bb', 'dd']", "['dd', 'dd']"]['weight'], 2)

        Gm = create_mk_abstraction_log(log, k=3)

        # edges_list = list(Gm.edges)

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', weight: ' + str(Gm.edges[e]['weight']))

        self.assertEqual(Gm.edges["['bb', 'dd', 'dd']","['dd', 'dd', 'aa']"]['weight'], 2)
        self.assertEqual(Gm.edges["['dd', 'dd', 'aa']","-"]['weight'], 2)


    def test2(self):
        log_path = 'xes_files/test/ElectronicInvoicingENG_Mapped.xes.gz'
        log = xes_importer.apply(log_path)

        variants_count = case_statistics.get_variant_statistics(log)

        Gm = create_mk_abstraction_log(log, k=2)

        total = 10722 + 4230 + 2625 + 2552 + 4 + 1 +1
        self.assertEqual(Gm.edges["['B', 'B']","['B', 'C']"]['weight'], total)
        self.assertEqual(Gm.edges["['E', 'E']","['E', 'D']"]['weight'], 2552)

        total2 = 2625*2 + 1 + 1 + 2552 + 10722
        self.assertEqual(Gm.edges["['E', 'F']","['F', 'F']"]['weight'], total2)
        self.assertEqual(Gm.edges["['D', 'D']","['D', 'I']"]['weight'], 4230)
        self.assertEqual(Gm.edges["['J', 'J']","['J', 'I']"]['weight'], total - 4230)
        self.assertEqual(Gm.edges["['J', 'I']","-"]['weight'], total - 4230)

        Gm = create_mk_abstraction_log(log, k=1)
        self.assertEqual(Gm.edges["['A']","['B']"]['weight'], total)
        self.assertEqual(Gm.edges["['D']","['I']"]['weight'], 4230)

        total2 = 10722 + 2625*2 + 2552 + 4 + 2 + 2
        self.assertEqual(Gm.edges["['E']","['F']"]['weight'], total2)

        Gm = create_mk_abstraction_log(log, k=3)

        self.assertEqual(Gm.edges["['E', 'D', 'D']","['D', 'D', 'E']"]['weight'], 2552)
        self.assertEqual(Gm.edges["['E', 'D', 'D']","['D', 'D', 'E']"]['weight'], 2552)
        self.assertEqual(Gm.edges["['E', 'F', 'H']","['F', 'H', 'H']"]['weight'], 5)
        self.assertEqual(Gm.edges["['E', 'E', 'F']","['E', 'F', 'H']"]['weight'], 5)


        for v in variants_count:
            print('variant: ' + str(v['variant']) + ', count: ' + str(v['count']))
        
        print('test')    


    def test3(self):
        log_path = 'xes_files/2/ElectronicInvoicingENG.xes.gz'
        log = xes_importer.apply(log_path)

        Gm = create_mk_abstraction_log(log, k=2)

        total2 = 2625*2 + 1 + 1 + 2552 + 10722
        self.assertEqual(Gm.edges["['Liquidation', 'Approve Liquidated Invoices']",
                                  "['Approve Liquidated Invoices', 'Approve Liquidated Invoices']"]['weight'], total2)
        

if __name__ == '__main__':
    unittest.main()
