import unittest
import networkx as nx
from simul_qual_metr.features.features_fitness import dist_edges_weighted
from simul_qual_metr.features.features_fitness import dist_edges_double_weighted
from simul_qual_metr.features import features_aux
from pm4py.objects.log.importer.xes import importer as xes_importer
from utils.converter.dfg_to_nx_graph import dfg_to_nx_graph


class TestDistEdgesWeightedFitness(unittest.TestCase):

    def test1(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',3),
                ('b','c',2),
                ('c','d',1),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
        ])

        a = dist_edges_weighted(G_model, G_log)
        b = 1/6

        self.assertEqual(a,b)

    def test2(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',3),
                ('b','c',2),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),
        ])

        a = dist_edges_weighted(G_model, G_log)
        b = 0

        self.assertEqual(a,b)
    

    def test3(self):
        log_path = 'simul_qual_metr/tests/logs/fitness/log1.xes'
        log = xes_importer.apply(log_path)

        Gl = dfg_to_nx_graph(log)

        Gm = nx.DiGraph()
        Gm.add_edges_from([
            ('a','b'),
            ('b','d'),
            ('a','c'),
        ])

        cost = dist_edges_double_weighted(Gm, Gl, features_aux.func_degree)
        expected = 0.28125

        self.assertEqual(expected, cost)

    
    # def test4(self):
    #     log_path = 'simul_qual_metr/tests/logs/fitness/log2.xes'
    #     log = xes_importer.apply(log_path)

    #     Gl = dfg_to_nx_graph(log)

    #     Gm = nx.DiGraph()
    #     Gm.add_edges_from([
    #         ('a','b'),
    #         ('b','d'),
    #         ('b','e'),
    #         ('e','d'),
    #     ])

    #     cost = dist_edges_double_weighted(Gm, Gl, features_aux.func_degree)
    #     expected = 0.090909

    #     self.assertEqual(expected, cost)


    def test5(self):
        log_path = 'simul_qual_metr/tests/logs/fitness/log1.xes'
        log = xes_importer.apply(log_path)

        Gl = dfg_to_nx_graph(log)

        Gm = nx.DiGraph()
        Gm.add_edges_from([
            ('a','b'),
            ('b','d'),
            ('a','c'),
            ('c','d'),
            ('d','e'),
        ])

        cost = dist_edges_double_weighted(Gm, Gl, features_aux.func_degree)
        expected = 0

        self.assertEqual(expected, cost)


if __name__ == '__main__':
    unittest.main()