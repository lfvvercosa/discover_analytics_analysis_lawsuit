import unittest
import networkx as nx
from features.precision_feat import \
    exceding_edges_model_weighted
from features.aux_feat import \
    func_degree,func_void
from features import aux_feat
from features.aux_feat import \
    activities_occurrence, calc_diff_func
from utils.converter.dfg_to_nx_graph import dfg_to_nx_graph
from pm4py.objects.log.importer.xes import importer as xes_importer



class TestExcedEdgeWeight(unittest.TestCase):

    def test1(self):
        log_path = 'xes_files/test/log6.xes'
        log = xes_importer.apply(log_path)

        Gl = dfg_to_nx_graph(log)

        Gm = nx.DiGraph()
        Gm.add_edges_from([
            ('a','b'),
            ('a','c'),
            ('b','d'),
            ('c','d'),
        ])

        act_occur = activities_occurrence(log)
        cost = exceding_edges_model_weighted(Gm, 
                                             Gl, 
                                             act_occur,
                                             aux_feat.func_degree)

        self.assertEqual(0.14881, cost)


    # def test2(self):
    #     log_path = 'xes_files/test/log7.xes'
    #     log = xes_importer.apply(log_path)

    #     Gl = dfg_to_nx_graph(log)

    #     Gm = nx.DiGraph()
    #     Gm.add_edges_from([
    #         ('a','b'),
    #         ('a','c'),
    #         ('b','e'),
    #         ('b','d'),
    #         ('b','c'),
    #         ('c','d'),
    #         ('e','d'),
    #     ])

    #     act_occur = activities_occurrence(log)
    #     cost = exceding_edges_model_weighted(Gm, 
    #                                          Gl, 
    #                                          act_occur,
    #                                          features_aux.func_degree)

    #     self.assertEqual(0.035151, cost)

    # def test3(self):
    #     log_path = 'xes_files/test/log6.xes'
    #     log = xes_importer.apply(log_path)

    #     Gl = dfg_to_nx_graph(log)

    #     Gm = nx.DiGraph()
    #     Gm.add_edges_from([
    #         ('a','b'),
    #         ('b','d'),
    #         ('c','d'),
    #     ])

    #     act_occur = activities_occurrence(log)
    #     cost = exceding_edges_model_weighted(Gm, 
    #                                          Gl, 
    #                                          act_occur,
    #                                          nx.betweenness_centrality)

    #     self.assertEqual(0, cost)


    # def test4(self):
    #     log_path = 'xes_files/test/log8.xes'
    #     log = xes_importer.apply(log_path)

    #     Gl = dfg_to_nx_graph(log)

    #     Gm = nx.DiGraph()
    #     Gm.add_edges_from([
    #         ('a','b'),
    #         ('a','d'),
    #         ('b','d'),
    #         ('d','c'),
    #     ])

    #     act_occur = activities_occurrence(log)
    #     cost = exceding_edges_model_weighted(Gm, 
    #                                          Gl, 
    #                                          act_occur,
    #                                          nx.betweenness_centrality)

    #     self.assertEqual(0.099537, cost)


    def test5(self):
        log_path = 'xes_files/test/log8.xes'
        log = xes_importer.apply(log_path)

        Gl = dfg_to_nx_graph(log)

        Gm = nx.DiGraph()
        Gm.add_edges_from([
            ('a','b'),
            ('a','d'),
            ('b','d'),
            ('d','c'),
        ])
        weight_val = calc_diff_func(Gm, Gl, func_degree)
        expected = {'a':0, 'b':0, 'c':1, 'd':1}
        self.assertEqual(expected, weight_val)


    def test6(self):
        log_path = 'xes_files/test/log8.xes'
        log = xes_importer.apply(log_path)

        Gl = dfg_to_nx_graph(log)

        Gm = nx.DiGraph()
        Gm.add_edges_from([
            ('a','b'),
            ('a','d'),
            ('b','d'),
            ('d','c'),
        ])
        weight_val = calc_diff_func(Gm, Gl, func_void)
        expected = {}
        self.assertEqual(expected, weight_val)
    

    # def test7(self):



if __name__ == '__main__':
    unittest.main()