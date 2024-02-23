import unittest
import networkx as nx
from features.precision_feat import \
    exceding_edges_model_weighted
from features.aux_feat import \
    func_degree,func_void
from features import aux_feat
from features.aux_feat import \
    activities_occurrence, calc_diff_func
from features import precision_feat
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.objects.petri_net.importer import importer as pnml_importer
from utils.converter.dfg_to_nx_graph import dfg_to_nx_graph
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.markov.create_markov_log_2 import create_mk_abstraction_log_2


class TestExcedEdgeWeight(unittest.TestCase):

    def test1(self):
        pn_path = 'petri_nets/tests/test_feature_markov3.pnml'
        log_path = 'xes_files/tests/test_feature_markov3_3.xes'
        k_markov = 2

        log = xes_importer.apply(log_path)
        net, im, fm = pnml_importer.apply(pn_path)

        ts = reachability_graph.construct_reachability_graph(net, im)
        Gd = convert_nfa_to_dfa(ts)
        Gr = reduceDFA(Gd)
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertAlmostEqual(p, 0.74, 2)

        k_markov = 1
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertEqual(p, 0.875)

        k_markov = 3
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertAlmostEqual(p, 0.62, 2)

    def test2(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5_2.xes'
        k_markov = 2

        log = xes_importer.apply(log_path)
        net, im, fm = pnml_importer.apply(pn_path)

        ts = reachability_graph.construct_reachability_graph(net, im)
        Gd = convert_nfa_to_dfa(ts)
        Gr = reduceDFA(Gd)
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        edges_list = list(Gm.edges)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertEqual(p, 0.6)

        k_markov = 1
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertEqual(p, 0.625)

        k_markov = 3
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertAlmostEqual(p, 0.58, 2)


    def test3(self):
        pn_path = 'petri_nets/tests/test_feat_markov_ks_diff6.pnml'
        log_path = 'xes_files/tests/test_feat_markov_ks_diff5_4.xes'
        k_markov = 2

        log = xes_importer.apply(log_path)
        net, im, fm = pnml_importer.apply(pn_path)

        ts = reachability_graph.construct_reachability_graph(net, im)
        Gd = convert_nfa_to_dfa(ts)
        Gr = reduceDFA(Gd)
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        edges_list = list(Gm.edges)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertEqual(p, 1)

        k_markov = 1
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertEqual(p, 1)

        k_markov = 3
        Gm = create_mk_abstraction_dfa_2(Gr, k=k_markov)
        Gl = create_mk_abstraction_log_2(log, k=k_markov)

        p = precision_feat.edges_only_model_w_2(Gm, Gl, log, k_markov)
        # print('prec cost: ' + str(p))

        self.assertEqual(p, 1)


if __name__ == '__main__':
    unittest.main()