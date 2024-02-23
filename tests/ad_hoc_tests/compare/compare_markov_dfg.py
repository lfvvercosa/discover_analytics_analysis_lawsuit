from os import listdir
from os.path import isfile, join, exists
import time
import unittest
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import libraries.networkx_graph as networkx_graph
from libraries.networkx_graph import find_first_n_paths_from_vertex_pair, \
                                     is_path_possible_in_trans_system
from experiments.models.get_markov import get_markov_model, \
    find_first_n_paths_markov

from utils.converter.markov.markov_utils import are_markov_paths_possible_2, \
                                         change_markov_labels, \
                                         accepts_empty_trace, \
                                         change_labels_all_dist
# from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa, \
#                                                       remove_empty_state
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_3 import convert_nfa_to_dfa_3, \
                                                      remove_empty_state
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter import tran_sys_to_nx_graph
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from tests.unit_tests.nfa_to_dfa.test_nfa_to_dfa import \
    find_first_n_paths_from_vertex_pair 
from utils.converter.all_dist import create_all_dist_dfa
from utils.converter.all_dist import create_all_dist_ts
from utils.converter.reach_graph_to_dfg import find_end_acts, \
                                               reach_graph_to_dfg, \
                                               find_start_acts, \
                                                reach_graph_to_dfg_start_end_2, \
                                               reach_graph_to_dfg_start_end, \
                                               transform_to_markov, \
                                               graph_to_test_paths

from utils.converter.markov import markov_utils
from libraries.networkx_graph import find_first_n_paths_from_vertex_pair
from utils.converter.markov.markov_utils import are_markov_paths_possible_2

from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from pm4py.algo.conformance.alignments.petri_net import algorithm as pn_alignments
from pm4py.visualization.transition_system import visualizer

import unittest


class CompareMarkovDFG(unittest.TestCase):
    
    def areGraphsEqual(self,Gm,G):
        for n in Gm.nodes:
            if not G.has_node(n):
                return False
        
        for n in G.nodes:
            if not Gm.has_node(n):
                return False
        
        for e in Gm.edges:
            if not G.has_edge(e[0],e[1]):
                return False
        
        for e in G.edges:
            if not Gm.has_edge(e[0],e[1]):
                return False

        return True


    # def test1(self):
    #     my_file = 'petri_nets/tests/test_markov_and.pnml'
    #     net, im, fm = pnml_importer.apply(my_file)
    #     suffix = '_'
    #     k = 1
    #     n = 50

    #     # gviz = pn_visualizer.apply(net, im, fm)
    #     # pn_visualizer.view(gviz)

    #     ts = reachability_graph.construct_reachability_graph(net, im)
    #     # create_all_dist_ts(ts, suffix)

    #     # gviz = visualizer.apply(ts)
    #     # visualizer.view(gviz)
        
    #     Gm, sa, ea = markov_utils.create_markov_from_pn(ts, k)
        
    #     dfg = nx_graph_to_dfg(Gm)
        
    #     gviz = dfg_visualization.apply(dfg)
    #     dfg_visualization.view(gviz)

    #     G, sa, ea = reach_graph_to_dfg_start_end(ts)
    #     G = transform_to_markov(G, sa, ea, ts)
    #     Gt = graph_to_test_paths(G)

    #     dfg2 = nx_graph_to_dfg(G)

    #     # gviz = dfg_visualization.apply(dfg2)
    #     # dfg_visualization.view(gviz)

    #     TS = tran_sys_to_nx_graph.convert(ts)

    #     paths_G = find_first_n_paths_from_vertex_pair(TS, 
    #                                                   v1=None, 
    #                                                   v2=None,
    #                                                   n=n)
    #     print(are_markov_paths_possible_2(Gm=Gt, paths=paths_G, k=1))

    #     print()


    # def test2(self):
    #     my_file = 'petri_nets/IMd/' + \
    #       'edited_hh104_labour.pnml'
    #     net, im, fm = pnml_importer.apply(my_file)
    #     suffix = '!#@#!'
    #     k = 1
        
    #     # gviz = pn_visualizer.apply(net, im, fm)
    #     # pn_visualizer.view(gviz)

    #     ts = reachability_graph.construct_reachability_graph(net, im)
    #     create_all_dist_ts(ts, suffix)

    #     gviz = visualizer.apply(ts)
    #     visualizer.view(gviz)
        
    #     Gm, sa, ea = markov_utils.create_markov_from_pn(ts, k)
        
    #     dfg = nx_graph_to_dfg(Gm)
        
    #     # gviz = dfg_visualization.apply(dfg)
    #     # dfg_visualization.view(gviz)

    #     G, sa, ea = reach_graph_to_dfg_start_end(ts)
    #     G = transform_to_markov(G, sa, ea, ts)

    #     dfg2 = nx_graph_to_dfg(G)
        
    #     print(self.areGraphsEqual(Gm,G))
    #     print()
    #     # gviz = dfg_visualization.apply(dfg2)
    #     # dfg_visualization.view(gviz)

    
    # def test3(self):
    #     my_file = 'petri_nets/IMd/'+\
    #       'BPI_Challenge_2013_open_problems.pnml'
    #     net, im, fm = pnml_importer.apply(my_file)
    #     suffix = '!#@#!'
    #     k = 1
        
    #     # gviz = pn_visualizer.apply(net, im, fm)
    #     # pn_visualizer.view(gviz)

    #     ts = reachability_graph.construct_reachability_graph(net, im)
    #     create_all_dist_ts(ts, suffix)

    #     # gviz = visualizer.apply(ts)
    #     # visualizer.view(gviz)
        
    #     Gm, sa, ea = markov_utils.create_markov_from_pn(ts, k)
        
    #     dfg = nx_graph_to_dfg(Gm)
        
    #     # gviz = dfg_visualization.apply(dfg)
    #     # dfg_visualization.view(gviz)

    #     G, sa, ea = reach_graph_to_dfg_start_end(ts)
    #     G = transform_to_markov(G, sa, ea, ts)

    #     dfg2 = nx_graph_to_dfg(G)
        
    #     # gviz = dfg_visualization.apply(dfg2)
    #     # dfg_visualization.view(gviz)


    # def test4(self):
    #     my_file = 'petri_nets/IMd/'+\
    #       'edited_hh104_labour.pnml'
    #     net, im, fm = pnml_importer.apply(my_file)
    #     suffix = '!#@#!'
    #     k = 1
    #     n=5000
        
    #     # gviz = pn_visualizer.apply(net, im, fm)
    #     # pn_visualizer.view(gviz)

    #     ts = reachability_graph.construct_reachability_graph(net, im)
    #     create_all_dist_ts(ts, suffix)

    #     # gviz = visualizer.apply(ts)
    #     # visualizer.view(gviz)
        
    #     print('create markov...')
    #     start = time.time()
    #     Gm, sa, ea = markov_utils.create_markov_from_pn(ts, k)
    #     Gm = pickle.load(open('temp/test.txt', 'rb'))
    #     end = time.time()

    #     print('time markov: ' + str(end-start))

    #     # dfg = nx_graph_to_dfg(Gm)
        
    #     # gviz = dfg_visualization.apply(dfg)
    #     # dfg_visualization.view(gviz)
        
    #     print('create DFG...')
    #     start = time.time()
    #     G, sa, ea = reach_graph_to_dfg_start_end(ts)
    #     G = transform_to_markov(G, sa, ea, ts)
    #     end = time.time()
    #     print('time dfg: ' + str(end-start))

    #     print('create DFG2...')
    #     start = time.time()
    #     G2, sa, ea = reach_graph_to_dfg_start_end_2(ts)
    #     G2 = transform_to_markov(G2, sa, ea, ts)
    #     end = time.time()
    #     print('time dfg: ' + str(end-start))

    #     self.areGraphsEqual(Gm, G)

    #     TS = tran_sys_to_nx_graph.convert(ts)
    #     Gt = graph_to_test_paths(G2)

    #     paths_G = find_first_n_paths_from_vertex_pair(TS, 
    #                                                   v1=None, 
    #                                                   v2=None,
    #                                                   n=n)
        
    #     print('###### TEST PATHS ######')
    #     print(are_markov_paths_possible_2(Gm=Gt, paths=paths_G, k=1))
    #     print('###### TEST PATHS ######')

    #     Gmt = graph_to_test_paths(Gm)

    #     paths_G = find_first_n_paths_from_vertex_pair(TS, 
    #                                                   v1=None, 
    #                                                   v2=None,
    #                                                   n=n)
        
    #     print('###### TEST PATHS MARKOV ######')
    #     print(are_markov_paths_possible_2(Gm=Gmt, paths=paths_G, k=1))
    #     print('###### TEST PATHS MARKOV ######')

    #     # dfg2 = nx_graph_to_dfg(G)
        
    #     # gviz = dfg_visualization.apply(dfg2)
    #     # dfg_visualization.view(gviz)

    #     # self.areGraphsEqual(Gm,G)

    #     print()


    def test5(self):
        my_file = 'petri_nets/ETM/BPI_Challenge_2014_Department control parcels.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '_'
        k = 1
        n = 50

        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        # create_all_dist_ts(ts, suffix)

        gviz = visualizer.apply(ts)
        visualizer.view(gviz)
        
        Gm, sa, ea = markov_utils.create_markov_from_pn(ts, k)
        
        dfg = nx_graph_to_dfg(Gm)
        
        gviz = dfg_visualization.apply(dfg)
        dfg_visualization.view(gviz)

        G, sa, ea = reach_graph_to_dfg_start_end(ts)
        G = transform_to_markov(G, sa, ea, ts)
        Gt = graph_to_test_paths(G)

        dfg2 = nx_graph_to_dfg(G)

        # gviz = dfg_visualization.apply(dfg2)
        # dfg_visualization.view(gviz)

        TS = tran_sys_to_nx_graph.convert(ts)

        paths_G = find_first_n_paths_from_vertex_pair(TS, 
                                                      v1=None, 
                                                      v2=None,
                                                      n=n)
        print(are_markov_paths_possible_2(Gm=Gt, paths=paths_G, k=1))

        print()


if __name__ == '__main__':
    unittest.main()
