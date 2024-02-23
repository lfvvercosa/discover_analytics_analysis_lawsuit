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
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_4 import convert_nfa_to_dfa_4                                               
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter import tran_sys_to_nx_graph
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from tests.unit_tests.nfa_to_dfa.test_nfa_to_dfa import \
    find_first_n_paths_from_vertex_pair 
from utils.converter.all_dist import create_all_dist_dfa
from utils.converter.all_dist import create_all_dist_ts, \
                                     create_name_all_dist_ts, \
                                     create_name_all_dist_ts_temp   
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


class CompareDFAOptimized(unittest.TestCase):
    def areGraphsEqual(self, Gm, G, t=-1):
        for n in Gm.nodes:
            if not G.has_node(n.name):
                if t == 7:
                    print('G2 doesnt have node: ' + str(n))
                    print('t = ' + str(t))

                    print('### G1 ###')

                    edges_list = list(Gm.edges)

                    for e in edges_list:
                        print('edge: ' + str(e) + ', activity: ' + \
                                str(Gm.edges[e]['activity']))

                    print('### G2 ###')

                    edges_list = list(G.edges)

                    for e in edges_list:
                        print('edge: ' + str(e) + ', activity: ' + \
                                str(G.edges[e]['activity']))

                return False
        
        for n in G.nodes:
            if not Gm.has_node(n.name):
                if t == 7:
                    print('G1 doesnt have node: ' + str(n))
                    print('t = ' + str(t))

                    print('### G1 ###')

                    edges_list = list(Gm.edges)

                    for e in edges_list:
                        print('edge: ' + str(e) + ', activity: ' + \
                                str(Gm.edges[e]['activity']))

                    print('### G2 ###')

                    edges_list = list(G.edges)

                    for e in edges_list:
                        print('edge: ' + str(e) + ', activity: ' + \
                                str(G.edges[e]['activity']))

                return False
        
        for e in Gm.edges:
            if not G.has_edge(e[0].name,e[1].name):
                print('2 doesnt have edge: ' + str(n))
                return False
        
        for e in G.edges:
            if not Gm.has_edge(e[0].name,e[1].name):
                print('G1 doesnt have edge: ' + str(n))
                return False

        return True


    def test1(self):
        print('### test 1 ###')
        my_file = 'petri_nets/tests/test_markov_and.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '!#@#!'
        k = 1
        
        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        # create_all_dist_ts(ts, suffix)

        # gviz = visualizer.apply(ts)
        # visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                  init_state=None,
                                  final_states=None,
                                  include_empty_state=True)

        # edges_list = list(Gd.edges)
        # print('### Gd ###')

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gd.edges[e]['activity']))

        Gd2 = convert_nfa_to_dfa_4(ts, 
                                   init_state=None,
                                   final_states=None,
                                   include_empty_state=True)

        # edges_list = list(Gd2.edges)
        # print('### Gd2 ###')

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gd2.edges[e]['activity']))

        print('### Test Equals (True) ###')
        print(self.areGraphsEqual(Gd, Gd2))
        print()
        self.assertTrue(self.areGraphsEqual(Gd, Gd2))
    

    def test2(self):
        print('### test 2 ###')
        my_file = 'petri_nets/IMd/'+\
                  'BPI_Challenge_2013_open_problems.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '!#@#!'
        k = 1
        
        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        # create_all_dist_ts(ts, suffix)

        gviz = visualizer.apply(ts)
        visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                  init_state=None,
                                  final_states=None,
                                  include_empty_state=True)

        # dict_edges = {}
        # edges_list = list(Gd.edges)
        # print('### Gd ###')

        # for e in edges_list:
        #     acts = Gd.edges[e]['activity']
        #     acts.sort()

        #     dict_edges[str(e)] = acts

        # for key in sorted(dict_edges):
        #     print("%s: %s" % (key, dict_edges[key]))

        print()

        Gd2 = convert_nfa_to_dfa_4(ts, 
                                   init_state=None,
                                   final_states=None,
                                   include_empty_state=True)

        dict_edges = {}
        edges_list = list(Gd2.edges)
        print('### Gd2 ###')

        for e in edges_list:
            acts = Gd2.edges[e]['activity']
            acts.sort()

            dict_edges[str(e)] = acts

        for key in sorted(dict_edges):
            print("%s: %s" % (key, dict_edges[key]))

        print()

        print('### Test Equals (False) ###')
        print(self.areGraphsEqual(Gd, Gd2))
        print()
        self.assertFalse(self.areGraphsEqual(Gd, Gd2))
    

    def test3(self):
        print('### test 3 ###')
        my_file = 'petri_nets/IMf/'+\
                  'ElectronicInvoicingENG.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '!#@#!'
        k = 1
        
        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        # create_all_dist_ts(ts, suffix)

        # gviz = visualizer.apply(ts)
        # visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                  init_state=None,
                                  final_states=None,
                                  include_empty_state=True)

        edges_list = list(Gd.edges)
        # print('### Gd ###')

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gd.edges[e]['activity']))

        Gd2 = convert_nfa_to_dfa_4(ts, 
                                   init_state=None,
                                   final_states=None,
                                   include_empty_state=True)

        edges_list = list(Gd2.edges)
        # print('### Gd2 ###')

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gd2.edges[e]['activity']))


        # self.assertTrue(self.areGraphsEqual(Gd, Gd2))
        print('### Test Equals (True) ###')
        print(self.areGraphsEqual(Gd, Gd2))
        print()
        self.assertTrue(self.areGraphsEqual(Gd, Gd2))


    def test4(self):
        print('### test 4 ###')
        my_file = 'petri_nets/IMf/'+\
                  'BPI_Challenge_2014_Reference alignment.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '!#@#!'
        k = 1
        
        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        # create_all_dist_ts(ts, suffix)

        # gviz = visualizer.apply(ts)
        # visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                  init_state=None,
                                  final_states=None,
                                  include_empty_state=True)
        # Gr = reduceDFA(Gd, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        edges_list = list(Gd.edges)
        # print('### Gd ###')

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gd.edges[e]['activity']))

        Gd2 = convert_nfa_to_dfa_4(ts, 
                                   init_state=None,
                                   final_states=None,
                                   include_empty_state=True)
        # Gr = reduceDFA(Gd2, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        # edges_list = list(Gd2.edges)
        # print('### Gd2 ###')

        # for e in edges_list:
        #     print('edge: ' + str(e) + ', activity: ' + \
        #           str(Gd2.edges[e]['activity']))

        print('### Test Equals (True) ###')
        print(self.areGraphsEqual(Gd, Gd2))
        print()
        self.assertTrue(self.areGraphsEqual(Gd, Gd2))
    

    def test5(self):
        print('### test 5 ###')
        my_file = 'petri_nets/IMf/'+\
                  'Sepsis Cases - Event Log.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '!#@#!'
        k = 1
        # debug_dict = {}
        
        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        # create_name_all_dist_ts_temp(ts)
        # create_all_dist_ts(ts, suffix)

        # gviz = visualizer.apply(ts)
        # visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                  init_state=None,
                                  final_states=None,
                                  include_empty_state=True)
        # Gr = reduceDFA(Gd, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        dict_edges = {}

        edges_list = list(Gd.edges)
        # print('### Gd ###')

        # for e in edges_list:
        #     acts = Gd.edges[e]['activity']
        #     acts.sort()

        #     dict_edges[str(e)] = acts

        # for key in sorted(dict_edges):
        #     print("%s: %s" % (key, dict_edges[key]))

        print()

        Gd2 = convert_nfa_to_dfa_4(ts, 
                                   init_state=None,
                                   final_states=None,
                                   include_empty_state=True)
        # Gr = reduceDFA(Gd2, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        dict_edges = {}

        # edges_list = list(Gd2.edges)
        # print('### Gd2 ###')

        # for e in edges_list:
        #     acts = Gd2.edges[e]['activity']
        #     acts.sort()

        #     dict_edges[str(e)] = acts

        # for key in sorted(dict_edges):
        #     print("%s: %s" % (key, dict_edges[key]))

        print('### Test Equals (True) ###')
        print(self.areGraphsEqual(Gd, Gd2))
        print()
        self.assertTrue(self.areGraphsEqual(Gd, Gd2))


    def test6(self):
        print('### test 6 ###')
        my_file = 'petri_nets/IMf/'+\
                  '3a_VARA_DE_EXECUCOES_FISCAIS_DA_COMARCA_DE_FORTALEZA_-_TJCE.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '!#@#!'
        k = 1
        # debug_dict = {}
        
        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        # create_name_all_dist_ts_temp(ts)
        # create_all_dist_ts(ts, suffix)

        # gviz = visualizer.apply(ts)
        # visualizer.view(gviz)

        Gd = convert_nfa_to_dfa_3(ts, 
                                  init_state=None,
                                  final_states=None,
                                  include_empty_state=True)
        # Gr = reduceDFA(Gd, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        dict_edges = {}

        edges_list = list(Gd.edges)
        # print('### Gd ###')

        # for e in edges_list:
        #     acts = Gd.edges[e]['activity']
        #     acts.sort()

        #     dict_edges[str(e)] = acts

        # for key in sorted(dict_edges):
        #     print("%s: %s" % (key, dict_edges[key]))

        print()

        Gd2 = convert_nfa_to_dfa_4(ts, 
                                   init_state=None,
                                   final_states=None,
                                   include_empty_state=True)
        # Gr = reduceDFA(Gd2, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        dict_edges = {}

        # edges_list = list(Gd2.edges)
        # print('### Gd2 ###')

        # for e in edges_list:
        #     acts = Gd2.edges[e]['activity']
        #     acts.sort()

        #     dict_edges[str(e)] = acts

        # for key in sorted(dict_edges):
        #     print("%s: %s" % (key, dict_edges[key]))

        print('### Test Equals (False) ###')
        print(self.areGraphsEqual(Gd, Gd2))
        print()
        self.assertFalse(self.areGraphsEqual(Gd, Gd2))


    def test7(self):
        print('### test 7 ###')
        my_file = 'petri_nets/ETM/'+\
                  'BPI_Challenge_2014_Control summary.pnml'
        net, im, fm = pnml_importer.apply(my_file)
        suffix = '!#@#!'
        k = 1
        # debug_dict = {}
        
        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)
        
        create_name_all_dist_ts_temp(ts)
        # create_all_dist_ts(ts, suffix)
        
        # gviz = visualizer.apply(ts)
        # visualizer.view(gviz)


        Gd = convert_nfa_to_dfa_3(ts, 
                                  init_state=None,
                                  final_states=None,
                                  include_empty_state=True)
        # Gr = reduceDFA(Gd, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        dict_edges = {}

        edges_list = list(Gd.edges)
        # print('### Gd ###')

        # for e in edges_list:
        #     acts = Gd.edges[e]['activity']
        #     acts.sort()

        #     dict_edges[str(e)] = acts

        # for key in sorted(dict_edges):
        #     print("%s: %s" % (key, dict_edges[key]))

        print()

        Gd2 = convert_nfa_to_dfa_4(ts, 
                                   init_state=None,
                                   final_states=None,
                                   include_empty_state=True)
        # Gr = reduceDFA(Gd2, include_empty_state=False)
        # Gt = readable_copy(Gr)
        # Gm = create_mk_abstraction_dfa_2(Gt, k=k)

        # dfg = nx_graph_to_dfg(Gm)
        # gviz = dfg_visualization.apply(dfg)
        # dfg_visualization.view(gviz)

        dict_edges = {}

        # edges_list = list(Gd2.edges)
        # print('### Gd2 ###')

        # for e in edges_list:
        #     acts = Gd2.edges[e]['activity']
        #     acts.sort()

        #     dict_edges[str(e)] = acts

        # for key in sorted(dict_edges):
        #     print("%s: %s" % (key, dict_edges[key]))

        print('### Test Equals (True) ###')
        print(self.areGraphsEqual(Gd, Gd2, 7))
        print()
        self.assertTrue(self.areGraphsEqual(Gd, Gd2, 7))
        print()

if __name__ == '__main__':
    unittest.main()
