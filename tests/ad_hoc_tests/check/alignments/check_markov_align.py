from os import listdir
from os.path import isfile, join, exists
import unittest
import pickle
import time
import matplotlib.pyplot as plt
import libraries.networkx_graph as networkx_graph
import networkx as nx
from libraries.networkx_graph import find_first_n_paths_from_vertex_pair, \
                                     is_path_possible_in_trans_system
from experiments.models.get_markov import get_markov_model, \
    find_first_n_paths_markov

from utils.converter.markov.markov_utils import are_markov_paths_possible_2, \
                                         change_markov_labels, \
                                         change_labels_all_dist
# from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa, \
#                                                       remove_empty_state
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_3 import convert_nfa_to_dfa_3, \
                                                      remove_empty_state
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_4 import convert_nfa_to_dfa_4
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_2 import convert_nfa_to_dfa_2
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter import tran_sys_to_nx_graph
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from tests.unit_tests.nfa_to_dfa.test_nfa_to_dfa import \
    find_first_n_paths_from_vertex_pair 
from utils.converter.all_dist import create_all_dist_dfa
from utils.converter.all_dist import create_all_dist_ts
from utils.converter.markov import markov_utils
from utils.converter.reach_graph_to_dfg import find_end_acts, \
                                               reach_graph_to_dfg_start_end_2, \
                                               transform_to_markov, \
                                               graph_to_test_paths, \
                                               consider_empty_trace, \
                                               accepts_empty_trace  
from utils.converter.dfa_to_dfg import dfa_to_dfg


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


def remove_null_align(pn_align):
    pn_new_align = {}

    for idx,a in enumerate(pn_align):
        pn_new_align[idx] = []

        for e in a['alignment']:
            if e[1] != None:
                pn_new_align[idx].append(e)

    return pn_new_align


def get_align_diff_idx(graph_align, pn_align, suffix):
    diff = set()
    pn_align = remove_null_align(pn_align)
    graph_align = remove_suffix_align(graph_align, suffix)

    for idx,a in pn_align.items():
        if len(a) != len(graph_align[idx]):
            diff.add(idx)
        else:
            for idx2 in range(len(a)):
                if pn_align[idx][idx2] != graph_align[idx][idx2]:
                    diff.add(idx)
                    break
    
    return diff


def remove_suffix_align(graph_align, suffix):
    graph_align_new = {}

    for idx,x in enumerate(graph_align):
        graph_align_new[idx] = []

        for e in x['alignment']:
            last_occur = e[1].rfind(suffix)

            if last_occur != -1:
                new_e = (e[0],e[1][:last_occur])
            else:
                new_e = e

            graph_align_new[idx].append(new_e)
    
    return graph_align_new



my_file = 'petri_nets/ETM/activitylog_uci_detailed_weekends.pnml'

my_log = 'xes_files/tests/test_align.xes'


suffix = '!#@#!'
log = xes_importer.apply(my_log)
net, im, fm = pnml_importer.apply(my_file)

gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.view(gviz)

print('##########')
print('Creating transition system...')
print('##########')
ts = reachability_graph.construct_reachability_graph(net, im)
# create_all_dist_ts(ts, suffix)

# gviz = visualizer.apply(ts)
# visualizer.view(gviz)

print('##########')
print('Creating DFA...')
print('##########')
Gd = convert_nfa_to_dfa_4(ts, 
                          init_state=None,
                          final_states=None,
                          include_empty_state=True)

Gr = reduceDFA(Gd, include_empty_state=False)
Gt = readable_copy(Gr)
Gt = create_all_dist_dfa(Gt, suffix)
edges_list = list(Gt.edges)

for e in edges_list:
    print('edge: ' + str(e) + ', activity: ' + str(Gt.edges[e]['activity']))

k = 1

print('##########')
print('Creating DFG...')
print('##########')


G, sa, ea = dfa_to_dfg(Gt, k)

# Test
G.remove_edge('watchingtv!#@#!2', 'shower!#@#!2')
G.remove_edge('eatingBreakfast!#@#!3', 'shower!#@#!2')

G2 = G.to_undirected()
components = list(nx.connected_components(G2))
idx = -1

if len(components[0]) > len(components[1]):
    idx = 0
else:
    idx = 1

for node in components[idx]:
    G.remove_node(node)

ea = {'watchingtv!#@#!2':1, 'eatingBreakfast!#@#!3':1}

H = G
# H = transform_to_markov(G, sa, ea, ts)
# H = consider_empty_trace(G, sa, ea, ts)

dfg = nx_graph_to_dfg(H)

gviz = dfg_visualization.apply(dfg)
dfg_visualization.view(gviz)

# print('changing Markov labels...')
# H, sa, ea = change_markov_labels(Gm, suffix)

accepts_empty = accepts_empty_trace(ts)

graph_align_value = 0

start = time.time()

print('aligning Markov model and log...')
graph_align = dfg_alignment.apply(log, 
                                    dfg, 
                                    sa, 
                                    ea, 
                                    variant=dfg_alignment.Variants.TEST2,
                                    parameters={'suffix':suffix,
                                                'accepts_empty':accepts_empty})

end = time.time()

print('total time: ' + str(end - start))

for t in graph_align:
    graph_align_value += t['fitness']

graph_align_value /= len(graph_align)

is_align_pn = True

print('graph_alignment_fit: ' + str(graph_align_value))

if is_align_pn:
    print('traditional alignment...')
    pn_align = pn_alignments.apply_log(log, net, im, fm)
    # my_path = 'temp/pn_align.txt'
    # pn_align = pickle.load(open(my_path, 'rb'))
    pn_align_value = 0

    for a in pn_align:
        pn_align_value += a['fitness']

    pn_align_value /= len(pn_align)

    print('pn_alignment_fit: ' + str(pn_align_value)) 

    # diff = get_align_diff_idx(graph_align, pn_align, suffix)
    # pn_align_act = remove_null_align(pn_align)
    # graph_align_act = remove_suffix_align(graph_align, suffix)

print()