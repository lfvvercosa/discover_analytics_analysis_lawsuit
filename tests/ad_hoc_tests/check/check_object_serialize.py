import pickle
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from utils.converter import tran_sys_to_nx_graph
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa
from tests.unit_tests.nfa_to_dfa.test_nfa_to_dfa import \
    find_paths_from_vertex_pair


my_file = 'petri_nets/IMf/' + \
              'BPI_Challenge_2014_Entitlement application.xes.gz.pnml'

net, im, fm = pnml_importer.apply(my_file)
ts = reachability_graph.construct_reachability_graph(net, im)

G = tran_sys_to_nx_graph.convert(ts)
# paths_G = find_paths_from_vertex_pair(G, '{source1}', '{sink1}')

Gd = convert_nfa_to_dfa(ts, include_empty_state=True)
paths_Gd = find_paths_from_vertex_pair(G, '{source1}', '{sink1}')

pickle.dump(G, open('tests/unit_tests/nfa_to_dfa/graphs/G.txt', 'wb'))
G = pickle.load(open('tests/unit_tests/nfa_to_dfa/graphs/G.txt', 'rb'))
paths_G = find_paths_from_vertex_pair(G, '{source1}', '{sink1}')

print(paths_G)