from features.fitness_feat import modify_markov_model
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2


pn_path = 'petri_nets/tests/test_feat_markov_ks_diff3.pnml'
# pn_path = 'petri_nets/IMf/BPI_Challenge_2014_Geo parcel document.pnml'
net, im, fm = pnml_importer.apply(pn_path)
ts = reachability_graph.construct_reachability_graph(net, im)
Gd = convert_nfa_to_dfa(ts)
Gr = reduceDFA(Gd)
Gr = readable_copy(Gr)
Gm = create_mk_abstraction_dfa_2(Gr, k=3)

print('###### Edges before ######')

edges_list = list(Gm.edges)
print('#### number edges: ' + str(len(edges_list)))

nodes_list = list(Gm.nodes)
print('#### number nodes: ' + str(len(nodes_list)))


for e in edges_list:
    # if e[1] == '-':
        print('edge Gm: ' + str(e))

Gm = modify_markov_model(Gm)

print('###### Edges after ######')

edges_list = list(Gm.edges)
print('#### number edges: ' + str(len(edges_list)))

nodes_list = list(Gm.nodes)
print('#### number nodes: ' + str(len(nodes_list)))
print('nodes: ' + str(nodes_list))

for e in edges_list:
    # if e[1] == '-':
        print('edge Gm: ' + str(e))
