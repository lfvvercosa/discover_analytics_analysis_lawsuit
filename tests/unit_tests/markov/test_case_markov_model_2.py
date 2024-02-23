from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
import time


my_file = 'petri_nets/IMf/'+\
    'Receipt phase of an environmental permit application process (_WABO_) CoSeLoG project.pnml'

net, im, fm = pnml_importer.apply(my_file)

gviz = pn_visualizer.apply(net, 
                           im, 
                           fm)
pn_visualizer.view(gviz) 

ts = reachability_graph.construct_reachability_graph(net, im)

gviz = ts_visualizer.apply(ts)
ts_visualizer.view(gviz)

Gd = convert_nfa_to_dfa(ts, 
                        init_state=None,
                        final_states=None,
                        include_empty_state=True)


Gr = reduceDFA(Gd, include_empty_state=False)
Gr = readable_copy(Gr)

edges_list = list(Gr.edges)

for e in edges_list:
    print('edge: ' + str(e) + ', activity: ' + str(Gr.edges[e]['activity']))

start = time.time()
Gm = create_mk_abstraction_dfa_2(Gr, k=2)
end = time.time()

print('nodes Markov: ' + str(len(Gm.nodes)))
print('edges Markov: ' + str(len(Gm.edges)))

print('time: ' + str(end-start))