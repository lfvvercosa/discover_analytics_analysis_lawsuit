from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_4 import convert_nfa_to_dfa_4
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter.all_dist import create_all_dist_dfa
from utils.converter.dfa_to_dfg import dfa_to_dfg
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg
from utils.converter.reach_graph_to_dfg import accepts_empty_trace



if __name__ == '__main__':
    # my_file = 'petri_nets/IMf/Production_Data.pnml'
    # my_log = 'xes_files/1/Production_Data.xes.gz'

    my_log = "/home/vercosa/Insync/doutorado/artigos/artigo_alignment/event_logs/tests/edited_hh104_labour.xes"
    my_file = "/home/vercosa/Insync/doutorado/artigos/artigo_alignment/event_logs/tests/edited_hh104_labour.pnml"

    suffix = '!#@#!'
    k_markov = 1

    print('loading log and Petri-net...')

    log = xes_importer.apply(my_log)
    net, im, fm = pnml_importer.apply(my_file)

    # gviz = pn_visualizer.apply(net, im, fm)
    # pn_visualizer.view(gviz)

    print('creating transition system...')

    ts = reachability_graph.construct_reachability_graph(net, im)

    # gviz = ts_visualizer.apply(ts)
    # ts_visualizer.view(gviz)

    print('creating DFA...')

    Gd = convert_nfa_to_dfa_4(ts, 
                              init_state=None,
                              final_states=None,
                              include_empty_state=True)

    print('reducing DFA...')

    Gr = reduceDFA(Gd, include_empty_state=False)

    # print('renaming states...')

    # Gt = readable_copy(Gr)

    print('adding suffix to the edges...')

    Gt = create_all_dist_dfa(Gr, suffix)

    print('creating DFG-TS...')

    G, sa, ea = dfa_to_dfg(Gt, k_markov)

    print('transforming to Counter object...')

    dfg = nx_graph_to_dfg(G)
    accepts_empty = accepts_empty_trace(ts)

    print('aligning log and DFG-TS...')

    graph_align = dfg_alignment.apply(log, 
                                    dfg, 
                                    sa, 
                                    ea, 
                                    variant=dfg_alignment.Variants.TEST,
                                    parameters={'suffix':suffix,
                                                'accepts_empty':accepts_empty,
                                                'debug':False})
    
    graph_align_value = 0

    for t in graph_align:
        graph_align_value += t['fitness']

    graph_align_value /= len(graph_align)

    print('approximated alignment mean value: ' + str(graph_align_value))
