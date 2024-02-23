# check error in smaller ETM model

import pandas as pd
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_4 import convert_nfa_to_dfa_4
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter.all_dist import create_all_dist_dfa
from utils.converter.dfa_to_dfg import dfa_to_dfg


def get_size_ts_graph(event_log):
    suffix = '!#@#!'
    net, im, fm = pnml_importer.apply(event_log)

    ts = reachability_graph.construct_reachability_graph(net, im)
    Gd = convert_nfa_to_dfa_4(ts, 
                              init_state=None,
                              final_states=None,
                              include_empty_state=True)
    Gr = reduceDFA(Gd, include_empty_state=False)
    Gt = readable_copy(Gr)
    Gt = create_all_dist_dfa(Gt, suffix)

    k = 1

    G, sa, ea = dfa_to_dfg(Gt, k)

    return len(G)


error_etm = 'experiments/results/reports/' + \
                  'comp_approaches/df_error_etm.csv'                 

df = pd.read_csv(error_etm, sep='\t')
df = df[[
    'EVENT_LOG',
    'DISCOVERY_ALG',
    'Proposta2',
]]

df = df[df['Proposta2'] > 0]
smaller = float('inf')
smaller_event_log = None

for i in range(len(df.index)):
    event_log = df.iloc[i]['EVENT_LOG']
    my_path = 'petri_nets/ETM/' + event_log + '.pnml'
    size = get_size_ts_graph(my_path)

    if size < smaller:
        smaller = size
        smaller_event_log = event_log

        print('size: ' + str(smaller))
        print('smaller ETM: ' + str(smaller_event_log))
        print()

