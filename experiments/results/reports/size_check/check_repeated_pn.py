from os import listdir
from os.path import isfile, join
from utils.global_var import DEBUG
import pandas as pd
import multiprocessing
import time

from features import LogFeatures

from utils.converter.reach_graph_to_dfg import transform_to_markov
from utils.converter.dfa_to_dfg import dfa_to_dfg
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_4 import convert_nfa_to_dfa_4
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.markov.markov_utils import areMarkovGraphsEqual
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter.all_dist import create_all_dist_dfa
from utils.read_and_write.files_handle import remove_suffix
from utils.converter.markov.markov_utils import change_markov_labels
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg
from utils.creation.create_df_from_dict import df_from_dict
from utils.converter.reach_graph_to_dfg import reach_graph_to_dfg_start_end, \
                                               transform_to_markov, \
                                               accepts_empty_trace
from utils.converter.all_dist import removeSuffixAct

from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment


def createMarkovAlign(ts, k_markov, suffix, ret_dict):
    print('converting NFA to DFA...')
    Gd = convert_nfa_to_dfa_4(ts, 
                              init_state=None,
                              final_states=None,
                              include_empty_state=True)
    Gr = reduceDFA(Gd, include_empty_state=False)
    Gt = readable_copy(Gr)
    Gt = create_all_dist_dfa(Gt, suffix)
    
    print('creating Markov DFG...')
    G, sa, ea = dfa_to_dfg(Gt, k_markov)
    # H = transform_to_markov(G, sa, ea, ts)

    ret_dict['markov'] = (G, sa, ea)


def calcMarkovTimeout(ts, k_markov, suffix, timeout):
        manager = multiprocessing.Manager()
        ret_dict = manager.dict()

        p = multiprocessing.Process(target=createMarkovAlign,
                                    args=(
                                             ts, 
                                             k_markov, 
                                             suffix,
                                             ret_dict
                                          )
                                   )

        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
        
            return -1
    
        p.join()

        return ret_dict['markov']


def get_distinct_act_dfa(G, suffix):
    acts = {}

    for n in G.nodes:
        a = removeSuffixAct(n, suffix)

        if a not in acts:
            acts[a] = True
    

    return len(acts.keys())


def get_all_act_dfa(G):
    return len(G.nodes)


if __name__ == '__main__':
    base_path = 'xes_files/'
    base_path_pn = 'petri_nets/'
    k_markov = 1
    algs = ['IMf', 'IMd', 'ETM']
    folders = ['1/', '2/', '3/', '4/', '5/']
    out_path = 'experiments/results/reports/size_check/size_check.csv'
    
    ref_path = 'experiments/results/markov/k_1/df_markov_k_1.csv'
    count_rec = 0
    suffix = '!#@#!'
    timeout = 60*60

    cols={
        'EVENT_LOG':{},
        'DISCOVERY_ALG':{},
        'LOG_DISTINCT_ACT':{},
        'MODEL_DISTINCT_ACT':{},
        'PERCENT_MODEL_REPEAT_ACT':{},
        'PERCENT_ACT':{},
    }

    df_ref = pd.read_csv(ref_path, sep='\t')

    df_excl = None

    for fol in folders:
        my_path = base_path + fol
        my_files = [f for f in listdir(my_path) \
                    if isfile(join(my_path, f))]

        for fil in my_files:
            for alg in algs:
                record = (remove_suffix(fil),alg)

                if len(df_ref[
                    (df_ref[['EVENT_LOG','DISCOVERY_ALG']].values == \
                        record).all(axis=1)]) == 0:
                        print('Skipping record (not in df_markov_k1)...')
                        continue
                
                if (df_excl is not None) and len(df_excl[
                    (df_excl[['EVENT_LOG','DISCOVERY_ALG']].values == \
                        record).all(axis=1)]) == 1:
                        print('Skipping record (already calculated)...')
                        continue

                test = ('BPI_Challenge_2013_incidents','ETM')
                if record != test:
                    print('Skipping record due to test')
                    continue

                log = xes_importer.apply(my_path + fil)
                
                if DEBUG:
                    print('### Algorithm: ' + str(alg))
                    print('### Folder: ' + str(fol))
                    print('### Log: ' + str(fil))

                pn_file = 'petri_nets/' + alg + '/' + remove_suffix(fil) + '.pnml'
                net, im, fm = pnml_importer.apply(pn_file)
                
                start = time.time()
                
                print('creating transition system...')
                ts = reachability_graph.construct_reachability_graph(net, im)
                
                res = calcMarkovTimeout(ts, k_markov, suffix, timeout)

                if res == -1:
                    print('Skipping record (Timeout)...')
                    continue

                H, sa, ea = res

                dfg = nx_graph_to_dfg(H)
                
                # gviz = dfg_visualization.apply(dfg)
                # dfg_visualization.view(gviz)

                cols['EVENT_LOG'][count_rec] = remove_suffix(fil)
                cols['DISCOVERY_ALG'][count_rec] = alg
                cols['LOG_DISTINCT_ACT'][count_rec] = LogFeatures.number_events_types(log)
                cols['MODEL_DISTINCT_ACT'][count_rec] = get_distinct_act_dfa(H, suffix)
                cols['PERCENT_MODEL_REPEAT_ACT'][count_rec] =  \
                    round(get_all_act_dfa(H)/cols['MODEL_DISTINCT_ACT'][count_rec],2)
                cols['PERCENT_ACT'][count_rec] = \
                    round(cols['MODEL_DISTINCT_ACT'][count_rec]\
                        /cols['LOG_DISTINCT_ACT'][count_rec],2)

                df_from_dict(cols, out_path)

                count_rec += 1