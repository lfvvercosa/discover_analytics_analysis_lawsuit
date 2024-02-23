from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.statistics.traces.generic.log import case_statistics
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.markov.create_markov_log_2 import create_mk_abstraction_log_2
from utils.converter.markov.dfa_to_markov import create_mk_abstraction_dfa
from utils.converter.markov.create_markov_log import create_mk_abstraction_log
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.read_and_write.files_handle import remove_preffix
from experiments.models import get_markov
from features import fitness_feat
from features import precision_feat


def get_align_avg(aligned_traces):
    v = 0
                
    for a in aligned_traces:
        v += a['fitness']

    return v / len(aligned_traces)


if __name__ == '__main__':
    alg = 'IMd'
    # pn_path = 'petri_nets/' + alg + '/'+\
    #           'Sepsis Cases - Event Log.pnml'
    pn_path = 'petri_nets/IMf/Production_Data.pnml'
    # log_path = 'xes_files/5/'+\
    #           'Sepsis Cases - Event Log.xes.gz'
    log_path = 'xes_files/1/Production_Data.xes.gz'

    net, im, fm = pnml_importer.apply(pn_path)

    # gviz = pn_visualizer.apply(net, im, fm)
    # pn_visualizer.view(gviz)

    log = xes_importer.apply(log_path)
    variants_count = case_statistics.get_variant_statistics(log)


    # aligned_traces = alignments.apply_log(log, net, im, fm)
    
    # print('alignment: ' + str(get_align_avg(aligned_traces)))
    

    ts = reachability_graph.construct_reachability_graph(net, im)

    # gviz = ts_visualizer.apply(ts)
    # ts_visualizer.view(gviz)

    Gd = convert_nfa_to_dfa(ts)
    Gr = reduceDFA(Gd)
    Gr = readable_copy(Gr)

    edges_list = list(Gr.edges)

    for e in edges_list:
        print('edge: ' + str(e) + ', activity: ' + str(Gr.edges[e]['activity']))

    Gm = create_mk_abstraction_dfa_2(Gr, k=1)
    # Gm = get_markov.get_markov_model(remove_preffix(log_path), alg,k=2)
    Gl = create_mk_abstraction_log_2(log, k=1)

    edges_list = list(Gm.edges)

    for e in edges_list:
        print('edge Gm: ' + str(e))

    edges_list = list(Gl.edges)

    for e in edges_list:
        print('edge Gl: ' + str(e))

    # f = fitness_feat.edges_only_log_w(Gm, Gl)

    # print('fit cost: ' + str(f))

    p = precision_feat.edges_only_model_w(Gm, Gl)
    print('prec cost: ' + str(p))

    print('test')
