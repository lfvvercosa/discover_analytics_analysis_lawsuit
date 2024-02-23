import unittest
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from utils.converter.markov.dfa_to_markov import create_mk_abstraction_dfa
from utils.converter.markov.create_markov_log import create_mk_abstraction_log
from features import fitness_feat


class TestFeatureMarkov(unittest.TestCase):
    def test1(self):
        log_path = 'xes_files/test/test_feature_markov5.xes'
        log = xes_importer.apply(log_path)

        pn_path = 'petri_nets/tests/test_feature_markov5.pnml'
        net, im, fm = pnml_importer.apply(pn_path)

        avg_alignment = get_average_alignment(log, net, im, fm)

        print('average alignment: ' + str(avg_alignment))

        metric_markov = get_metric_markov(log, net, im, 1)
        metric_markov_2 = get_metric_markov(log, net, im, 2)
        metric_markov_3 = get_metric_markov(log, net, im, 3)

        print('metric markov (k=1): ' + str(metric_markov))
        print('metric markov (k=2): ' + str(metric_markov_2))
        print('metric markov (k=3): ' + str(metric_markov_3))

        print('')


def get_average_alignment(log, net, im, fm):
    aligned_traces = alignments.apply_log(log, net, im, fm)

    avg_alignment = 0

    for a in aligned_traces:
        avg_alignment += a['fitness']

    avg_alignment /= len(aligned_traces)
    avg_alignment = round(avg_alignment,2)

    return avg_alignment


def get_metric_markov(log, net, im, k):
    ts = reachability_graph.construct_reachability_graph(net, im)
    Gd = convert_nfa_to_dfa(ts)
    Gr = reduceDFA(Gd)
    Gm = create_mk_abstraction_dfa(Gr, k=k)
    Gl = create_mk_abstraction_log(log, k=k)

    return round(fitness_feat.edges_only_log_w(Gm, Gl),2)




if __name__ == '__main__':
    unittest.main()
