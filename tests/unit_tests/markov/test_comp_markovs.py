import unittest
import time
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.importer import importer as pnml_importer
from utils.converter.markov.dfa_to_markov import create_mk_abstraction_dfa
from utils.converter.markov.create_markov_log import create_mk_abstraction_log
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import convert_nfa_to_dfa, \
                                                      remove_empty_state
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from experiments.log_filtering import most_frequent_and_random_filter
from features import features



class TestMarkovModel(unittest.TestCase):

    def test1(self):
        log_path = 'xes_files/1/' + \
                   'activitylog_uci_detailed_labour.xes.gz'
        log = xes_importer.apply(log_path)
        
        # model_path = 'petri_nets/IMf/activitylog_uci_detailed_labour.xes.gz.pnml'
        # net, im, fm = pnml_importer.apply(model_path)

        Gm_log = create_mk_abstraction_log(log, k=2)

        log = most_frequent_and_random_filter(log, 0.6, 0.6)
        net, im, fm = inductive_miner.apply(log)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        ts = reachability_graph.construct_reachability_graph(net, im)

        # gviz = ts_visualizer.apply(ts)
        # ts_visualizer.view(gviz)

        Gd = convert_nfa_to_dfa(ts, 
                                init_state=None,
                                final_states=None,
                                include_empty_state=True)
        Gd = readable_copy(Gd)
        Gr = reduceDFA(Gd, include_empty_state=False)        
        
        
        start = time.time()
        Gm_model = create_mk_abstraction_dfa(Gr, k=2)
        end = time.time()
        time_Gmr = end - start

        # edges_list = list(Gm_log.edges)

        # for e in edges_list:
        #     print('(Gm_log) edge: ' + str(e) + ', weight: ' + str(Gm_log.edges[e]['weight']))
        
        # edges_list = list(Gmr_model.edges)

        # for e in edges_list:
        #     print('(Gmr_model) edge: ' + str(e))

        print('time Gmr: ' + str(time_Gmr))

        count = 0
        count_nodes = 0

        for e in Gm_log.edges:
            if Gm_model.has_edge(e[0],e[1]):
                count += 1

        for n in Gm_log.nodes:
            if Gm_model.has_node(n):
                count_nodes += 1
        
        
        print('total edges Gm_log: ' + str(len(Gm_log.edges)))
        print('total edges Gm_model: ' + str(len(Gm_model.edges)))
        print('edges Gm_log contained in Gmr_model: ' + str(count))

        print('')
        print('')

        print('total nodes Gm_log: ' + str(len(Gm_log.nodes)))
        print('total nodes Gm_model: ' + str(len(Gm_model.nodes)))
        print('nodes Gm_log contained in Gmr_model: ' + str(count_nodes))

        dist_nodes = features.dist_nodes_percent(Gm_model, Gm_log)
        print('dist nodes: ' + str(dist_nodes))

        dist_edges = features.dist_edges_percent(Gm_model, Gm_log)
        print('dist edges: ' + str(dist_edges))




    def test2(self):
        print('test')








if __name__ == '__main__':
    unittest.main()
