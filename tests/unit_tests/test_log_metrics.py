import unittest
import networkx as nx
from simulations.dependency_graph.features.features import edit_distance
from simul_qual_metr.utils.get_filtered_log_graph import get_filtered_weighted_graph
from metrics import petri_net_metrics, networkx_graph
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer


class TestLogMetrics(unittest.TestCase):

    def test1(self):
        base_path = 'xes_files/real_processes/set_for_simulations/3/'
        log_name = '1a_VARA_DE_FEITOS_TRIBUTARIOS_DO_ESTADO_-_TJMG.xes'

        log = xes_importer.apply(base_path + log_name)
        alg = 'ETM'

        (G_log, filt_log) = get_filtered_weighted_graph(log_name, log, alg)

        self.assertNotEqual(len(log), len(filt_log))


    def test2(self):
        base_path = 'xes_files/real_processes/set_for_simulations/1/'
        log_name = 'edited_hh104_labour.xes.gz'

        log = xes_importer.apply(base_path + log_name)
        alg = 'ETM'

        (G_log, filt_log) = get_filtered_weighted_graph(log_name, log, alg)

        self.assertEqual(len(log), len(filt_log))


    def test3(self):
        base_path = 'xes_files/real_processes/set_for_simulations/1/'
        log_name = 'edited_hh104_labour.xes.gz'

        log = xes_importer.apply(base_path + log_name)
        alg = 'IMd'

        (G_log, filt_log) = get_filtered_weighted_graph(log_name, log, alg)

        self.assertNotEqual(len(log), len(filt_log))


    def test4(self):
        base_path = 'xes_files/real_processes/set_for_simulations/1/'
        log_name = 'edited_hh104_labour.xes.gz'

        log = xes_importer.apply(base_path + log_name)
        alg = 'IMd'

        (G_log, filt_log) = get_filtered_weighted_graph(log_name, log, alg)
        btw = networkx_graph.calcBetweenness(G_log, stat='median', normalized=True)
        btw_weighted = networkx_graph.calcBetweenness(G_log, stat='median', weight='weight', normalized=True)

        self.assertNotEqual(btw, btw_weighted)

    
    def test5(self):
        base_path = 'xes_files/real_processes/set_for_simulations/1/'
        log_name = 'edited_hh104_labour.xes.gz'

        log = xes_importer.apply(base_path + log_name)
        alg = 'IMd'

        (G_log, filt_log) = get_filtered_weighted_graph(log_name, log, alg)
        cc = networkx_graph.calcClustCoef(G_log, stat='max')
        cc_weighted = networkx_graph.calcClustCoef(G_log, stat='max', weight='weight')

        self.assertNotEqual(cc, cc_weighted)

if __name__ == '__main__':
    unittest.main()