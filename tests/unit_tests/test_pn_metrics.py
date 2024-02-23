import unittest
import networkx as nx
from simulations.dependency_graph.features.features import edit_distance
from metrics import petri_net_metrics
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.importer import importer as pnml_importer


class TestPetriNetMetrics(unittest.TestCase):

    def test1(self):
        my_file = 'simul_qual_metr/petri_nets/'+\
            'BPI_Challenge_2014_Reference alignment.xes.gz.pnml'
        net, im, fm = pnml_importer.apply(my_file)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        count = petri_net_metrics.countInvisibleTransitions(net)
        expected = 5

        self.assertEqual(expected,count)
    

    def test2(self):
        my_file = 'simul_qual_metr/petri_nets/'+\
            'BPI_Challenge_2014_Reference alignment.xes.gz.pnml'
        net, im, fm = pnml_importer.apply(my_file)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        count = petri_net_metrics.countTransitions(net)
        expected = 3

        self.assertEqual(expected,count)


    def test3(self):
        my_file = 'simul_qual_metr/petri_nets/'+\
            'BPI_Challenge_2014_Reference alignment.xes.gz.pnml'
        net, im, fm = pnml_importer.apply(my_file)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        count = petri_net_metrics.countPlaces(net)
        expected = 7

        self.assertEqual(expected,count)

    
    def test4(self):
        my_file = 'simul_qual_metr/petri_nets/'+\
            'BPI_Challenge_2014_Reference alignment.xes.gz.pnml'
        net, im, fm = pnml_importer.apply(my_file)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        count = petri_net_metrics.countArcs(net, stat='mean')
        expected = round(16/7,4)

        self.assertEqual(expected,count)


    def test5(self):
        my_file = 'simul_qual_metr/petri_nets/'+\
            'BPI_Challenge_2014_Reference alignment.xes.gz.pnml'
        net, im, fm = pnml_importer.apply(my_file)

        # gviz = pn_visualizer.apply(net, im, fm)
        # pn_visualizer.view(gviz)

        count = petri_net_metrics.countArcs(net, stat='max')
        expected = 3

        self.assertEqual(expected,count)

if __name__ == '__main__':
    unittest.main()