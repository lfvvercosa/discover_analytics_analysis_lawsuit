import unittest
import networkx as nx
from utils.converter.nfa_to_dfa.Vertex import Vertex
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph


class TestVertex(unittest.TestCase):

    def test1(self):
        my_file = 'petri_nets/IMf/' + \
          'BPI_Challenge_2014_Control summary.xes.gz.pnml'
        
        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)
        states = set(ts.states)

        v = Vertex(set(ts.states), [])

        self.assertListEqual(list(states), list(v.states))


    def test2(self):
        my_file = 'petri_nets/IMf/' + \
          'BPI_Challenge_2014_Control summary.xes.gz.pnml'
        
        net, im, fm = pnml_importer.apply(my_file)
        ts = reachability_graph.construct_reachability_graph(net, im)
        states = set(ts.states)

        v = Vertex(states, [])
        s = states.pop()
        v2 = Vertex(states, [])

        G = nx.DiGraph()
        G.add_edge(v, v2)

        v.states.remove(s)
        self.assertCountEqual(v.states, v2.states)


if __name__ == '__main__':
    unittest.main()