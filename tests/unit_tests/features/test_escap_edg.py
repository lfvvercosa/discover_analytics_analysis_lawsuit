import unittest
import networkx as nx
from features.features import escaping_edges_model
from utils.creation import creation_utils


class TestEscapingEdgesModel(unittest.TestCase):

    def test1(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
                ('b','c',3),
        ])
        creation_utils.add_nodes_label(G_log)
        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('b','d'),
        ])
        creation_utils.add_nodes_label(G_model)
        a = escaping_edges_model(G_model, G_log)
        b = 2/5

        self.assertEqual(a,b)
    

    def test2(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',300),
                ('b','c',300),
                ('f','b',200),
                ('b','j',200),

        ])
        creation_utils.add_nodes_label(G_log)
        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('b','d'),
                    ('b','e'),
        ])
        creation_utils.add_nodes_label(G_model)
        a = escaping_edges_model(G_model, G_log)
        b = 2 * (500/1000)

        self.assertEqual(a,b)




if __name__ == '__main__':
    unittest.main()