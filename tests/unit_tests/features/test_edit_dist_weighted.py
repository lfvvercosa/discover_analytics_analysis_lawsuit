import unittest
import networkx as nx
from features.features import \
    edit_distance_weighted
from utils.creation import creation_utils

class TestEditDistWeighted(unittest.TestCase):

    def test_edge_ins(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
                ('b','c',3),
                ('c','d',5),
                ('d','a',4),
                
        ])
        creation_utils.add_nodes_label(G_log)
        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),
        ])
        creation_utils.add_nodes_label(G_model)
        a = edit_distance_weighted(G_model, G_log)
        b = 4/14

        self.assertEqual(a,b)

    
    def test_edge_ins2(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
                ('b','c',3),
                ('c','d',5),
                ('d','a',4),
                
        ])
        creation_utils.add_nodes_label(G_log)
        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
        ])
        G_model.add_nodes_from(['c','d'])
        creation_utils.add_nodes_label(G_model)
        a = edit_distance_weighted(G_model, G_log)
        b = 12/14

        self.assertEqual(a,b)

    def test_edge_delet(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
                ('b','c',3),
                ('c','d',5),
                
        ])
        creation_utils.add_nodes_label(G_log)
        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),
                    ('d','a')
        ])
        creation_utils.add_nodes_label(G_model)
        a = edit_distance_weighted(G_model, G_log)
        b = 1/3

        self.assertEqual(a,b)

    
    def test_edge_delet2(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
                ('b','c',3),
                ('c','d',5),
                
        ])
        creation_utils.add_nodes_label(G_log)
        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),
                    ('d','a'),
                    ('c','b'),
        ])
        creation_utils.add_nodes_label(G_model)
        a = edit_distance_weighted(G_model, G_log)
        b = 2/3

        self.assertEqual(a,b)


    def test_edge_subst(self):
        print('a')


    def test_edge_subst(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
                ('a','d',3),
                ('d','c',4),
                
        ])
        creation_utils.add_nodes_label(G_log)
        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('a','d'),
                    ('b','c'),
        ])
        creation_utils.add_nodes_label(G_model)
        a = edit_distance_weighted(G_model, G_log)
        b = 1/3 + 4/9

        self.assertEqual(a,b)

    
    def test_node_ins(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
                ('b','c',3),
                ('a','d',4),
        ])
        creation_utils.add_nodes_label(G_log)

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('a','d'),
        ])
        creation_utils.add_nodes_label(G_model)

        a = edit_distance_weighted(G_model, G_log)
        b = 3/9 + 3/9

        self.assertEqual(a,b)


    def test_node_del(self):
        G_log = nx.DiGraph()
        G_log.add_weighted_edges_from([
                ('a','b',2),
        ])
        creation_utils.add_nodes_label(G_log)

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
        ])
        creation_utils.add_nodes_label(G_model)

        a = edit_distance_weighted(G_model, G_log)
        b = 1/2 + 1/1 + 2

        self.assertEqual(a,b)

if __name__ == '__main__':
    unittest.main()