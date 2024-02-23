import unittest
import networkx as nx
from features.features import dist_nodes_percent


class TestDistFeature(unittest.TestCase):

    def test1(self):
        G_log = nx.DiGraph()
        G_log.add_edges_from([
                ('a','b'),
                ('b','c'),
                ('c','d'),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
        ])

        a = dist_nodes_percent(G_model, G_log)
        b = 1/4

        self.assertEqual(a,b)


    def test2(self):
        G_log = nx.DiGraph()
        G_log.add_edges_from([
                ('a','b'),
                ('b','c'),
                ('c','d'),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),
        ])

        a = dist_nodes_percent(G_model, G_log)
        b = 0

        self.assertEqual(a,b)


    def test3(self):
        G_log = nx.DiGraph()
        G_log.add_edges_from([
                ('a','b'),
                ('b','c'),
                ('c','d'),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),
                    ('e','f'),
                    ('g','h'),
        ])

        a = dist_nodes_percent(G_model, G_log)
        b = 4/8

        self.assertEqual(a,b)


    def test4(self):
        G_log = nx.DiGraph()
        G_log.add_edges_from([
                ('a','b'),
                ('b','c'),
                ('c','d'),
                ('e','f'),
                ('g','h'),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),         
        ])

        a = dist_nodes_percent(G_model, G_log)
        b = 4/8

        self.assertEqual(a,b)


    def test5(self):
        G_log = nx.DiGraph()
        G_log.add_edges_from([
                ('a','b'),
                ('b','c'),
                ('c','d'),
                ('e','f'),
                ('g','h'),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'), 
                    ('c','i'),        
        ])

        a = dist_nodes_percent(G_model, G_log)
        b = 5/9

        self.assertEqual(a,b)

    

if __name__ == '__main__':
    unittest.main()