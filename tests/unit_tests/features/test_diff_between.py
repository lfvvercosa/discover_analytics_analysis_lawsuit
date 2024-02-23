import unittest
import networkx as nx
from features.features import between_diff


class TestBetweennessFeature(unittest.TestCase):

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

        a = between_diff(G_model, G_log)
        b = round((1/2)/3 - (2/6+2/6)/4, 4)

        self.assertEqual(a,b)

    def test2(self):
        G_log = nx.DiGraph()
        G_log.add_edges_from([
                ('a','b'),
                ('a','c'),
                ('c','d'),
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('a','c'),
                    ('a','d'),
                    ('b','d'),
                    ('c','d'),
        ])

        a = between_diff(G_model, G_log)
        b = round(0 - 1/6/4, 4)

        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()