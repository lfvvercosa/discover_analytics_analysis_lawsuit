import unittest
import networkx as nx
from features.features import out_degree_diff


class TestInDegreeFeature(unittest.TestCase):

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

        a = out_degree_diff(G_model, G_log)
        b = round(2/3 - 3/4, 4)

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

        a = out_degree_diff(G_model, G_log)
        b = round(5/4 - 3/4, 4)

        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()