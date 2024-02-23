import unittest
import networkx as nx
from features.features import entropy_diff


class TestEntropyFeature(unittest.TestCase):

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

        a = entropy_diff(G_model, G_log)
        b = -0.0817

        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()