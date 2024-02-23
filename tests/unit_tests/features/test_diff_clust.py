import unittest
import networkx as nx
from features.features import clust_diff


class TestDistFeature(unittest.TestCase):

    # def test1(self):
    #     G_log = nx.DiGraph()
    #     G_log.add_edges_from([
    #             ('a','b'),
    #             ('b','c'),
    #             ('c','d'),
    #     ])

    #     G_model = nx.DiGraph()
    #     G_model.add_edges_from([
    #                 ('a','b'),
    #                 ('b','c'),
    #     ])

    #     a = clust_diff(G_model, G_log)
    #     b = 0

    #     self.assertEqual(a,b)

    def test2(self):
        G_log = nx.Graph()
        G_log.add_edges_from([
                ('a','b'),
                ('a','c'),
                ('c','d'),
        ])

        G_model = nx.Graph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('a','c'),
                    ('a','d'),
                    ('b','d'),
                    ('c','d'),
        ])

        a = clust_diff(G_model, G_log)
        b = round(10/12,4)

        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()