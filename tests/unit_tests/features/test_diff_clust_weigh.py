import unittest
import networkx as nx
from features.features import clust_diff_weighted


class TestDiffClusterWeighted(unittest.TestCase):

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
        G_log.add_weighted_edges_from([
                ('a','b',3),
                ('a','c',2),
                ('c','b',4),
                ('a','d',5),
        ])

        G_model = nx.Graph()
        G_model.add_edges_from([
                ('a','b'),
                ('a','c'),
                ('b','d'),
                ('a','d'),
        ])

        a = clust_diff_weighted(G_model, G_log)
        b = (1/3 * ((2*3*4)**(1. / 3))) - \
            (1/3 * ((5*3*(14/4))**(1. / 3)))

        self.assertEqual(a,b)


if __name__ == '__main__':
    unittest.main()