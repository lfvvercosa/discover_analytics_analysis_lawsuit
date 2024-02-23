import unittest
import networkx as nx
from features.features import edit_distance


class TestDistEdges(unittest.TestCase):

    def test1(self):
        G_log = nx.DiGraph()
        G_log.add_edges_from([
                ('a','b'),
                ('b','c'),
                ('c','d'),
                ('d','a'),
                
        ])

        G_model = nx.DiGraph()
        G_model.add_edges_from([
                    ('a','b'),
                    ('b','c'),
                    ('c','d'),
        ])

        a = edit_distance(G_model, G_log)
        b = 1/8

        self.assertEqual(a,b)

    

if __name__ == '__main__':
    unittest.main()