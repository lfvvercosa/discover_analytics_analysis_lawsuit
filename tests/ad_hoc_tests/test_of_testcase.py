import unittest
import networkx as nx
from features.features import between_diff


class TestBetweennessFeature(unittest.TestCase):

    def test1(self):
        val = [True, True, False, True, True]

        for v in val:
            self.assertTrue(v)
            print('oi') 


if __name__ == '__main__':
    unittest.main()