import networkx as nx

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg


suffix = '!#@#!'


# G = nx.DiGraph()
# G.add_edge('A','C')
# G.add_edge('A','B')
# G.add_edge('C','B')
# G.add_edge('B','C')
# G.add_edge('C','D')
# G.add_edge('B','D')

# dfg = nx_graph_to_dfg(G)
# sa = {'A':1}
# ea = {'D':1}

# log = xes_importer.apply('tests/my_tests/test_feature_markov4.xes')


# G = nx.DiGraph()
# G.add_edge('A','C')
# G.add_edge('A','B')
# G.add_edge('C','D')
# G.add_edge('B','E')
# G.add_edge('E','D')

# dfg = nx_graph_to_dfg(G)
# sa = {'A':1}
# ea = {'D':1}

# log = xes_importer.apply('tests/my_tests/test_align_works.xes')


# G = nx.DiGraph()
# G.add_edge('A','C' + suffix + '1')
# G.add_edge('B','C' + suffix + '2')
# G.add_edge('C' + suffix + '1','D')
# G.add_edge('C' + suffix + '2','E')

# dfg = nx_graph_to_dfg(G)
# sa = {'A':1, 'B':1}
# ea = {'D':1, 'E':1}

# log = xes_importer.apply('tests/my_tests/test_markov_align.xes')


# G = nx.DiGraph()
# G.add_edge('A','C')
# G.add_edge('B','F')
# G.add_edge('C','D')
# G.add_edge('F','E')

# dfg = nx_graph_to_dfg(G)
# sa = {'A':1, 'B':1}
# ea = {'D':1, 'E':1}

# log = xes_importer.apply('tests/my_tests/test_markov_align2.xes')


# G = nx.DiGraph()
# G.add_edge('A','C' + suffix + '1')
# G.add_edge('C' + suffix + '1', 'B' + suffix + '1')
# G.add_edge('B' + suffix + '1', 'D' + suffix + '1')
# G.add_edge('A','B' + suffix + '2')
# G.add_edge('B' + suffix + '2', 'C' + suffix + '2')
# G.add_edge('C' + suffix + '2', 'D' + suffix + '2')

# dfg = nx_graph_to_dfg(G)
# sa = {'A':1}
# ea = {'D' + suffix + '1':1, 'D' + suffix + '2':1}

# log = xes_importer.apply('tests/my_tests/test_feature_markov4.xes')


# G = nx.DiGraph()
# G.add_edge('A','C' + suffix + '1')
# G.add_edge('C' + suffix + '1', 'D' + suffix + '1')
# G.add_edge('A','B' + suffix + '2')
# G.add_edge('B' + suffix + '2', 'C' + suffix + '2')
# G.add_edge('C' + suffix + '2', 'D' + suffix + '2')

# dfg = nx_graph_to_dfg(G)
# sa = {'A':1}
# ea = {'D' + suffix + '1':1, 'D' + suffix + '2':1}

# log = xes_importer.apply('tests/my_tests/test_feature_markov4.xes')


G = nx.DiGraph()
G.add_edge('A','B')
G.add_edge('B','C' + suffix + '1')
G.add_edge('B','D' + suffix + '2')
G.add_edge('C' + suffix + '1', 'D' + suffix + '1')
G.add_edge('D' + suffix + '2', 'C' + suffix + '2')

dfg = nx_graph_to_dfg(G)
sa = {'A':1}
ea = {'D' + suffix + '1':1, 'C' + suffix + '2':1}

log = xes_importer.apply('xes_files/tests/test_feat_markov_ks_diff6.xes')

# G = nx.DiGraph()
# G.add_edge('A','B')
# G.add_edge('B','C')
# G.add_edge('C','D')

# dfg = nx_graph_to_dfg(G)
# sa = {'A':1}
# ea = {'D':1}

# log = xes_importer.apply('tests/my_tests/test_feat_markov_ks_diff6.xes')

mean_fitness = 0

if sa and ea:
    alignments = dfg_alignment.apply(log, 
                                     dfg, 
                                     sa, 
                                     ea, 
                                     variant=dfg_alignment.Variants.MARKOV,
                                     parameters={'suffix':suffix})

    for t in alignments:
        mean_fitness += t['fitness']

    mean_fitness /= len(alignments)


print('mean_fitness align: ' + str(mean_fitness))

