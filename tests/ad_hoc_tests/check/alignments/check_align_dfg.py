import networkx as nx

from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg 

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment


G = nx.DiGraph()
G.add_edge('B','C')
G.add_edge('C','D')
G.add_edge('D','G')
G.add_edge('B','E')
G.add_edge('E','F')
G.add_edge('F','H')

dfg = nx_graph_to_dfg(G)
sa = {'B':1}
ea = {'G':1, 'H':1}

log = xes_importer.apply('xes_files/tests/test_align.xes')

mean_fitness = 0

if sa and ea:
    alignments = dfg_alignment.apply(log, 
                                     dfg, 
                                     sa, 
                                     ea
                                    )

    for t in alignments:
        mean_fitness += t['fitness']

    mean_fitness /= len(alignments)


print('mean_fitness align: ' + str(mean_fitness))
print()

for a in alignments:
    print(a)
