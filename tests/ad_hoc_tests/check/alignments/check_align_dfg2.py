import networkx as nx

from utils.converter.nx_graph_to_dfg import nx_graph_to_dfg 

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment


G = nx.DiGraph()
G.add_edge('Z!#@#!1','X!#@#!1')
G.add_edge('X!#@#!1','Z!#@#!2')

dfg = nx_graph_to_dfg(G)
sa = {'Z!#@#!1':1}
ea = {'Z!#@#!2':1}

log = xes_importer.apply('xes_files/tests/test_align2.xes')

mean_fitness = 0
suffix = '!#@#!'

if sa and ea:
    alignments = dfg_alignment.apply(log, 
                                     dfg, 
                                     sa, 
                                     ea,
                                     variant=dfg_alignment.Variants.MARKOV,
                                     parameters={'suffix':suffix}
                                    )

    for t in alignments:
        mean_fitness += t['fitness']

    mean_fitness /= len(alignments)


print('mean_fitness align: ' + str(mean_fitness))
print()

for a in alignments:
    print(a)
