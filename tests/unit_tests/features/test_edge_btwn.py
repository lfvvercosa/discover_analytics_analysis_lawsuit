from experiments.models.get_markov import get_markov_log, \
                                          get_markov_model
import networkx as nx

fil = 'Production_Data.xes.gz'
alg = 'IMf'
k_markov = 2

Gm_log = get_markov_log(fil, k_markov)

v = nx.edge_betweenness_centrality(Gm_log, weight='weight')

print(v)

