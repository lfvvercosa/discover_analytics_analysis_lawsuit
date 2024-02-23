import os
import networkx as nx
from collections import Counter
from utils.converter.dfg_to_nx_graph import dfg_to_nx_graph
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery


def nx_graph_to_dfg(G):
    edges = {}

    if nx.is_weighted(G):
        for n in G.nodes:
            for n2 in G.nodes:
                if G.has_edge(n, n2):
                    edges[(n,n2)] = G[n][n2]['weight']
    else:
        for n in G.nodes:
            for n2 in G.nodes:
                if G.has_edge(n, n2):
                    edges[(n,n2)] = 1
    
    return Counter(edges)


if __name__ == '__main__':
    log_path = os.path.join("xes_files", 
                            "real_processes",
                            "set_for_simulations", 
                            "5",
                            "Sepsis Cases - Event Log.xes.gz",
                           )
    log = xes_importer.apply(log_path)
    dfg = dfg_discovery.apply(log)
    G = dfg_to_nx_graph(log)
    dfg2 = nx_graph_to_dfg(G)
    print(dfg == dfg2)




