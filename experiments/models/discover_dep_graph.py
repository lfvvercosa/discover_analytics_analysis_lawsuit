import os
import networkx as nx
from utils.creation import creation_utils
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer


def from_heu_miner(heu_net, dep_thr):
    G = nx.DiGraph()

    dep_matrix = heu_net.dependency_matrix
    freq_dfg = heu_net.dfg

    for k,v in dep_matrix.items():
        for t,u in v.items():
            if u > dep_thr:
                G.add_edge(k,t, weight=freq_dfg[(k,t)])
    
    return creation_utils.add_nodes_label(G)


def discover_dep_graph(log, dep_thr):
    heu_net = heuristics_miner.apply_heu(log, 
                parameters={heuristics_miner.Variants.CLASSIC.\
                            value.Parameters.DEPENDENCY_THRESH: dep_thr})
    return from_heu_miner(heu_net, dep_thr)


if __name__ == '__main__':
    log_path = os.path.join("simulations", 
                            "flower_model",
                            "source_files", 
                            "logs",
                            "airline_log.xes")
    log = xes_importer.apply(log_path)
    dep_thr = 0.8

    heu_net = heuristics_miner.apply_heu(log, 
                parameters={heuristics_miner.Variants.CLASSIC.\
                            value.Parameters.DEPENDENCY_THRESH: dep_thr})

    # gviz = hn_visualizer.apply(heu_net)
    # hn_visualizer.view(gviz)

    G = from_heu_miner(heu_net, dep_thr)

    for e in G.edges:
        print('source:' + str(e[0]), 
              'dest:' + str(e[1]),
              'weight:' + str(G[e[0]][e[1]]['weight']))
            
