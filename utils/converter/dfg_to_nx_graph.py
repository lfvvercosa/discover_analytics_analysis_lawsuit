import os
import networkx as nx
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization


def dfg_to_nx_graph(log):
    dfg = dfg_discovery.apply(log)
    G = nx.DiGraph()

    # gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
    # dfg_visualization.view(gviz)

    weighted_edges = [(e[0],e[1],dfg[e]) for e,w in dict(dfg).items()]
    G.add_weighted_edges_from(weighted_edges)

    return G



if __name__ == '__main__':
    log_path = os.path.join("xes_files", 
                            "real_processes",
                            "set_for_simulations", 
                            "5",
                            "Sepsis Cases - Event Log.xes.gz",
                           )
    log = xes_importer.apply(log_path)
    G = dfg_to_nx_graph(log)

    print('### The edges are:')
    print(G.edges)

    print('done!')





