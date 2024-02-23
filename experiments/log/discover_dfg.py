import os
from utils.converter.dfg_to_nx_graph import dfg_to_nx_graph
from utils.creation import creation_utils
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization


def discover_dfg(log):
    return creation_utils.add_nodes_label(dfg_to_nx_graph(log))


if __name__ == '__main__':
    log_path = os.path.join("xes_files", 
                            "real_processes",
                            "set_for_simulations", 
                            "5",
                            "Sepsis Cases - Event Log.xes.gz",
                           )
    log = xes_importer.apply(log_path)
    G = discover_dfg(log)
    G = creation_utils.add_nodes_label(G)

    print('### The edges are:')
    print(G.edges)
    