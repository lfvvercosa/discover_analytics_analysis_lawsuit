import os
import sys
import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
from utils.creation import creation_utils
from utils.converter.reach_graph_to_dfg import reach_graph_to_dfg
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.utils import reachability_graph


def petri_net_to_dfg(net, im, fm, ret_dict):
    print('## Obtaining reachability graph')
    TS = reachability_graph.construct_reachability_graph(net, im)
    print('## Converting reach graph to DFG')
    G = reach_graph_to_dfg(TS)

    ret_dict['dfg'] = G

    return G


def petri_net_to_dfg_with_timeout(net, im, fm, timeout):
    print("### creating DFG from Petri-net")
    manager = multiprocessing.Manager()
    ret_dict = manager.dict()

    p = multiprocessing.Process(target=petri_net_to_dfg,
                                    args=(
                                            net,
                                            im,
                                            fm,
                                            ret_dict
                                         )
                               )

    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
    
        return None

    p.join()

    if 'dfg' in ret_dict:
        return ret_dict['dfg']
    
    return None
        

if __name__ == '__main__':
    net, im, fm = \
        pnml_importer.apply(os.path.join("simul_qual_metr",
                                         "petri_nets",
                                         "BPI_Challenge_2014_Reference alignment.xes.gz.pnml"))
    
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)

    G = petri_net_to_dfg(net, im, fm)

    print(G.edges)
    print('# edges: ' + str(len(G.edges)))

