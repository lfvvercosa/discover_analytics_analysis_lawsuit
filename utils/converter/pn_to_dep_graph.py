import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from utils.creation import creation_utils
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def remove_silent_nodes(G):
    sil_nodes = [n for n in list(G.nodes) if n[1] is None]

    for n in sil_nodes:
        in_edges = list(G.in_edges(n))
        out_edges = list(G.out_edges(n))

        for i in in_edges:
            for o in out_edges:
                G.add_edge(i[0], o[1])

        G.remove_edges_from(in_edges)
        G.remove_edges_from(out_edges)

    G.remove_nodes_from(sil_nodes)
    
    return G


def petri_net_to_dep_graph(net, im, fm):
    G = nx.DiGraph()

    try:
        for place in net.places:
            for in_arc in place.in_arcs:
                if in_arc.source is not None:
                    for out_arc in place.out_arcs:
                        if out_arc.target is not None:
                            # print('source: ' + str(in_arc.source))
                            # print('target: ' + str(out_arc.target))

                            G.add_edge((in_arc.source.name, in_arc.source.label), 
                                       (out_arc.target.name, out_arc.target.label))

        G = remove_silent_nodes(G) 
        G = rename_nodes(G)
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        print("Next entry.")

    return creation_utils.add_nodes_label(G)


def rename_nodes(G):
    mapping = {}

    for n in list(G.nodes):
        mapping[n] = n[1]

    G = nx.relabel_nodes(G, mapping)

    return G


if __name__ == '__main__':
    net, im, fm = \
        pnml_importer.apply(os.path.join("simulations",
                                         "dependency_graph",
                                         "petri_nets",
                                         "BPI_Challenge_2014_Reference alignment.xes.gz.pnml"))
    
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)

    G = petri_net_to_dep_graph(net, im, fm)

    print(G.edges)
    print('# edges: ' + str(len(G.edges)))

    #visualize
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G,'state')
    nx.draw_networkx_labels(G, pos, labels = node_labels)
    nx.draw(G, pos)
    plt.show()

