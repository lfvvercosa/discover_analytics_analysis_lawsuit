import networkx as nx
from utils.creation import creation_utils
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer


def is_silent_trans(name):
    return ', None)' in name


def trans_label(name):
    return name.split(',')[1][2:-2]


def reach_graph_to_dfg(TS):

    G = nx.DiGraph()

    for s in TS.states:
        for trans in s.incoming:
            a = trans.name
            if not is_silent_trans(a):
                reach_graph_to_dfg_aux(G, 
                                       a, 
                                       s,
                                       {})

    G = nx.relabel_nodes(G, lambda x: trans_label(x))
    G = creation_utils.add_nodes_label(G)

    return G


def reach_graph_to_dfg_aux(G, a, s, visited):

    if s.name not in visited:
        visited[s.name] = True
        for trans in s.outgoing:
            a2 = trans.name
            if not is_silent_trans(a2):
                if not G.has_edge(a,a2):
                    G.add_weighted_edges_from([(a,a2,1)])
            else:
                reach_graph_to_dfg_aux(G, 
                                       a, 
                                       trans.to_state,
                                       visited)


def petri_net_to_dfg(net, im):
    TS = reachability_graph.construct_reachability_graph(net, im)

    # gviz = ts_visualizer.apply(TS)
    # ts_visualizer.view(gviz)

    return reach_graph_to_dfg(TS)


if __name__ == '__main__':

    # my_file = 'utils/converter/models/examples/' + \
    #       'base_example3.pnml'
    my_file = 'simul_qual_metr/tests/dfg_precision/tests/petri_net_test.pnml'

    net, im, fm = pnml_importer.apply(my_file)

    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)

    TS = reachability_graph.construct_reachability_graph(net, im)

    gviz = ts_visualizer.apply(TS)
    ts_visualizer.view(gviz)

    G = reach_graph_to_dfg(TS)

    print(G.edges)

    total_nodes = len(G.nodes)
    total_edges = len(G.edges)
    percent_edges = total_edges/(total_nodes**2)
    print('### edges/possible edges:' + str(round(percent_edges,2)))
    print('### The edges are:')
    print(G.edges)
