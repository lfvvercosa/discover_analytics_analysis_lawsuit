import networkx as nx
from utils.creation import creation_utils
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from utils.converter.all_dist import create_all_dist_ts
from utils.converter import tran_sys_utils

def is_silent_trans(name):
    return ', None)' in name


def trans_label(name):
    return name.split(',')[1][2:-2]


def transform_to_markov(G, sa, ea, ts):
    for s in sa:
        G.add_edge("-", s)
    
    for e in ea:
        G.add_edge(e, "-")

    if accepts_empty_trace(ts):
        G.add_edge("-", "-")

    G = creation_utils.add_nodes_label(G)

    return G


def consider_empty_trace(G, sa, ea, ts):
    if accepts_empty_trace(ts):
        for s in sa:
            for e in ea:
                G.add_edge(s, e)

    return G


def reach_graph_to_dfg_start_end(ts):
    G = reach_graph_to_dfg(ts)
    sa = find_start_acts(ts)
    ea = find_end_acts(ts)

    return G, sa, ea


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


def reach_graph_to_dfg_start_end_2(ts):
    G = reach_graph_to_dfg_2(ts)
    sa = find_start_acts(ts)
    ea = find_end_acts(ts)

    return G, sa, ea


def reach_graph_to_dfg_2(TS):

    G = nx.DiGraph()

    for s in TS.states:
        for trans in s.incoming:
            a = trans.name
            if not is_silent_trans(a):
                reach_graph_to_dfg_aux_2(G, 
                                         a, 
                                         s,
                                         {})

    G = nx.relabel_nodes(G, lambda x: trans_label(x))
    G = creation_utils.add_nodes_label(G)

    return G


def reach_graph_to_dfg_aux_2(G, a, s, visited):

    for trans in s.outgoing:
        name_trans = tran_sys_utils.get_name_edge(trans)

        if name_trans not in visited:
            visited[name_trans] = True
            a2 = trans.name

            if not is_silent_trans(a2):
                if not G.has_edge(a,a2):
                    G.add_weighted_edges_from([(a,a2,1)])
            else:
                reach_graph_to_dfg_aux_2(G, 
                                         a, 
                                         trans.to_state,
                                         visited)


def find_start_acts(TS):
    source = tran_sys_utils.get_source(TS.states)
    start_acts = set()
    visited = {}

    __find_start_acts_aux(source, visited, start_acts)


    return start_acts


def __find_start_acts_aux(s, visited, start_acts):
    for trans in s.outgoing:
        edge_name = tran_sys_utils.get_name_edge(trans)
        
        if edge_name not in visited:
            visited[edge_name] = True
        
            if is_silent_trans(trans.name):
                __find_start_acts_aux(trans.to_state, 
                                      visited, 
                                      start_acts)
            else:
                act = tran_sys_utils.get_activity_name(trans.name)
                start_acts.add(act)


def find_end_acts(TS):
    source = tran_sys_utils.get_source(TS.states)
    final_states = tran_sys_utils.get_final_states(TS.states, 
                                                   source, 
                                                   {})
    end_acts = set()
    visited = {}

    for s in final_states:
        __find_end_acts_aux(s, visited, end_acts)


    return end_acts


def __find_end_acts_aux(s, visited, end_acts):
    for trans in s.incoming:
        edge_name = tran_sys_utils.get_name_edge(trans)
        
        if edge_name not in visited:
            visited[edge_name] = True
        
            if is_silent_trans(trans.name):
                __find_end_acts_aux(trans.from_state, 
                                      visited, 
                                      end_acts)
            else:
                act = tran_sys_utils.get_activity_name(trans.name)
                end_acts.add(act)


def accepts_empty_trace(TS):
    source = tran_sys_utils.get_source(TS.states)
    final_states = tran_sys_utils.get_final_states(TS.states, 
                                                   source, 
                                                   {})
    
    return source in final_states


def graph_to_test_paths(G):
    H = G.copy(as_view=False)
    map_labels = {}

    for n in H.nodes:
        acts = H.nodes[n]['attr']['activity']

        if n != '-':
            new_label = "['" + acts[-1] + "']"
            map_labels[n] = new_label
    
    H = nx.relabel_nodes(H, map_labels)
    
    return H


if __name__ == '__main__':

    # my_file = 'utils/converter/models/examples/' + \
    #       'base_example3.pnml'
    my_file = 'petri_nets/IMd/'+\
              'edited_hh104_labour.pnml'

    net, im, fm = pnml_importer.apply(my_file)

    # gviz = pn_visualizer.apply(net, im, fm)
    # pn_visualizer.view(gviz)

    suffix = '!#@#!'

    TS = reachability_graph.construct_reachability_graph(net, im)

    for t in TS.transitions:
        print(t)

    create_all_dist_ts(TS, suffix)

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
