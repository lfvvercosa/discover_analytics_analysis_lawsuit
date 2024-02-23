import networkx as nx

from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_3 import convert_nfa_to_dfa_3
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg_2 import convert_nfa_to_dfa_2
from utils.converter.markov.dfa_to_markov_2 import create_mk_abstraction_dfa_2
from utils.converter.nfa_to_dfa.create_readable_copy import readable_copy
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA


def get_start_state_markov_2(Gm, repr):
    for n in Gm.nodes:
        if n == repr:
            return n
    
    return None


def are_markov_paths_possible_2(Gm, paths, k, repr='-'):
    n0 = get_start_state_markov_2(Gm, repr)
    
    for p in paths:
        if not is_markov_path_possible_2(Gm, k, p):
            return False

    return True


def is_markov_path_possible_2(Gm, k, path):
    my_path = path.copy()
    n = '-'

    while(my_path):
        act = my_path.pop(0)
        n = next_markov_node_2(Gm, n, act)

        if not n:
            return False

    return is_final_node_reachable(Gm, n, k)


def next_markov_node_2(Gm, n, act):
    for e in Gm.edges(n): 
        n_dest = e[1]

        acts = Gm.nodes[n_dest]['attr']['activity']

        if acts:
            if acts[-1] == act:
                return n_dest
    
    return None


def is_final_node_reachable(Gm, n, k):
    i = 0
    curr = Gm.nodes[n]['attr']['activity']

    while(len(curr[i:]) > 1):
        if not Gm.has_edge(fill_suffix(curr[i:], k), 
                           fill_suffix(curr[i+1:], k)):
            return False
        
        i += 1
    
    return Gm.has_edge(fill_suffix(curr[i:], k), '-')


def fill_suffix(n, k):
    t = n.copy()

    while len(t) < k:
        t.append('-')
    
    return str(t)


def is_tail_node(acts):
    return len(acts) > 1 and acts[-1] == '-'


def connect_previous_to_end(G, n):
    for e in G.in_edges(n):
        G.add_edge(e[0],'-')


def remove_suffix_if_exists(label, suffix):
    pos = label.rfind(suffix)

    if pos != -1:
        return label[:pos]
    else:
        return label


def change_markov_labels(Gm, suffix):
    H = Gm.copy(as_view=False)
    map_labels = {}
    count_labels = {}
    remove_list = []

    for n in H.nodes:
        acts = H.nodes[n]['attr']['activity']

        if n != '-':
            if not is_tail_node(acts):
                new_label = acts[-1]
                new_label = remove_suffix_if_exists(new_label, suffix)

                if new_label not in count_labels:
                    count_labels[new_label] = 1
                else:
                    count_labels[new_label] += 1
                
                new_label += suffix + str(count_labels[new_label])
                map_labels[n] = new_label
            else:
                connect_previous_to_end(H, n)

                remove_list.append(n)

    H.remove_nodes_from(remove_list)
    H = nx.relabel_nodes(H, map_labels)
    sa = get_starting_nodes(H)
    ea = get_end_nodes(H)

    # if accepts_empty_trace(H):
    #     H.add_edge(list(sa.keys())[0], list(ea.keys())[0])

    return (H, sa, ea)


def remove_tail_nodes(Gm):
    for n in Gm.nodes:
        acts = Gm.nodes[n]['attr']['activity']


def get_starting_nodes(Gm):
    sa = {}

    for e in Gm.out_edges('-'):
        if e[1] != '-':
            sa[e[1]] = 1
    
    return sa


def get_end_nodes(Gm):
    ea = {}

    for e in Gm.in_edges('-'):
        if e[0] != '-':
            ea[e[0]] = 1
    
    return ea


def accepts_empty_trace(G):
    return G.has_edge('-', '-')


def change_labels_all_dist(Gm):
    H = Gm.copy(as_view=False)
    map_labels = {}

    for n in H.nodes:
        acts = H.nodes[n]['attr']['activity']

        if n != '-':
            new_label = acts[-1]
            map_labels[n] = new_label
    
    H = nx.relabel_nodes(H, map_labels)
    sa = get_starting_nodes(H)
    ea = get_end_nodes(H)

    return (H, sa, ea)


def create_markov_from_pn(ts, k):
    Gd = convert_nfa_to_dfa_2(ts, 
                              init_state=None,
                              final_states=None,
                              include_empty_state=True)
    Gr = reduceDFA(Gd, include_empty_state=False)
    Gt = readable_copy(Gr)
    Gm = create_mk_abstraction_dfa_2(Gt, k=k)
    Gm, sa, ea = change_labels_all_dist(Gm)

    return Gm, sa, ea


def areMarkovGraphsEqual(Gm,G):
        for n in Gm.nodes:
            if not G.has_node(n):
                return False
        
        for n in G.nodes:
            if not Gm.has_node(n):
                return False
        
        for e in Gm.edges:
            if not G.has_edge(e[0],e[1]):
                return False
        
        for e in G.edges:
            if not Gm.has_edge(e[0],e[1]):
                return False

        return True