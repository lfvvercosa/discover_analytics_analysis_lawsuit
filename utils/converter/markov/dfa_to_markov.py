import networkx as nx
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import remove_brackets_if_exists


def get_outgoing_arcs(Gd, n):
    arcs = []
    
    for e in Gd.edges(n):
        for a in Gd.edges[e]['activity']:
            a = remove_brackets_if_exists(a)
            arcs.append((e[0], e[1], a))
    
    return arcs


def init_dict(Gd):
    D = {}
    for n in Gd.nodes:
        D[n] = []

    return D


def get_init_state(Gd):
    for n in Gd.nodes:
        if n.is_init_state:
            return n


def add_start_node(Gm, repr='-'):
    Gm.add_node(repr, attr={'size': 1, 'activity': []})

    return repr


def create_mk_abstraction_dfa(Gd, k):
    Gm = nx.DiGraph()
    s0 = add_start_node(Gm)
    q0 = get_init_state(Gd)
    
    queue = []
    queue.append(q0)

    V = init_dict(Gd)
    X = init_dict(Gd)
    
    V[q0].append([])

    while queue:
        n = queue.pop(0)
        vn = V[n]
        O = get_outgoing_arcs(Gd, n)
        visited = []

        while vn:
            subtrace = vn.pop(0)
            t = subtrace.copy()

            if not O or n.is_final_state:
                my_node = add_node_if_needed(Gm, t)
                
                if len(t) < k:
                    add_edge_if_needed(Gm, s0, my_node)

                add_edge_if_needed(Gm, my_node, s0)

            if O:
                for (n, nt, a) in O:
                    t = subtrace.copy()
                    t.append(a)

                    curr_t = get_curr_subtrace(t, k)
                    shifted_t = get_shifted_subtrace(t, k)

                    if len(t) == k:
                        my_node = add_node_if_needed(Gm, t)
                        add_edge_if_needed(Gm, s0, my_node)

                    elif len(t) > k:
                        my_node = add_node_if_needed(Gm, shifted_t)
                        add_edge_if_needed(Gm, get_node(curr_t), my_node)

                    if len(t) >= k:
                        if nt.is_final_state:
                            add_edge_if_needed(Gm, my_node, s0)

                    # if len(t) < k and n.is_final_state:
                    #     my_node = add_node_if_needed(Gm, t)
                    #     add_edge_if_needed(Gm, s0, my_node)
                    #     add_edge_if_needed(Gm, my_node, s0)

                    if not shifted_t in X[nt]:
                        V[nt].append(shifted_t)

                        if nt not in queue:
                            queue.append(nt)

            X[n].append(subtrace)

    return Gm


def add_node_if_needed(Gm, t):
    my_name = str(t)

    if not Gm.has_node(my_name):
        Gm.add_node(my_name, attr={'activity': t.copy()})
    
    return my_name


def add_edge_if_needed(Gm, s1, s2):
    if not Gm.has_edge(s1, s2):
        Gm.add_edge(s1, s2)


def get_curr_subtrace(t, k):
    return t[0:k].copy()


def get_shifted_subtrace(t, k):
    my_t = t.copy()
    if len(my_t) <= k:
        return my_t
    else:
        return my_t[1:k+1] 


def get_node(t):
    return str(t)    




