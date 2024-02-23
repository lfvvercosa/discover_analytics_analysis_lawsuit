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


def get_subtrace_with_preffix(subtrace, k):
    preffix_size = k - len(subtrace)
    pre = []

    if preffix_size > 0:
        for i in range(preffix_size):
            pre.append('-')
    
    return pre + subtrace


def get_subtrace_with_suffix(subtrace, k):
    suffix_size = k - len(subtrace)
    pos = []

    if suffix_size > 0:
        for i in range(suffix_size):
            pos.append('-')
    
    return subtrace + pos


def create_mk_abstraction_dfa_2(Gd, k):
    Gm = nx.DiGraph()
    s0 = add_start_node(Gm)
    q0 = get_init_state(Gd)
    
    queue = []
    queue.append(q0)

    V = init_dict(Gd)
    X = init_dict(Gd)
    
    V[q0].append(get_subtrace_with_preffix([], k))

    while queue:
        n = queue.pop(0)
        vn = V[n]
        O = get_outgoing_arcs(Gd, n)

        while vn:
            subtrace = vn.pop(0)
            t = subtrace.copy()

            if n.is_final_state:
                add_end_nodes(Gm, t, s0)
            
            if O:
                for (n, nt, a) in O:
                    t = subtrace.copy()
                    t.append(a)

                    curr_t = get_curr_subtrace(t, k)
                    shifted_t = get_shifted_subtrace(t, k)

                    if curr_t == get_subtrace_with_preffix([], k):
                        my_node = add_node_if_needed(Gm, shifted_t)
                        add_edge_if_needed(Gm, s0, my_node)
                    
                    else:
                        curr_node = add_node_if_needed(Gm, curr_t)
                        next_node = add_node_if_needed(Gm, shifted_t)
                        add_edge_if_needed(Gm, curr_node, next_node)
                    
                    if not shifted_t in X[nt]:
                        V[nt].append(shifted_t)

                        if nt not in queue:
                            queue.append(nt)

            X[n].append(subtrace)

    return Gm


def add_end_nodes(Gm, t, s0):

    if is_empty_trace(t):
        add_edge_if_needed(Gm, '-', '-')
    else:
        my_t = t.copy()
        k = len(t)

        next_node = add_node_if_needed(Gm, my_t)

        for i in range(k - 1):
            curr_t = get_subtrace_with_suffix(my_t[i:k], k)
            next_t = get_subtrace_with_suffix(my_t[i+1:k], k)

            curr_node = add_node_if_needed(Gm, curr_t)
            next_node = add_node_if_needed(Gm, next_t)

            add_edge_if_needed(Gm, curr_node, next_node)

        add_edge_if_needed(Gm, next_node, s0)


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


def is_empty_trace(trace):
    for t in trace:
        if t != '-':
            return False
    
    return True




