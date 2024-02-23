import networkx as nx
from utils.converter.nfa_to_dfa.Vertex import Vertex
from pm4py.objects.transition_system.obj import TransitionSystem


def get_final_state_dict(Gd):
    final_states = {}

    for n in Gd.nodes:
        final_states[n.name] = n.is_final_state
    
    return final_states


def get_final_states(Gd):
    final_states = []

    for n in Gd.nodes:
        if n.is_final_state:
            final_states.append(n)
    
    return final_states


def get_init_state(Gd):
    for n in Gd.nodes:
        if n.is_init_state:
            return n


def find_all_activities(Gd):
    acts = set()

    for n in Gd.nodes:
        for e in Gd.edges(n.name):
            for a in Gd.get_edge_data(*e)['activity']:
                acts.add(a)
    
    return acts


def find_nodes_reachability(Gd):
    reach = {}
    # final_states = get_final_state_dict(Gd)

    for n in Gd.nodes:

        reach[n.name] = {}

        for e in Gd.edges(n.name):
            # is_final_state = final_states[e[1]]

            for a in Gd.get_edge_data(*e)['activity']:
                # reach[n.name][a] = is_final_state
                reach[n.name][a] = e[1].name


    return reach


def init_distinguish(reach):
    distinguish = {}

    for k in reach:
        distinguish[k] = {}
        for k2 in reach:
            distinguish[k][k2] = None
    
    return distinguish


def distinguish_final_states(distinguish, Gd):
    final_state_dict = get_final_state_dict(Gd)

    for k in final_state_dict:
        if final_state_dict[k]:
            for k2 in final_state_dict:
                # if not final_state_dict[k2]:
                #     distinguish[k][k2] = True
                    
                if final_state_dict[k2]:
                    distinguish[k][k2] = False
                    distinguish[k2][k] = False
                else:
                    distinguish[k][k2] = True
                    distinguish[k2][k] = True
    
    return distinguish


def is_distinguish_pair(k, k2, reach, distinguish):
    for a in reach[k]:
        if a in reach[k2]:
            target_k = reach[k][a]
            target_k2 = reach[k2][a]

            if distinguish[target_k][target_k2]:
                return True
            
    
    return None



# def is_distinguish_pair(k, k2, reach):
#     is_distinguish = False

#     # if k == '{q2}' and k2 == '{q3}':
#     #     print('testing...')

#     for a in reach[k]:
#         if a in reach[k2]:
#             if reach[k][a] != reach[k2][a]:
#                 is_distinguish = True
#                 break
#         # else:
#         #     is_distinguish = True

#         elif reach[k][a]:
#             is_distinguish = True
#             break
    
#     return is_distinguish


def find_distinguish_pairs(Gd):
    reach = find_nodes_reachability(Gd)
    distinguish = init_distinguish(reach)
    distinguish = distinguish_final_states(distinguish, Gd)
    is_pair_marked = True

    while(is_pair_marked):
        is_pair_marked = False

        for k in reach:
            for k2 in reach:
                if k != k2 and not distinguish[k][k2]:
                    if is_distinguish_pair(k, k2, reach, distinguish):
                        distinguish[k][k2] = True
                        distinguish[k2][k] = True
                        is_pair_marked = True

    for k in distinguish:
        for i in distinguish[k]:
            if not distinguish[k][i]:
                distinguish[k][i] = False

    return distinguish


# def find_distinguish_pairs(Gd):
#     reach = find_nodes_reachability(Gd)
#     # acts = find_all_activities(Gd)
#     distinguish = init_distinguish(reach)
#     distinguish = distinguish_final_states(distinguish, Gd)

#     for k in reach:
#         for k2 in reach:
#             if distinguish[k][k2] is None:
#                 if k != k2:
#                     if is_distinguish_pair(k, k2, reach) or \
#                     is_distinguish_pair(k2, k, reach):
#                         distinguish[k][k2] = True
#                     else:
#                         distinguish[k][k2] = False
#                 else:
#                     distinguish[k][k2] = False
    
#     return distinguish


def get_distinct_sets(distinguish):
    sets = []

    for k in distinguish:
        if belongs_to_set(k, sets):
            continue
        
        sets.append(creates_new_set(distinguish[k]))
    
    return sets


def belongs_to_set(k, sets):
    for s in sets:
        if k in s:
            return True
    
    return False


def creates_new_set(d):
    new_set = {}

    for k in d:
        if not d[k]:
            new_set[k] = True
    
    return new_set


def get_nodes_sets(Gd, sets):

    for n in Gd.nodes:
        for s in sets:
            if n.name in s.keys():
                s[n.name] = n
        
    return sets


def has_final_state(final_states, keys):
    for s in final_states:
        if final_states[s]:
            if s in keys:
                return True
    
    return False


def get_empty_sets(sets):
    empty_sets = []

    for se in sets:
        states = set(se.values())

        for st in states:
            if st.is_empty_state:
                empty_sets.append(se)

                break
    
    return empty_sets


def has_empty_state(my_set):
    states = set(my_set.values())

    for s in states:
        if s.is_empty_state:
            return states
    
    return []


def has_empty_state_in_edge(my_edge):
    s0 = my_edge[0]
    s1 = my_edge[1]

    if s0.is_empty_state or s1.is_empty_state:
        return True
    
    return False


def create_reduced_graph(Gd, sets, sep='#'):
    final_states = get_final_states(Gd)
    init_state = get_init_state(Gd)
    sets = get_nodes_sets(Gd, sets)
    map_states = {}

    Gr = nx.DiGraph()

    for s in sets:
        states = set(s.values())
        v = Vertex(states, final_states, init_state, sep)
        Gr.add_node(v)
        
        for st in states:
            map_states[st.name] = v
    
    for e in Gd.edges:
        source = map_states[e[0]]
        dest = map_states[e[1]]
        Gr.add_edge(source, dest)
        add_activity_to_edge(Gr, 
                             source, 
                             dest, 
                             Gd.edges[e]['activity'])

    return Gr


def add_activity_to_edge(G, v1, v2, vector):
    if not G.edges[(v1, v2)]:
        nx.set_edge_attributes(G, {(v1, v2): {'activity': []}})
    
    G.edges[(v1, v2)]['activity'] += vector
    G.edges[(v1, v2)]['activity'] = \
        list(set(G.edges[(v1, v2)]['activity']))


def reduceDFA(Gd, include_empty_state=False):
    distinguish = find_distinguish_pairs(Gd)
    sets = get_distinct_sets(distinguish)
    Gr = create_reduced_graph(Gd, sets, '#')

    if not include_empty_state:
        Gr = remove_empty_states(Gr)

    return Gr


def remove_empty_states(Gr):
    removal_list = []

    for n in Gr.nodes:
        for s in n.states:
            if s.is_empty_state:
                removal_list.append(n)
                
                break
    
    for n in removal_list:
        Gr.remove_node(n)

    return Gr


# def set_reducedDFA_edges(Gr, Gd)

def mark(Gd):
    print('test')
