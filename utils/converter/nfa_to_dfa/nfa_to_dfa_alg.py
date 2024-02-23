# Algorithm from the book "An introduction to formal languages and automata" \
# by Peter Linz

import networkx as nx
import re
from utils.converter.nfa_to_dfa.Vertex import Vertex


# def get_source(states):
#     for s in states:
#         if len(s.incoming) == 0 and \
#            bool(re.match('^source[0-9]+$', s.name)):
#                 return s
            
#     return None

def get_source(states):
    for s in states:
        if len(s.incoming) == 0 and len(s.outgoing) > 0:
            return s
            
    return None


def get_all_activities(transitions):
    act = []

    for t in transitions:
        name = get_activity_name(t.name)
        if not is_silent_trans(name) and name not in act:
           act.append(name)
    
    return act


def is_silent_trans(name):
    return name == 'None'


def get_activity_name(trans):
    sep = ', '
    name = trans[trans.find(sep) + len(sep):-1]
    
    return remove_brackets_if_exists(name)


def remove_brackets_if_exists(name):
    begin = 0
    end = len(name)

    if name[0] == "'" or name[0] == '"':
        begin = 1
    
    if name[-1] == "'" or name[-1] == '"':
        end = -1

    return name[begin:end]


# def get_final_states(states):
#     final_states = []

#     for s in states:
#         if bool(re.match('^sink[0-9]+$', s.name)):
#             final_states.append(s)
    
#     return final_states


def get_final_states(states):
    final_states = []

    for s in states:
        if len(s.outgoing) == 0 and len(s.incoming) > 0:
            final_states.append(s)
            
    return final_states


def convert_nfa_to_dfa(ts, 
                       init_state=None, 
                       final_states=None,
                       include_empty_state=True):

    Gd = nx.DiGraph()
    vertex_list = []
    acts =  get_all_activities(ts.transitions)

    if not final_states:
        final_states = get_final_states(ts.states)
    
    if not init_state:
        init_state = get_source(ts.states)
    
    if not final_states or not init_state:
        return None

    if init_state is not None and final_states:        
        v = Vertex({init_state}, final_states, init_state)
        Gd.add_node(v)
        vertex_list.append(v.name)

        while vertex_list:
            vertex = get_node(Gd, vertex_list.pop(0))
            states = vertex.states

            for a in acts:
                reachable = get_all_reachable_states(states, a)
                v = get_or_create_node(Gd, 
                                       reachable, 
                                       final_states, 
                                       init_state)

                if not include_empty_state and v.is_empty_state:
                    continue

                if not Gd.has_node(v.name):
                    vertex_list.append(v)
                    Gd.add_node(v)
                                
                if not Gd.has_edge(vertex.name, v.name):
                    Gd.add_edge(vertex, v)

                add_activity_to_edge(Gd, vertex, v, a)

        return Gd
    else:
        return None


def get_node(Gd, name):
    for v in Gd.nodes:
        if name == v.name:
            return v
    
    return None


def get_or_create_node(Gd, reachable, final_states, init_state):
    v = Vertex(reachable, final_states, init_state)

    for n in Gd.nodes:
        if n.name == v.name:
            return n
    
    return v


def get_all_reachable_states(states, a):
    
    all_reachable = {}

    for s in states:
        reachable = {}
        visited = {}
        get_reachable_states(s, 
                             a, 
                             reachable, 
                             visited)

        all_reachable.update(reachable)

    return set(all_reachable.keys())


def get_reachable_states(state, 
                         a, 
                         reachable, 
                         visited):

    for t in list(state.outgoing):
        s_new = t.to_state
        edge_name = get_name_edge(t, state, s_new)

        if edge_name not in visited:
            
            name = get_activity_name(t.name)
            
            if a == name:
                reachable[s_new] = True

                get_reachable_states(s_new, 
                                     None, 
                                     reachable,
                                     dict(visited, **{edge_name : True}))
                                     
            elif is_silent_trans(name):
                
                if not a:
                    reachable[s_new] = True

                get_reachable_states(s_new, 
                                     a, 
                                     reachable, 
                                     dict(visited, **{edge_name : True}))

    return reachable


def get_name_edge(t, state, s_new):
    sep = '#'

    return state.name + sep + t.name + sep + s_new.name


def add_activity_to_edge(G, v1, v2, a):
    if not G.edges[(v1.name, v2.name)]:
        nx.set_edge_attributes(G, {(v1.name, v2.name): {'activity': []}})
    
    G.edges[(v1.name, v2.name)]['activity'].append(a)


def remove_empty_state(G):
    for n in G.nodes:
        if n.is_empty_state:
            n_e = n
            break
    
    G.remove_node(n_e)

    return G


if __name__ == '__main__':
    print('test')
