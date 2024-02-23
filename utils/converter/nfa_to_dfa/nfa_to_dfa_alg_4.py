# Algorithm from the book "An introduction to formal languages and automata" \
# by Peter Linz

import networkx as nx
import re
from utils.converter.nfa_to_dfa.Vertex import Vertex


count_calls = 0
        

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


def get_final_states(states, init_state, reach_from_state):
    final_states = []

    for s in states:
        if len(s.outgoing) == 0 and len(s.incoming) > 0:
            final_states.append(s)

    # check whether empty trace is accepted
    reachable = get_all_reachable_states({init_state}, 
                                         None, 
                                         reach_from_state,
                                         True)

    for r in reachable:
        if r in final_states:
            if init_state not in final_states:
                final_states.append(init_state)


    return final_states


def convert_nfa_to_dfa_4(ts, 
                       init_state=None, 
                       final_states=None,
                       include_empty_state=True):

    Gd = nx.DiGraph()
    vertex_list = []
    acts =  get_all_activities(ts.transitions)
    reach_from_state = {}
    global count_calls

    # cache null activities individually
    get_all_reachable_states(ts.states, 
                             None, 
                             reach_from_state,
                             False)

    if not init_state:
        init_state = get_source(ts.states)
    
    if not final_states:
        final_states = get_final_states(ts.states, 
                                        init_state, 
                                        reach_from_state)
        
    if not final_states or not init_state:
        return None

    if init_state is not None and final_states:        
        v = Vertex({init_state}, final_states, init_state)
        Gd.add_node(v)
        vertex_list.append(v.name)

        while vertex_list:

            # print('vertex list size: ' + str(len(vertex_list)))

            vertex = get_node(Gd, vertex_list.pop(0))
            states = vertex.states

            # if vertex.name == '{p_1,p_11,p_6}':
            #     print()

            for a in acts:
                reachable = get_all_reachable_states(states, 
                                                     a, 
                                                     reach_from_state,
                                                     True)
                
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

        # print('### Total Calls ###')
        # print(str(count_calls))
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


def get_all_reachable_states(states, 
                             a, 
                             reach_from_state,
                             regular_cache):
    
    all_reachable = {}

    if not regular_cache:
        reach = None
    else:
        reach = reach_from_state

    for s in states:
        reachable = {}
        visited = {}

        if s not in reach_from_state:
            reach_from_state[s] = {}

        if a in reach_from_state[s]:
            reachable.update(reach_from_state[s][a].copy())
        else:
            get_reachable_states(s, 
                                 a, 
                                 reachable, 
                                 visited,
                                 reach)

        if not regular_cache:
            reach_from_state[s][a] = reachable

        all_reachable.update(reachable)

        # if s not in reach_from_state:
        #     reach_from_state[s] = {}

        # if a in reach_from_state[s]:
        #     reachable.update(reach_from_state[s][a].copy())
        # else:
        #     get_reachable_states(s, 
        #                          a, 
        #                          reachable, 
        #                          visited,
        #                          reach_from_state)
            
        #     reach_from_state[s][a] = reachable.copy()

        # all_reachable.update(reachable)

    return set(all_reachable.keys())


def get_reachable_states(state, 
                         a, 
                         reachable, 
                         visited,
                         reach_from_state):

    global count_calls
    count_calls += 1

    if reach_from_state and state not in reach_from_state:
        reach_from_state[state] = {}

    reachable_bef = reachable.copy()
    # my_reachable = reachable.copy()

    for t in list(state.outgoing):
        s_new = t.to_state
        edge_visit = get_name_edge_visit(t, state, s_new, a)

        if edge_visit not in visited:
            
            visited = dict(visited, **{edge_visit : True})
            name = get_activity_name(t.name)
            next_call = False
            
            if a == name:
                reachable[s_new] = 1
                next_a = None
                next_call = True
                # visited = {}
                                     
            elif is_silent_trans(name):
                if not a:
                    if s_new not in reachable:
                        reachable[s_new] = 1
                    else:
                        reachable[s_new] += 1

                next_call = True
                next_a = a

            if next_call:
                                
                if reach_from_state and s_new not in reach_from_state:
                    reach_from_state[s_new] = {}

                if reach_from_state and next_a in reach_from_state[s_new]:
                    # reachable.update(reach_from_state[s_new][next_a].copy())
                    update_reachable(reachable, 
                                     reach_from_state[s_new][next_a].copy()) 
                else:
                    get_reachable_states(s_new, 
                                         next_a, 
                                         reachable, 
                                         visited,
                                         reach_from_state)
        else:
            if reach_from_state and s_new in reach_from_state:
                if a in reach_from_state[s_new]:
                    update_reachable(reachable, 
                                     reach_from_state[s_new][a].copy())

    reachable_now = reachable.copy()
    clear_past_reachable(reachable_now, reachable_bef)

    if reach_from_state:
        reach_from_state[state][a] = reachable_now
               

    return reachable


def update_reachable(reachable, cache):
    for k in cache:
        if k not in reachable:
            reachable[k] = cache[k]
        else:
            reachable[k] += cache[k]


def clear_past_reachable(reachable_now, reachable_bef):
    for k in reachable_bef:
        reachable_now[k] -= reachable_bef[k]

        if reachable_now[k] == 0:
            del reachable_now[k]


def get_name_edge_visit(t, state, s_new, a):
    sep = '#'

    return str(a) + sep + state.name + sep + \
           t.name + sep + s_new.name


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


def accepts_empty_trace():
    return


if __name__ == '__main__':
    print('test')
