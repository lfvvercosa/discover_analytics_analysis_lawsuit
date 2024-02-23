

def get_source(states):
    for s in states:
        if len(s.incoming) == 0 and len(s.outgoing) > 0:
            return s
            
    return None


def get_activity_name(trans):
    sep = ', '
    name = trans[trans.find(sep) + len(sep):-1]
    
    return remove_brackets_if_exists(name)


def get_name_edge(t):
    sep = '#'

    return t.from_state.name + sep + t.name + sep + t.to_state.name


def remove_brackets_if_exists(name):
    begin = 0
    end = len(name)

    if name[0] == "'" or name[0] == '"':
        begin = 1
    
    if name[-1] == "'" or name[-1] == '"':
        end = -1

    return name[begin:end]


def get_final_states(states, init_state, reachable_from_state):
    final_states = []

    for s in states:
        if len(s.outgoing) == 0 and len(s.incoming) > 0:
            final_states.append(s)

    # check whether empty trace is accepted
    reachable = get_all_reachable_states({init_state}, 
                                         None, 
                                         reachable_from_state)

    for r in reachable:
        if r in final_states:
            final_states.append(init_state)


    return final_states


def get_all_reachable_states(states, a, reachable_from_state):
    
    all_reachable = {}

    for s in states:
        reachable = {}
        visited = {}

        if s not in reachable_from_state:
            reachable_from_state[s] = {}

        if a in reachable_from_state[s]:
            reachable = reachable_from_state[s][a].copy()
        else:
            get_reachable_states(s, 
                                 a, 
                                 reachable, 
                                 visited)
            
            reachable_from_state[s][a] = reachable.copy()

        all_reachable.update(reachable)

    return set(all_reachable.keys())


def get_reachable_states(state, 
                         a, 
                         reachable, 
                         visited):

    for t in list(state.outgoing):
        s_new = t.to_state
        edge_name = get_name_edge(t)

        if edge_name not in visited:
            
            name = get_activity_name(t.name)
            
            if a == name:
                reachable[s_new] = True

                get_reachable_states(s_new, 
                                     None, 
                                     reachable,
                                     dict(visited, **{edge_name : True}))
                
                # get_reachable_states(s_new, 
                #                      None, 
                #                      reachable,
                #                      {})
                                     
            elif is_silent_trans(name):
                
                if not a:
                    reachable[s_new] = True

                get_reachable_states(s_new, 
                                     a, 
                                     reachable, 
                                     dict(visited, **{edge_name : True}))

    return reachable


def is_silent_trans(name):
    return name == 'None'