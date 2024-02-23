import networkx as nx
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import get_activity_name, \
                                                      is_silent_trans

def create_all_dist_dfa(Gd, suffix):
    acts = {}
    Gn = Gd.copy(as_view=False)

    for e in Gd.edges:
        l = Gd.edges[e]['activity']
        m = []

        for a in l:
            if a not in acts:
                acts[a] = 1

            m.append(a + suffix + str(acts[a]))
            acts[a] += 1

        Gn.edges[e]['activity'] = m

    return Gn  


def create_all_dist_ts(ts, suffix):
    rename_aux = {}

    for t in ts.transitions:
        name  = get_activity_name(t.name)
        code = get_name_code(t.name)

        if not is_silent_trans(name):
            if name not in rename_aux:
                rename_aux[name] = 1
            
            new_name = name + suffix + str(rename_aux[name])
            rename_aux[name] += 1
            t.set_name("("+code+", '"+new_name+"')")


def create_name_all_dist_ts(ts):
    count = 0
    dict_names = {}

    for t in ts.states:
        if 'source' not in t.name and 'sink' not in t.name:
            dict_names[t.name] = "p_" + str(count)
            t.set_name("p_" + str(count))
            count += 1
        else:
            dict_names[t.name] = t.name
    
    print(dict_names)


def create_name_all_dist_ts_temp(ts):
    dict_names = {'n111n61n91': 'p_0', 'n121n61n91': 'p_1', 'n121n61n71': 'p_2', 
    'n101n111n61': 'p_3', 'n131n61n91': 'p_4', 'n101n131n61': 'p_5', 'n21': 'p_6', 
    'n131n61n81': 'p_7', 'n111n51n71': 'p_8', 'n11': 'p_9', 'n101n121n61': 'p_10', 
    'n121n61n81': 'p_11', 'n41': 'p_12', 'n131n61n71': 'p_13', 'n31': 'p_14', 
    'n111n61n81': 'p_15'}

    for t in ts.states:
        if 'source' not in t.name and 'sink' not in t.name:
            t.set_name(dict_names[t.name])
    

def get_name_code(name):
    return name[1:name.find(',')]


def removeSuffixAct(act, suffix):
    idx = act.rfind(suffix)

    if idx == -1:
        return act
    else:
        return act[:act.rfind(suffix)]