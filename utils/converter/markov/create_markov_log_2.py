import networkx as nx
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter
from experiments import creation_utils


def add_node_if_needed(Gm, l):
    
    my_name = str(l)

    if not Gm.has_node(my_name):
        Gm.add_node(my_name, attr={'activity': l})
    
    return my_name


def add_start_node(Gm, repr='-'):
    Gm.add_node(repr, attr={'size': 1, 'activity': []})

    return repr


def get_sub_trace(l, start, end):
    pre = []
    pos = []

    while start < 0:
        start += 1
        pre += ['-']

    while end > len(l):
        end -= 1
        pos += ['-']

    return pre + l[start:end] + pos


def create_mk_abstraction_log_2(log, k):
    variants = variants_filter.get_variants(log)
    Gm = nx.DiGraph()

    s0 = add_start_node(Gm)

    for v in variants:
        l = variants[v][0]._list
        l = [act['concept:name'] for act in l]
        len_var = len(l)
        freq = len(variants[v])

        start = -(k-1)
        end = 1
        sw = get_sub_trace(l, start, end)
        sw = add_node_if_needed(Gm, sw)
        add_or_update_edge(Gm, s0, sw, freq)
        
        for i in range(len_var + k - 2):
            sx = sw
            start += 1
            end += 1
            sw = get_sub_trace(l, start, end)
            sw = add_node_if_needed(Gm, sw)
            add_or_update_edge(Gm, sx, sw, freq) 

        add_or_update_edge(Gm, sw, s0, freq)  

    return creation_utils.add_nodes_label(Gm)


def add_or_update_edge(Gm, s1, s2, freq):
    if not Gm.has_edge(s1, s2):
        Gm.add_weighted_edges_from([(s1, s2, freq)])
    else:
        Gm.edges[s1,s2]['weight'] += freq


if __name__ == '__main__':
    log_path = 'xes_files/test/test_markov.xes'
    log = xes_importer.apply(log_path)
    
    Gm = create_mk_abstraction_log_2(log, k=2)

    edges_list = list(Gm.edges)

    for e in edges_list:
        print('edge: ' + str(e) + ', weight: ' + \
              str(Gm.edges[e]['weight']))

    print('test')