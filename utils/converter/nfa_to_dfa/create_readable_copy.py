import networkx as nx
import copy
from utils.converter import tran_sys_to_nx_graph
from tests.unit_tests.nfa_to_dfa.test_nfa_to_dfa import \
    find_shortest_path_vertex_pair
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import \
    convert_nfa_to_dfa
from utils.converter.nfa_to_dfa.reduction_dfa import reduceDFA
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph


def readable_copy(Gd):
    Gs = nx.DiGraph()
    map_names = {}
    count_final = 0
    count_init = 0
    count_others = 0
    count_empty = 0

    for v in Gd.nodes:
        v2 = copy.deepcopy(v)

        if v2.is_final_state:
            new_name = 'f' + str(count_final)
            count_final += 1
        elif v2.is_init_state:
            new_name = 's' + str(count_init)
            count_init += 1
        elif v2.is_empty_state:
            new_name = 'e' + str(count_empty)
            count_empty += 1
        else:
            new_name = 'q' + str(count_others)
            count_others += 1

        v2.name = new_name
        map_names[v.name] = v2
        Gs.add_node(v2) 
    
    for e in Gd.edges:
        source_node = map_names[e[0].name]
        target_node = map_names[e[1].name]
        act = Gd.edges[e]['activity']

        Gs.add_edge(source_node, target_node)
        nx.set_edge_attributes(Gs, 
                            #   {(source_node.name, target_node.name): \
                              {(source_node, target_node): \
                              {'activity': act}})

    return Gs


if __name__ == '__main__':
    my_file = 'petri_nets/IMf/' + \
              'Production_Data.xes.gz.pnml'

    net, im, fm = pnml_importer.apply(my_file)
    ts = reachability_graph.construct_reachability_graph(net, im)

    G = tran_sys_to_nx_graph.convert(ts)
    short_path_G = find_shortest_path_vertex_pair(G, '{source1}', '{sink1}')

    Gd = convert_nfa_to_dfa(ts, include_empty_state=False)
    Gs = readable_copy(Gd)
    Gr = reduceDFA(Gs)
    Gs2 = readable_copy(Gr)

    print('test')

    edges_list = list(Gs2.edges('s0'))

    for e in edges_list:
        print('edge: ' + str(e) + ', activity: ' + str(Gs2.edges[e]['activity']))

    # short_path_Gd = find_shortest_path_vertex_pair(G, '{source1}', '{sink1}')



    
