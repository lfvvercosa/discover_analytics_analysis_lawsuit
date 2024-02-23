import networkx as nx
from utils.converter.nfa_to_dfa.Vertex import Vertex
from utils.converter.nfa_to_dfa.nfa_to_dfa_alg import get_final_states, \
                                                      get_activity_name, \
                                                      add_activity_to_edge, \
                                                      get_source
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.utils import reachability_graph


def get_trans_name(name):
    return '{' + name + '}'


def convert(ts):
    G = nx.DiGraph()
    final_states = get_final_states(ts.states)
    init_state = get_source(ts.states)

    for s in ts.states:
        G.add_node(Vertex({s}, final_states, init_state))
    
    for t in ts.transitions:
        from_vertex = Vertex({t.from_state}, final_states, init_state)
        to_vertex = Vertex({t.to_state}, final_states, init_state)

        G.add_edge(from_vertex.name, to_vertex.name)
        act = get_activity_name(t.name)

        add_activity_to_edge(G, from_vertex, to_vertex, act)
    
    return G



if __name__ == '__main__':
    my_file = 'petri_nets/IMf/' + \
              'BPI_Challenge_2014_Entitlement application.xes.gz.pnml'

    net, im, fm = pnml_importer.apply(my_file)
    ts = reachability_graph.construct_reachability_graph(net, im)

    G = convert(ts)

    edges_list = list(G.edges)

    for e in edges_list:
        print('edge: ' + str(e) + ', activity: ' + \
            str(G.edges[e]['activity']))


    print('test')