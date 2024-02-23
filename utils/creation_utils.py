import networkx as nx


def add_nodes_label(G):
    attr = {}

    for n in list(G.nodes):
        attr[n] = {'label':str(n), 'attr':{'activity':[str(n)]}}
    
    nx.set_node_attributes(G, attr)

    return G