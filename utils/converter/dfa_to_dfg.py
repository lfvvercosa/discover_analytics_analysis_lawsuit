import networkx as nx


def dfa_to_dfg(Gt, k):
    if k == 1:
        G = nx.DiGraph()
        edges = set()
        nodes = set()
        sa = {}
        ea = {}

        for n in Gt.nodes:
            for i in Gt.in_edges(n):
                for act_i in Gt.edges[i]['activity']:
                    nodes.add(act_i)

                    if i[0].is_init_state:
                        sa[act_i] = 1
                    
                    if n.is_final_state:
                        ea[act_i] = 1

                    for o in Gt.out_edges(n):
                        for act_o in Gt.edges[o]['activity']:
                            nodes.add(act_o)
                            edges.add((act_i,act_o,1))

                            # if o[1].is_final_state:
                            #     ea[act_o] = 1
        
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)

        return (G,sa,ea)
    
    return None