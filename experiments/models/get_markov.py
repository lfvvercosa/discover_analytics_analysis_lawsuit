import pickle
import copy
import networkx as nx
from utils.creation import creation_utils
from libraries.networkx_graph import is_path_possible_in_trans_system


base_path = 'experiments/features_creation/'


def get_markov_model(f, alg, k):

    my_path = base_path + 'markov/from_dfa_reduced/k_'

    try:
        markov_path = my_path + str(k) + '/' + alg + '/' + f + '.txt'
        Gm_model = pickle.load(open(markov_path, 'rb'))

        return creation_utils.add_nodes_label(Gm_model)
    
    except:
        
        return None


def get_markov_log(f, k):

    my_path = base_path + 'markov_log/k_'

    try:
        markov_path = my_path + str(k) + '/' + f + '.txt'
        Gm_log = pickle.load(open(markov_path, 'rb'))

        return creation_utils.add_nodes_label(Gm_log)
    
    except:
        
        return None


def find_first_n_paths_markov(Gm, n):
    paths_acts = []
    Gm_copy = copy.deepcopy(Gm)
    Gm_copy.add_node('*', attr={'size': 1, 'activity': []})

    for e in Gm_copy.out_edges('-'):
        Gm_copy.add_edge('*', e[1])

    

    path_generator = nx.all_simple_paths(Gm_copy, 
                                         source="*", 
                                         target="-")

    while len(paths_acts) < n:
        a = []

        try:
            path = next(path_generator)
            path = path[1:-1]
        except:
            break

        for idx in range(len(path)):
            if idx == 0 or len(path) == 1:
                a += Gm_copy.nodes[path[idx]]['attr']['activity']
            else:
                a += [Gm_copy.nodes[path[idx]]['attr']['activity'][-1]]
        
        if len(a) > 0:
            paths_acts.append(a)

    return paths_acts


if __name__ == '__main__':
    print('test')