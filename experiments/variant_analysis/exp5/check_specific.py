import pm4py
from pm4py.algo.simulation.playout.process_tree import algorithm as playout_tree
from pm4py.algo.simulation.playout.process_tree.variants import basic_playout


if __name__ == "__main__":
    path_log1 = 'models/trees/exp5/tree_0.ptml'
    path_log2 = 'models/trees/exp5/tree_2.ptml'

    tree_log1 = pm4py.read_ptml(path_log1)
    tree_log2 = pm4py.read_ptml(path_log2)

    pm4py.view_process_tree(tree_log1, format='png')
    pm4py.view_process_tree(tree_log2, format='png')