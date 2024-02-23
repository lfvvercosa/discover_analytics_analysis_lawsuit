import pm4py
import pm4py.algo.simulation.tree_generator.variants.ptandloggenerator as ptand
import pm4py.algo.simulation.tree_generator.algorithm as algorithm
from pm4py.visualization.petri_net import visualizer as pn_visualizer


if __name__ == "__main__":

    out_path = 'models/trees/exp5/'
    count = 0
    total_trees = 2

    params = {
        "sequence":0.45,
        "choice":0.29,
        "parallel":0.22,
        "loop":0.04,
        "or":0,
        "mode":20,
        "min":20,
        "max":20,
        "silent":0.2,
        "duplicate":0,
        "no_models":total_trees,
    }

    variant = algorithm.Variants.PTANDLOGGENERATOR  

    my_trees = pm4py.generate_process_tree(variant=variant, parameters=params)

    pm4py.view_process_tree(my_trees[0], format='png')

    for i in range(len(my_trees)):
        pm4py.write_ptml(my_trees[i], out_path + 'tree_'+ str(count) +'.ptml')
        count += 1

    

    print('done!')