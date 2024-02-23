import pm4py
import pm4py.algo.simulation.tree_generator.variants.ptandloggenerator as ptand
import pm4py.algo.simulation.tree_generator.algorithm as algorithm


if __name__ == "__main__":


    params = {
        "sequence":0.5,
        "choice":0.5,
        "parallel":0,
        "loop":0,
        "or":0,
        "mode":4,
        "min":4,
        "max":4,
        "silent":0,
        "duplicate":0,
        "no_models":2,
    }

    variant = algorithm.Variants.PTANDLOGGENERATOR  

    my_trees = pm4py.generate_process_tree(variant=variant, parameters=params)

    pm4py.view_process_tree(my_trees[0], format='png')
    pm4py.view_process_tree(my_trees[1], format='png')

    pm4py.play_out(my_trees[0])
