import os
import sys
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
# import pm4py.algo.discovery.inductive.parameters as Parameters
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.pandas import case_statistics


from pm4py.algo.discovery.inductive.variants.im_d.dfg_based import Parameters


my_file = 'xes_files/real_processes/set_for_simulations/3/' + \
          'edited_hh104_weekends.xes.gz'

log = xes_importer.apply(my_file)

net, im, fm = heuristics_miner.apply(log, 
                parameters={heuristics_miner.Variants.CLASSIC.\
                            value.Parameters.DEPENDENCY_THRESH: 0.7})


gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.view(gviz)

try:
    print('Precision Alignments:')
    prec = precision_evaluator.apply(log, net, im, fm, 
            variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    print(prec)

    print('Fitness Alignments:')
    fitness = replay_fitness_evaluator.apply(log, net, im, fm, 
            variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
    print(fitness)

except Exception as exc:
    print(exc)
    print("keep going!")
