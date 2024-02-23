from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.conformance.alignments.petri_net import variants

import time
                

# def get_model_cost_func(pn):
#     model_cost_function = dict()
    
#     for t in pn.transitions:
#         if t.label is not None:
#             model_cost_function[t] = 1
#         else:
#             model_cost_function[t] = 0


#     return model_cost_function
   

log_path = "/home/vercosa/Insync/doutorado/artigos/artigo_alignment/event_logs/tests/edited_hh104_labour.xes"
pn_path = "/home/vercosa/Insync/doutorado/artigos/artigo_alignment/event_logs/tests/edited_hh104_labour.pnml"

log = xes_importer.apply(log_path)
net, im, fm = pnml_importer.apply(pn_path)

start = time.time()
# aligned_traces = alignments.apply_log(log, net, im, fm)
my_variant = variants.dijkstra_less_memory
# my_variant = variants.state_equation_a_star
parameters = {}
parameters[replay_fitness_evaluator.Parameters.ALIGN_VARIANT] = my_variant

# parameters[my_variant.PARAM_MODEL_COST_FUNCTION] = get_model_cost_func(net)

# best_worst = my_variant.get_best_worst_cost(net, im, fm, parameters)

v = replay_fitness_evaluator.apply(log, net, im, fm, 
                        variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED,
                        parameters=parameters)
v = v['log_fitness']
end = time.time()

print(end - start)