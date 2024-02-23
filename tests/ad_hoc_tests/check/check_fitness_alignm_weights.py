from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.conformance.alignments.petri_net import variants


import time
                

def get_model_cost_func(pn, log_cost_func):
    model_cost_function = dict()
    
    for t in pn.transitions:
        if t.label is not None:
            model_cost_function[t] = log_cost_func[t.label] 
            # model_cost_function[t] = 1
        else:
            model_cost_function[t] = 0


    return model_cost_function


# def get_sync_cost_func(pn):
#     sync_cost_function = dict()
    
#     for t in pn.transitions:
#         sync_cost_function[t] = 0


#     return sync_cost_function


# log_path = "/home/vercosa/Insync/doutorado/artigos/artigo_alignment/event_logs/tests/edited_hh104_labour.xes"
# pn_path = "/home/vercosa/Insync/doutorado/artigos/artigo_alignment/event_logs/tests/edited_hh104_labour.pnml"

log_path = "xes_files/tests/log_parallel6.xes"
pn_path = "models/petri_nets/tests/pn_parallel6.pnml"
log_cost_func = {'A':1,'B':2,'C':2}

log = xes_importer.apply(log_path)
net, im, fm = pnml_importer.apply(pn_path)

start = time.time()
my_variant = variants.my_dijkstra
# my_variant = variants.state_equation_a_star

parameters = {}
parameters[replay_fitness_evaluator.Parameters.ALIGN_VARIANT] = my_variant
parameters[my_variant.Parameters.PARAM_MODEL_COST_FUNCTION] = \
                get_model_cost_func(net, log_cost_func)
parameters[my_variant.Parameters.PARAM_STD_SYNC_COST] = 0
parameters[my_variant.Parameters.PARAM_TRACE_COST_FUNCTION] = log_cost_func


best_worst = my_variant.get_best_worst_cost(net, im, fm, parameters)

aligned_traces = alignments.apply_log(log, net, im, fm,
                        variant=my_variant,
                        parameters=parameters)

v = replay_fitness_evaluator.apply(log, net, im, fm, 
                        variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED,
                        parameters=parameters)
# v = v['log_fitness']
end = time.time()

# print(end - start)

print('fitness: ' + str(v['log_fitness']))