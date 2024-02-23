from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.conformance.alignments.petri_net import variants


pn_path = "models/petri_nets/tests/pn_parallel5.pnml"

net, im, fm = pnml_importer.apply(pn_path)

variant_a_star = variants.state_equation_a_star
parameters = {}

best_worst = variant_a_star.get_best_worst_cost(net, im, fm, parameters)

print('min cost: ' + str(best_worst))

########################################################

pn_path = "models/petri_nets/tests/pn_parallel6.pnml"

net, im, fm = pnml_importer.apply(pn_path)

variant_a_star = variants.state_equation_a_star
parameters = {}

best_worst = variant_a_star.get_best_worst_cost(net, im, fm, parameters)

print('min cost: ' + str(best_worst))