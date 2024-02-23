from pm4py.objects.log.importer.xes import importer as xes_importer
from utils.converter.markov.create_markov_log_2 import create_mk_abstraction_log_2
from experiments.variant_analysis.MarkovMeasures import MarkovMeasures


path_p2v2 = 'xes_files/test_variants/exp4/exp4_p2_v2.xes'
log_p2v2 = xes_importer.apply(path_p2v2)
k_markov = 2
markov_fitness = MarkovMeasures(log_p2v2, k_markov)

print('fit gaussian: ' + str(markov_fitness.get_fitness_gaussian(n=0.3)))
print('fit mean: ' + str(markov_fitness.get_fitness_mean(p=0.3)))

path_p2v2 = 'xes_files/test_variants/exp4/exp4_p2v1_p2v2_perc.xes'
log_p2v2 = xes_importer.apply(path_p2v2)
k_markov = 2
markov_fitness = MarkovMeasures(log_p2v2, k_markov)

print('fit gaussian (noise): ' + str(markov_fitness.get_fitness_gaussian(n=0.3)))
print('fit mean (noise): ' + str(markov_fitness.get_fitness_mean(p=0.3)))
