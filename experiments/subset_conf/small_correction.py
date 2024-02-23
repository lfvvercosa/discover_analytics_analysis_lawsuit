import time

from experiments.subset_conf import subset_alg

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.petri_net.importer import importer as pnml_importer


log_path = 'xes_files/3/1a_VARA_DE_FEITOS_TRIBUTARIOS_DO_ESTADO_-_TJMG.xes'
algs = ['IMf', 'IMd', 'ETM']


log = xes_importer.apply(log_path)
variants_count1 = case_statistics.get_variant_statistics(log)

log_25_freq = subset_alg.filter_variants_frequency(log, 0.25)
variants_count2 = case_statistics.get_variant_statistics(log_25_freq)

log_50_freq = subset_alg.filter_variants_frequency(log, 0.5)
variants_count3 = case_statistics.get_variant_statistics(log_50_freq)

log_25_rand = subset_alg.random_filter_variants(log, 0.25)
variants_count4 = case_statistics.get_variant_statistics(log_25_freq)

log_50_rand = subset_alg.random_filter_variants(log, 0.5)
variants_count5 = case_statistics.get_variant_statistics(log_50_freq)

for a in algs:
    pn_path = 'petri_nets/' + a + '/DocumentProcessingENG.pnml'
    net, im, fm = pnml_importer.apply(pn_path)

    # tempo_inicial = time.time()
    # fitness_freq_50 = replay_fitness_evaluator.apply(log_50_freq, net, im, fm, 
    #                         variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
    # tempo_final = time.time()
    # tempo_50_freq_fit = tempo_final - tempo_inicial

    # tempo_inicial = time.time()
    # fitness_freq_25 = replay_fitness_evaluator.apply(log_25_freq, net, im, fm, 
    #                         variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
    # tempo_final = time.time()
    # tempo_25_freq_fit = tempo_final - tempo_inicial

    # tempo_inicial = time.time()
    # fitness_rand_50 = replay_fitness_evaluator.apply(log_50_rand, net, im, fm, 
    #                         variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
    # tempo_final = time.time()
    # tempo_50_rand_fit = tempo_final - tempo_inicial

    # tempo_inicial = time.time()
    # fitness_rand_25 = replay_fitness_evaluator.apply(log_25_rand, net, im, fm, 
    #                         variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)["log_fitness"]
    # tempo_final = time.time()
    # tempo_25_rand_fit = tempo_final - tempo_inicial


    tempo_inicial = time.time()
    precision_rand_25 = precision_evaluator.apply(log_25_rand, net, im, fm, 
                                variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    tempo_final = time.time()
    tempo_25_rand_prec = tempo_final - tempo_inicial
    
    tempo_inicial = time.time()
    precision_freq_25 = precision_evaluator.apply(log_25_freq, net, im, fm, 
                                variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    tempo_final = time.time()
    tempo_25_freq_prec = tempo_final - tempo_inicial


    print('### alg: ' + str(a))

    print()

    # print('fitness_freq_50: ' + str(fitness_freq_50))
    # print('fitness_freq_25: ' + str(fitness_freq_25))
    # print('fitness_rand_50: ' + str(fitness_rand_50))
    # print('fitness_rand_25: ' + str(fitness_rand_25))
    print('precision_freq_25: ' + str(precision_freq_25))
    print('precision_rand_25: ' + str(precision_rand_25))

    print()

    # print('tempo_50_freq_fit: ' + str(tempo_50_freq_fit))
    # print('tempo_25_freq_fit: ' + str(tempo_25_freq_fit))
    print('tempo_25_freq_prec: ' + str(tempo_25_freq_prec))
    # print('tempo_50_rand_fit: ' + str(tempo_50_rand_fit))
    # print('tempo_25_rand_fit: ' + str(tempo_25_rand_fit))
    print('tempo_25_rand_prec: ' + str(tempo_25_rand_prec))


