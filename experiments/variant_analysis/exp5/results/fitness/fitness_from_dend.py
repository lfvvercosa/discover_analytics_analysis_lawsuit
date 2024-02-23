import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.approaches.core.metrics.MarkovMeasures import MarkovMeasures
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.FindClustersHier import FindClustersHierarchy
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity


import numpy as np


if __name__ == '__main__':
    log_file = 'xes_files/test_variants/exp5/exp5.xes'
    clusters_number = 5
    k_markov = 2
    result_file1 = 'experiments/variant_analysis/exp5/results/' + \
                  '1step_log_crossf.txt'
    result_file2 = 'experiments/variant_analysis/exp5/results/' + \
                  '2step_2gram_dbs_crossf_log_1.txt'
    result_file3 = 'experiments/variant_analysis/exp5/results/' + \
                  '2step_2gram_dbs_crossf_log_2.txt'
    result_file4 = 'experiments/variant_analysis/exp5/results/' + \
                  '2step_ngram_kms_crossf.txt'
    result_file5 = 'experiments/variant_analysis/exp5/results/' + \
                  '2step_leven_dbs_cross_fit.txt'

    
    result_file_test = 'experiments/variant_analysis/test/results/' + \
                       '2step_2gram_dbs_crossf_log_1.txt'
    
    fit_complex = FindFitnessComplexity()

    fit, complex = fit_complex.get_metrics_from_result(log_file, 
                                                       result_file5, 
                                                       clusters_number, 
                                                       k_markov,
                                                       True)
    
    print('fit: ', fit)
    print('complex: ', complex)

    print()

    # fit, complex = fit_complex.get_fitness_and_complexity(log_file, 
    #                                                       result_file3, 
    #                                                       clusters_number, 
    #                                                       k_markov)

    # print('fit (2step_2gram_dbs_crossf_log_2): ', fit)
    # print('complex (2step_2gram_dbs_crossf_log_2): ', complex)
