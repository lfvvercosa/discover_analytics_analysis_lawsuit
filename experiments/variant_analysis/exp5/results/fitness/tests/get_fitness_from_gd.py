import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.approaches.core.metrics.MarkovMeasures import MarkovMeasures
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.FindClustersHier import FindClustersHierarchy

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy 



def reject_outliers(distrib, m=2):
    data = np.array(distrib)


    return list(data[abs(data - np.median(data)) <= m * np.std(data)])



if __name__ == '__main__':
    log1_file = 'xes_files/test_variants/exp5/exp5_0.xes'
    log2_file = 'xes_files/test_variants/exp5/exp5_1.xes'
    log3_file = 'xes_files/test_variants/exp5/exp5_2.xes'
    log4_file = 'xes_files/test_variants/exp5/exp5_3.xes'
    log5_file = 'xes_files/test_variants/exp5/exp5_4.xes'
    
    log1 = xes_importer.apply(log1_file, variant=xes_importer.Variants.LINE_BY_LINE)
    log2 = xes_importer.apply(log2_file, variant=xes_importer.Variants.LINE_BY_LINE)
    log3 = xes_importer.apply(log3_file, variant=xes_importer.Variants.LINE_BY_LINE)
    log4 = xes_importer.apply(log4_file, variant=xes_importer.Variants.LINE_BY_LINE)
    log5 = xes_importer.apply(log5_file, variant=xes_importer.Variants.LINE_BY_LINE)
    
    logs = [log1, log2, log3, log4, log5]
    k_markov = 2
    n = 1

    ## Get weighted fitness for clusters
    fit = 0
    complexity = 0
    total_variants = 0


    for idx,log_cluster in enumerate(logs):
        ## Show histogram Markov edges frequency distribution
        markov_measures = MarkovMeasures(log_cluster, k_markov)

        # distrib = markov_measures.get_edges_freq_distrib()

        # plt.hist(distrib, 20)
        # plt.show(block=True)

        # distrib2 = reject_outliers(distrib, m=3)

        # plt.hist(distrib2, 20)
        # plt.show(block=True)

        number_variants = \
            len(pm4py.get_variants_as_tuples(log_cluster).keys())
        total_variants += number_variants

        cluster_fit = round(markov_measures.get_fitness_mean2(n),4)
        fit += cluster_fit * number_variants

        cluster_complex = round(markov_measures.get_network_complexity(),4)
        complexity += cluster_complex * number_variants

        print('cluster ' + str(idx) + ' fitness: ' + str(cluster_fit))
        print('cluster ' + str(idx) + ' number of variants: ' + str(number_variants))
        print()

        print('cluster ' + str(idx) + ' complexity: ' + str(cluster_complex))
        print()

    fit /= total_variants
    complexity /= total_variants

    print('weighted fitness: ', round(fit,3))
    print('weighted network complexity: ', round(complexity,3))