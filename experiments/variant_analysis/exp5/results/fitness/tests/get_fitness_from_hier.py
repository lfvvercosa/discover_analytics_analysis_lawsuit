import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.approaches.core.metrics.MarkovMeasures import MarkovMeasures
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.FindClustersHier import FindClustersHierarchy

import numpy as np


def parse_dend_from_results(my_file):
    with open(my_file, 'r') as f:
        last_line = f.readlines()[-2]

    if 'dend:' in last_line:
        return eval(last_line[6:-1])
    else:
        raise Exception('Dend was not found!')


def find_best_match_clusters(Z, clusters_number, min_size, log, traces):
    curr_min_size = min_size
    logs = None
    

    while logs is None:
        logs = find_clusters.get_n_clusters_hier(Z, 
                                     clusters_number, 
                                     curr_min_size, 
                                     log, 
                                     traces)
        curr_min_size -= 1

        if curr_min_size <= 1:
            raise Exception('Min Size <= 1, clusters were not found!')
    

    return logs


def remove_outliers_distrib(distrib, m=2):
    data = np.array(distrib)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    
    
    return list(data[s < m])


def reject_outliers(distrib, m=2):
    data = np.array(distrib)


    return list(data[abs(data - np.median(data)) <= m * np.std(data)])


if __name__ == '__main__':
    log_file = 'xes_files/test_variants/exp5/exp5.xes'
    result_file = 'experiments/variant_analysis/exp5/results/' + \
                  '1step_log_crossf.txt'
    clusters_number = 5
    k_markov = 2

    ## Obtain hierarchical clustering Z variable
    dend = parse_dend_from_results(result_file)
    agglomClust = CustomAgglomClust()   
    
    Z = agglomClust.gen_Z(dend)
    
    ## Obtain number of variants in log
    full_log = xes_importer.apply(log_file, 
                             variant=xes_importer.Variants.LINE_BY_LINE)
    df = convert_to_dataframe(full_log)
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()

    variants_number = len(traces)

    ## Set initial minimum cluster size
    min_size = int(variants_number/clusters_number)
    
    ## Find clusters based on hierarchy
    find_clusters = FindClustersHierarchy()
    logs = find_best_match_clusters(Z, 
                                    clusters_number, 
                                    min_size, 
                                    full_log, 
                                    traces)

    ## Get weighted fitness for clusters
    fit = 0
    complexity = 0
    total_variants = 0

    for idx,log_cluster in enumerate(logs):
        ## Show histogram Markov edges frequency distribution
        markov_measures = MarkovMeasures(log_cluster, k_markov)

        distrib = markov_measures.get_edges_freq_distrib()
        # distrib = reject_outliers(distrib, m=3)
        number_variants = \
            len(pm4py.get_variants_as_tuples(log_cluster).keys())
        total_variants += number_variants

        cluster_fit = round(markov_measures.get_fitness_mean2(n=1),4)
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
