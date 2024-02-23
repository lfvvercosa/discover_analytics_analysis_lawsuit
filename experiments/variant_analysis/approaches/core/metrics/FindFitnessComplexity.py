import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.approaches.core.metrics.MarkovMeasures import MarkovMeasures
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.FindClustersHier import FindClustersHierarchy

import numpy as np


class FindFitnessComplexity:
    DEBUG = False

    def parse_dend_from_results(self, my_file):
        with open(my_file, 'r') as f:
            last_line = f.readlines()[-2]

        if 'dend:' in last_line:
            return eval(last_line[6:-1])
        else:
            raise Exception('Dend was not found!')


    def parse_dict_gd_from_results(self, my_file):
        with open(my_file, 'r') as f:
            line = f.readlines()[-4]

        if 'dict_gd:' in line:
            return eval(line[9:-1])
        else:
            raise Exception('Dict_gd was not found!')
    

    def find_best_match_clusters_hier(self, 
                                 Z, 
                                 clusters_number, 
                                 min_size, 
                                 log, 
                                 traces, 
                                 dict_gd):
        curr_min_size = min_size
        logs = None
        find_clusters = FindClustersHierarchy()

        while logs is None:
            logs = find_clusters.get_n_clusters_hier(Z, 
                                                clusters_number, 
                                                curr_min_size, 
                                                log, 
                                                traces,
                                                dict_gd)
            curr_min_size -= 1

            if curr_min_size <= 1:
                return None
                # raise Exception('Min Size <= 1, clusters were not found!')
        

        return logs


    def find_best_match_clusters(self, 
                                 clusters_number, 
                                 min_size, 
                                 log, 
                                 traces,
                                 labels 
                                 ):
        
        curr_min_size = min_size
        logs = None
        find_clusters = FindClustersHierarchy()

        while logs is None:
            logs = find_clusters.get_n_clusters(
                                                clusters_number, 
                                                curr_min_size, 
                                                log, 
                                                traces,
                                                labels
                                                )
            curr_min_size -= 1

            if curr_min_size <= 1:
                break
                # raise Exception('Min Size <= 1, clusters were not found!')
        

        return logs
    

    def reject_outliers(self, distrib, m=2):
        data = np.array(distrib)


        return list(data[abs(data - np.median(data)) <= m * np.std(data)])
    

    def get_metrics_from_result(self, 
                                log_file, 
                                result_file, 
                                clusters_number, 
                                k_markov,
                                is_two_steps_clust):

        ## Obtain hierarchical clustering Z variable
        dend = self.parse_dend_from_results(result_file)

        ## Get dict_gd in case it is a two step clustering algorithm
        dict_gd = None

        if is_two_steps_clust:
            dict_gd = self.parse_dict_gd_from_results(result_file)

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
        logs = self.find_best_match_clusters_hier(Z, 
                                             clusters_number, 
                                             min_size, 
                                             full_log, 
                                             traces,
                                             dict_gd)

        
        return self.get_metrics(logs, k_markov)
    

    def get_metrics(self, logs, k_markov):
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

            if self.DEBUG:
                print('cluster ' + str(idx) + ' fitness: ' + str(cluster_fit))
                print('cluster ' + str(idx) + ' number of variants: ' + str(number_variants))
                print()

                print('cluster ' + str(idx) + ' complexity: ' + str(cluster_complex))
                print()

        fit /= total_variants
        complexity /= total_variants

        
        return fit, complexity
    

    def get_metrics_from_simulation(self, df_variants, k_markov):
        logs = self.split_logs(df_variants)


        return self.get_metrics(logs, k_markov)
    

    def split_logs(self, df_variants):
        logs = []
        clusters = set(df_variants['cluster_label'].to_list())

        for c in clusters:
            df_log = df_variants[df_variants['cluster_label'] == c]
            logs.append(pm4py.convert_to_event_log(df_log))

        
        return logs
