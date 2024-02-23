import pandas as pd
from scipy.cluster import hierarchy 
from collections import Counter

import pm4py
from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.utils.Utils import Utils
        

class FindClustersHierarchy:
    DEBUG = True


    def __map_dict_gd_dict_var(self, dict_gd):
        dict_var = {}

        for k in dict_gd:
            dict_var[k] = []
            for l in dict_gd[k]:
                dict_var[k] += dict_gd[k][l]
        

        return dict_var


    def __map_clusters(self, dict_gd, traces, labels):
        dict_var = self.__map_dict_gd_dict_var(dict_gd)
        utils = Utils()
        y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

        return y_pred


    def __has_n_big_clusters(self, bigger_clusters, n, min_size):
        if len(bigger_clusters) >= n:
            return bigger_clusters[n-1][1] >= min_size
        else:
            return None
        

    def get_n_clusters_hier(self, Z, clusters_number, min_size, log, traces, dict_gd):
        max_val = max(Z[:,2])

        min_perc = 0
        max_perc = 0.9
        step_perc = 0.025
        
        perc = max_perc

        while perc >= min_perc:
            labels = hierarchy.fcluster(Z=Z, 
                                        t=perc*max_val, 
                                        criterion='distance')
            labels = [l-1 for l in labels]

            if dict_gd is not None:
                labels = self.__map_clusters(dict_gd, traces, labels)

            counter = Counter(labels)
            bigger_clusters = counter.most_common(clusters_number)
            
            res = self.__has_n_big_clusters(bigger_clusters, 
                                            clusters_number, 
                                            min_size)
            
            # if res is None:
            #     return None
            # else:
            
            if res:
                return self.__create_n_clusters(log, 
                                                bigger_clusters, 
                                                labels, 
                                                traces)


            perc -= step_perc


        return None


    def get_n_clusters(self, clusters_number, min_size, log, traces, labels):
        counter = Counter(labels)
        bigger_clusters = counter.most_common(clusters_number)
        
        res = self.__has_n_big_clusters(bigger_clusters, 
                                        clusters_number, 
                                        min_size)
        
        if res:
            return self.__create_n_clusters(log, 
                                            bigger_clusters, 
                                            labels, 
                                            traces)
        

        return None

    
    def __create_n_clusters(self, log, bigger_clusters, labels, traces):
        df_variants, df_distrib = self.__get_df_with_labels(log, labels)
        logs = []

        for (c,_) in bigger_clusters:
            df_c = df_variants[df_variants['cluster_label'] == c]
            logs.append(pm4py.convert_to_event_log(df_c))

        if self.DEBUG:
            dict_var = self.__get_traces_by_cluster(traces, labels)
            ids = [c[0] for c in bigger_clusters]
            variants = []

            for log in logs:
                variants.append(list(pm4py.get_variants_as_tuples(log).keys()))

        logs_new = self.__allocate_remaining(logs, 
                                             labels, 
                                             bigger_clusters, 
                                             df_variants)

        if self.DEBUG:
            variants_new = []

            for log in logs_new:
                variants_new.append(list(pm4py.get_variants_as_tuples(log).keys()))


        return logs_new


    def __get_traces_by_cluster(self, traces, labels):
        stats = StatsClust()
        dict_var = stats.get_variants_by_cluster(traces, labels)


        return dict_var


    def __get_df_with_labels(self, log, labels):
        df = pm4py.convert_to_dataframe(log)
        
        split_join = SplitJoinDF(df)
        traces = split_join.split_df()
        ids = split_join.ids
        ids_clus = [l[0] for l in ids]
        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
        
        # get df-log with cluster labels
        split_join.join_df(labels)
        df = split_join.df
        df_variants = split_join.join_df_uniques(labels)

        stats = StatsClust()
        df_distrib = stats.get_distrib(df, df_ids)


        return df_variants, df_distrib


    def __allocate_remaining(self, logs, labels, bigger_clusters, df_variants):
        cluster_ids = [c[0] for c in bigger_clusters]
        mapping = {}

        for c in labels:
            if c not in cluster_ids and c not in mapping:
                df = df_variants[df_variants['cluster_label'] == c]
                log_comp = pm4py.convert_to_event_log(df)
                mapping[c] = (self.__get_closer_cluster(logs, log_comp), df.copy())
        

        return self.__allocate(mapping, logs)


    def __get_closer_cluster(self, logs, log_comp):
        fit = -1
        pos = -1

        for idx,log in enumerate(logs):
            fit_list = logs_alignments.apply(log_comp, log)
            curr_fit = self.__get_mean_fitness(fit_list)

            if curr_fit > fit:
                fit = curr_fit
                pos = idx
        

        return pos
    

    def __get_mean_fitness(self, fit_list):
        fit_sum = 0

        for fit in fit_list:
            fit_sum += fit['fitness']

        return fit_sum/len(fit_list)
    

    def __allocate(self, mapping, logs):
        merge = {}
        logs_new = []


        for c in mapping:
            if mapping[c][0] not in merge:
                merge[mapping[c][0]] = []
            
            merge[mapping[c][0]].append(mapping[c][1])

        for c in merge:
            df = pd.concat(merge[c])
            df_c = pm4py.convert_to_dataframe(logs[c])

            df_new = pd.concat([df,df_c])
            logs_new.append(pm4py.convert_to_event_log(df_new))

        for idx in range(len(logs)):
            if idx not in merge:
                logs_new.append(logs[idx])

        
        return logs_new




