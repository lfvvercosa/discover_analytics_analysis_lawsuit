import pandas as pd
import statistics as stats
import pm4py
import Levenshtein 
from weighted_levenshtein import dam_lev
import time
import warnings
from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
from scipy.cluster import hierarchy 
from collections import Counter


from clustering.SplitJoinDF import SplitJoinDF
from clustering.StatsClust import StatsClust
        

class FindClustersHierarchy:
    DEBUG = True
    
    # pm4py.utils.warnings.simplefilter('ignore')
    warnings.filterwarnings("ignore")


    def __map_dict_gd_dict_var(self, dict_gd):
        dict_var = {}

        for k in dict_gd:
            dict_var[k] = []
            for l in dict_gd[k]:
                dict_var[k] += dict_gd[k][l]
        

        return dict_var


    def __has_n_clusters_bigger_than(self, bigger_clusters, n, min_size):
        if len(bigger_clusters) >= n:
            return bigger_clusters[n-1][1] >= min_size
        else:
            return False
        
    
    def __has_n_clusters_smaller_than(self, bigger_clusters, n, max_size):
        if len(bigger_clusters) >= n:
            return bigger_clusters[0][1] <= max_size
        else:
            return False
        

    def get_n_clusters_hier(self, 
                            Z, 
                            dist_matrix,
                            clusters_number, 
                            min_size, 
                            max_size,
                            method, 
                            log, 
                            traces,
                            metric,
                            ins_del_weight,
                            subst_weigh,
                            trans_weight,):

        perc = 0.9
        max_val = max(Z[:,-2])
        bigger_than = False
        smaller_than = False

        while perc > 0:

            labels = hierarchy.fcluster(Z=Z, 
                                        t=perc * max_val, 
                                        criterion='distance')
            
            labels = [l-1 for l in labels]
            counter = Counter(labels)
            bigger_clusters = counter.most_common(clusters_number)
            
            if not bigger_than:
                bigger_than = self.__has_n_clusters_bigger_than(bigger_clusters, 
                                                                clusters_number, 
                                                                min_size)

            if not smaller_than:
                smaller_than = self.__has_n_clusters_smaller_than(bigger_clusters,
                                                                  clusters_number, 
                                                                  max_size)

            # if bigger_than:
            #     print('### min size satisfied')

            # if smaller_than:
            #     print('### max size satisfied')


            if bigger_than and smaller_than:
                print('bigger clusters: ' + str(bigger_clusters))
                return self.__create_n_clusters2(log, 
                                                 dist_matrix,
                                                 bigger_clusters, 
                                                 labels, 
                                                 traces,
                                                 method,
                                                 metric,
                                                 ins_del_weight,
                                                 subst_weigh,
                                                 trans_weight,
                                                )
            
            if not self.is_min_size_reachable(bigger_clusters, 
                                              min_size, 
                                              clusters_number):
                return None
            
            perc -= 0.05

        return None


    def is_min_size_reachable(self, bigger_clusters, min_size, clusters_number):
        needed = min_size * clusters_number
        remaining = 0

        for t in bigger_clusters:
            if t[1] >= min_size:
                remaining += t[1]
            else:
                break

        
        return remaining >= needed


    def get_n_clusters(self, clusters_number, min_size, log, traces, labels):
        counter = Counter(labels)
        bigger_clusters = counter.most_common(clusters_number)
        
        res = self.__has_n_clusters_bigger_than(bigger_clusters, 
                                        clusters_number, 
                                        min_size)
        
        if res:
            return self.__create_n_clusters(log, 
                                            bigger_clusters, 
                                            labels, 
                                            traces)
        

        return None

    
    def __create_n_clusters(self, log, bigger_clusters, labels, traces):
        df_variants = self.__get_df_with_labels(log, labels)
        logs = {}

        for (c,_) in bigger_clusters:
            df_c = df_variants[df_variants['cluster_label'] == c]
            logs[c] = pm4py.convert_to_event_log(df_c)

        # print('### allocate remaining')
        labels_new = self.__allocate_remaining(logs, 
                                               labels, 
                                               bigger_clusters, 
                                               df_variants)


        return labels_new


    def __create_n_clusters2(self, 
                             log, 
                             dist_matrix,
                             bigger_clusters, 
                             labels, 
                             traces, 
                             method,
                             metric,
                             ins_del_weight,
                             subst_weigh,
                             trans_weight,
                            ):
        clusters = {}
        remaining = {}
        bigger_clusters_labels = [x[0] for x in bigger_clusters]

        for idx,l in enumerate(labels):
            if l in bigger_clusters_labels:
                if l not in clusters:
                    clusters[l] = []
                clusters[l].append(idx)
            else:
                if l not in remaining:
                    remaining[l] = []
                remaining[l].append(idx)

        print('### allocate remaining')

        labels_new = self.__allocate_remaining2(clusters, 
                                                remaining,
                                                dist_matrix,
                                                labels, 
                                                traces,
                                                method,
                                                metric,
                                                ins_del_weight,
                                                subst_weigh,
                                                trans_weight,
                                               )


        return labels_new

    # def __get_traces_by_cluster(self, traces, labels):
    #     stats = StatsClust()
    #     dict_var = stats.get_variants_by_cluster(traces, labels)


    #     return dict_var


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


        return df_variants


    def __allocate_remaining(self, logs, labels, bigger_clusters, df_variants):
        cluster_ids = [c[0] for c in bigger_clusters]
        mapping = {}

        print('allocating remaining records...')

        for c in labels:
            if c not in cluster_ids and c not in mapping:
                df = df_variants[df_variants['cluster_label'] == c]
                log_comp = pm4py.convert_to_event_log(df)
                mapping[c] = self.__get_closer_cluster(logs, log_comp)
        

        return self.__update_labels(mapping, labels)


    def __allocate_remaining2(self, 
                              clusters,
                              remaining, 
                              dist_matrix,
                              labels,
                              traces,
                              method,
                              metric,
                              ins_del_weight,
                              subst_weigh,
                              trans_weight,
                             ):
        mapping = {}
        total_variants = len(traces)

        for l in remaining:
            mapping[l] = self.get_closer_cluster2(remaining[l],
                                                  clusters,
                                                  dist_matrix,
                                                  total_variants,
                                                  traces,
                                                  method,
                                                  metric,
                                                  ins_del_weight,
                                                  subst_weigh,
                                                  trans_weight
                                                 )


        return self.__update_labels(mapping, labels)


    def __get_closer_cluster(self, logs, log_comp):
        fit = -1
        new_label = -1

        for label in logs:
            fit_list = logs_alignments.apply(log_comp, logs[label])
            curr_fit = self.__get_mean_fitness(fit_list)

            if curr_fit > fit:
                fit = curr_fit
                new_label = label
        

        return new_label
    

    def get_position_in_matrix(self, m, i, j):
        return m * i + j - ((i + 2) * (i + 1)) // 2


    def get_closer_cluster2(self, 
                            group, 
                            clusters, 
                            dist_matrix,
                            total_variants,
                            traces,
                            method,
                            metric,
                            ins_del_weight,
                            subst_weigh,
                            trans_weight):
        best_dist = float('inf')
        best_clus = None

        for label in clusters:
            dists = []
            metric_dist = None


            if dist_matrix is not None:
                for idx in group:
                    for idx2 in clusters[label]:
                        i = min(idx,idx2)
                        j = max(idx,idx2)
                        pos = self.get_position_in_matrix(total_variants,
                                                        i,
                                                        j)
                        dists.append(dist_matrix[pos])
            else:
                for trace in group:
                    for trace2 in clusters[label]:
                        if metric == 'levenshtein':
                            dists.append(self.leven(trace,
                                                    trace2))
                        elif metric == 'weighted_levenshtein':
                            dists.append(self.weighted_leven(
                                            trace,
                                            trace2,
                                            ins_del_weight,
                                            trans_weight,
                                            subst_weigh
                                        ))
            
            if method == 'average':
                metric_dist = stats.mean(dists)
            elif method == 'single':
                metric_dist = min(dists)
            elif method == 'complete':
                metric_dist = max(dists)
            else:
                raise Exception('invalid metric')
            
            if metric_dist < best_dist:
                best_dist = metric_dist
                best_clus = label
        
        clusters[best_clus] += group


        return best_clus


    def leven(self, a, b):
        norm = (len(a) + len(b))
        # lev = dam_lev(a,b)
        
        lev = Levenshtein.distance(a, b)
        
        
        return lev / norm
    

    def weighted_leven(self, a, b, ins_del_weight, trans_weight, subst_weigh):
        norm = (len(a) + len(b))

        dist = dam_lev(a,
                       b,
                       insert_costs = ins_del_weight, 
                       delete_costs = ins_del_weight,
                       transpose_costs = trans_weight,
                       substitute_costs = subst_weigh,
                      )
        

        return dist / norm


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
    

    def __update_labels(self, mapping, labels):
        new_labels = []
        
        labels.copy()

        for l in labels:
            if l not in mapping:
                new_labels.append(l)
            else:
                new_labels.append(mapping[l])


        return new_labels
    

    def find_best_match_clusters_hier(self, 
                                      Z,
                                      dist_matrix, 
                                      clusters_number, 
                                      min_size, 
                                      max_size,
                                      method,
                                      log, 
                                      traces,
                                      metric,
                                      ins_del_weight,
                                      subst_weigh,
                                      trans_weight
                                      ):
        curr_min_size = min_size
        curr_max_size = max_size
        count = 1
        count2 = 1
        final_labels = None

        # print('### finding best clusters match')

        while final_labels is None:
            # print('### curr_max_size: ' + str(curr_max_size))
            # print('### curr_min_size: ' + str(curr_min_size))
            final_labels = self.get_n_clusters_hier(Z,
                                                  dist_matrix, 
                                                  clusters_number, 
                                                  curr_min_size,
                                                  max_size, 
                                                  method,
                                                  log, 
                                                  traces,
                                                  metric,
                                                  ins_del_weight,
                                                  subst_weigh,
                                                  trans_weight,
                                                  )

            if curr_min_size < 1 and curr_max_size > len(traces):
                # return None
                raise Exception('Min Size <= 1, clusters were not found!')
        
            if curr_min_size < 1:
                count = 0
                curr_max_size = (1 + count2/2) * max_size
                count2 += 1

            curr_min_size = (1 - count/10) * min_size
            count += 1

        
        return final_labels
    

    def get_clusters(self, logs):
        count = 0
        dfs = []

        for log in logs:
            df_log = pm4py.convert_to_dataframe(log)
            df_log['cluster_label'] = count
            dfs.append(df_log)
            count += 1

        df = pd.concat(dfs)
        df = df[['case:concept:name','cluster_label']]


        return df.drop_duplicates(subset='case:concept:name')
    

if __name__ == "__main__":



    print()