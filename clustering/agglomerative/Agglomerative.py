import matplotlib.pyplot as plt
import pandas as pd
import Levenshtein 
import numpy as np
import pm4py
import pickle

from pathlib import Path
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist 
from weighted_levenshtein import dam_lev

from clustering.SplitJoinDF import SplitJoinDF
from clustering.agglomerative.FindClustersHier import FindClustersHierarchy


class Agglomerative():
    
    df_map = None
    variants = None
    split_join = None
    labels_train = None
    is_map_act = None
    map_act = None
    last_ascii_code = None
    ins_del_weight = None
    trans_weight = None
    subst_weigh = None
    metric = None
    method = None
    dist_matrix = None
    ASCII_START = 65
    ASCII_END = 254


    def __init__(self, 
                 df,
                 method='single',
                 metric='levenshtein',
                 ins_del_weight=1,
                 trans_weight=1,
                 subst_weight=1, 
                 act_col='concept:name', 
                 is_map_act=True):
        
        if df is not None:
            if is_map_act:
                self.df_map = self.map_activities(df, act_col)
            else:
                self.df_map = df

            self.split_join = SplitJoinDF(self.df_map)
            self.variants = self.split_join.split_df()
            self.is_map_act = is_map_act

            # Set weights for weighted levenshtein operations
            self.set_ops_weights(ins_del_weight, 
                                 trans_weight, 
                                 subst_weight)
            self.metric = metric
            self.method = method


    def map_activities(self, df, act_col):
        acts = df[act_col].drop_duplicates().to_list()
        ascii_code = self.ASCII_START
        map_act = {act_col:acts, 'ascii_mapping':[]}

        for a in acts:
            map_act['ascii_mapping'].append(str(chr(ascii_code)))
            ascii_code += 1

            if ascii_code == 127:
                ascii_code += 1

            if ascii_code > self.ASCII_END:
                raise Exception('ASCII Limit Exceeded')

        df_map = pd.DataFrame.from_dict(map_act)
        df_map = df.merge(df_map, on=act_col, how='left')
        df_map = df_map.drop(columns=act_col)
        df_map = df_map.rename(columns={'ascii_mapping':act_col})
    

        if self.map_act is None:
            self.map_act = map_act
            self.last_ascii_code = ascii_code

        return df_map
    

    def map_act_valid(self, df, act_col):
        acts = df[act_col].drop_duplicates().to_list()
        new_acts = [a for a in acts if a not in self.map_act[act_col]]

        for a in new_acts:
            self.map_act[act_col].append(a)
            self.map_act['ascii_mapping'].append(str(chr(self.last_ascii_code)))
            self.last_ascii_code += 1

            if self.last_ascii_code == 127:
                self.last_ascii_code += 1

            if self.last_ascii_code > 254:
                raise Exception('ASCII Limite Exceeded')
            
        df_map = pd.DataFrame.from_dict(self.map_act)
        df_map = df.merge(df_map, on=act_col, how='left')
        df_map = df_map.drop(columns=act_col)
        df_map = df_map.rename(columns={'ascii_mapping':act_col})


        return df_map        


    def leven(self, a, b):
        norm = (len(a) + len(b))
        # lev = dam_lev(a,b)
        
        lev = Levenshtein.distance(a, b)
        
        
        return lev / norm
    

    def weighted_leven(self, a, b):
        norm = (len(a) + len(b))

        dist = dam_lev(a,
                       b,
                       insert_costs = self.ins_del_weight, 
                       delete_costs = self.ins_del_weight,
                       transpose_costs = self.trans_weight,
                       substitute_costs = self.subst_weigh,
                      )
        

        return dist / norm
    

    def lev_metric(self, x, y):
        i, j = int(x[0]), int(y[0])     # extract indices


        return self.leven(self.variants[i], self.variants[j])
    

    def weighted_leven_metric(self, x, y):
        i, j = int(x[0]), int(y[0])     # extract indices


        return self.weighted_leven(self.variants[i], self.variants[j])
    

    def cluster(self, method, metric, n_clusters):
        X = np.arange(len(self.variants)).reshape(-1, 1)

        if metric == 'levenshtein':
            metric = self.lev_metric
        elif metric == 'weighted_levenshtein':
            metric = self.weighted_leven_metric
        else:
            raise Exception('undefined metric!')

        Z = hierarchy.linkage(X, method=method, metric=metric)
        labels = hierarchy.fcluster(Z=Z, t=n_clusters, criterion='maxclust')


        return labels
    

    def create_dendrogram(self, dist_matrix, method, metric):
        # X = np.arange(len(self.variants)).reshape(-1, 1)

        if metric == 'levenshtein':
            metric = self.lev_metric
        elif metric == 'weighted_levenshtein':
            metric = self.weighted_leven_metric
        else:
            raise Exception('undefined metric!')


        return hierarchy.linkage(dist_matrix, method=method, metric=metric)
    

    def create_dist_matrix(self):
        print('### creating dendrogram')

        X = np.arange(len(self.variants)).reshape(-1, 1)

        if self.metric == 'levenshtein':
            metric = self.lev_metric
        elif self.metric == 'weighted_levenshtein':
            metric = self.weighted_leven_metric
        else:
            raise Exception('undefined metric!')
        

        return pdist(X, metric=metric)
    

    def run_agglomerative(self, method, metric, n_clusters):
        # extract variants from log
        split_join = SplitJoinDF(self.df_map)
        self.variants = split_join.split_df()

        labels = self.cluster(method, metric, n_clusters)
        df_clus = split_join.join_df(labels)


        return df_clus[['case:concept:name','cluster_label']]


    def set_ops_weights(self, ins_del_weight, trans_weight, subst_weight):
        self.ins_del_weight = np.full(self.ASCII_END + 1, ins_del_weight, dtype=np.float64)
        self.subst_weigh = np.full((self.ASCII_END + 1, self.ASCII_END + 1), 
                                    subst_weight, dtype=np.float64)
        self.trans_weight = np.full((self.ASCII_END + 1, self.ASCII_END + 1), 
                                    trans_weight, dtype=np.float64)
        

    def run(self, 
            n_clusters, 
            max_size_perc,
            dist_matrix=None 
           ):

        log_map = pm4py.convert_to_event_log(self.df_map)
        find_clusters = FindClustersHierarchy()

        ## Set initial minimum cluster size
        total_variants = len(self.variants)
        min_size = int(total_variants/n_clusters)
        max_size_par = max_size_perc * n_clusters
        max_size = max(min_size, max_size_par * min_size)

        # print('max_size_par: ' + str(max_size_par))

        

        if dist_matrix is None:
            dist_matrix = self.create_dist_matrix()
        
        self.dist_matrix = dist_matrix
        Z = self.create_dendrogram(dist_matrix, self.method, self.metric)

        print('finding best clusters match...')
        self.labels_train = \
            find_clusters.find_best_match_clusters_hier(
                                                        Z,
                                                        dist_matrix, 
                                                        n_clusters,
                                                        min_size,
                                                        max_size,
                                                        self.method,
                                                        log_map,
                                                        self.variants,
                                                        self.metric,
                                                        self.ins_del_weight,
                                                        self.subst_weigh,
                                                        self.trans_weight,
                                                       )

        df_clus = self.split_join.join_df(self.labels_train)
        df_clus = df_clus.drop_duplicates('case:concept:name')


        return df_clus[['case:concept:name','cluster_label']]


    def validate(self, log_valid, act_col='concept:name'):
        df = convert_to_dataframe(log_valid)

        if df is not None:
            if self.map_act:
                df_map = self.map_act_valid(df, act_col)
            else:
                df_map = df

        split_join = SplitJoinDF(df_map)
        traces = split_join.split_df()
        labels_valid = []
        clusters = {}
        find_cluster = FindClustersHierarchy()

        for (l,t) in zip(self.labels_train, self.variants):
            if l not in clusters:
                clusters[l] = []
            
            clusters[l].append(t)

        for t in traces:
            labels_valid.append(find_cluster.get_closer_cluster2([t], 
                                                                 clusters,
                                                                 None,
                                                                 None, 
                                                                 traces,
                                                                 self.method,
                                                                 self.metric,
                                                                 self.ins_del_weight,
                                                                 self.subst_weigh,
                                                                 self.trans_weight
                                                                )
                               )

        df_ret = split_join.join_df(labels_valid)
        df_ret = df_ret[['case:concept:name','cluster_label']]
        df_ret = df_ret.drop_duplicates('case:concept:name')


        return df_ret
    

    def save_cash(self, path, params, var):
        Path(path).mkdir(parents=True, exist_ok=True)
        saving_path = path + str(params)
        pickle.dump(var, open(saving_path, 'wb'))


        return saving_path
    

    def retrieve_cash(self, path, params):
        retrieving_path = path + str(params)
        var = pickle.load(open(retrieving_path, 'rb'))


        return var


if __name__ == "__main__": 
    d = {'A':['Abacate','Laranja','Limão','Limão'],
         'B':['Carro','Ônibus','Caminhão Grande','Moto']}

    a = 'CBD'
    b = 'BCD'

    agglom = Agglomerative(None)
    # print(agglom.map_activities(pd.DataFrame.from_dict(d), 'A'))
    print(agglom.leven(a,b))

    # path = 'clustering/agglomerative/tests/test_dendrogram4.xes'
    path = 'clustering/test/test_dendrogram.xes'
    path_valid = 'clustering/test/test_dendrogram4_valid.xes'

    log = xes_importer.apply(path, 
                             variant=xes_importer.Variants.LINE_BY_LINE)
    df_log = pm4py.convert_to_dataframe(log)

    log_valid = xes_importer.apply(path_valid, 
                                   variant=xes_importer.Variants.LINE_BY_LINE)

    agglom = Agglomerative(df_log, 
                           'average',
                           'weighted_levenshtein',
                           1,
                           1,
                           1,
                           is_map_act=False)

    plt.figure(figsize=(8, 4))
    dist_matrix = agglom.create_dist_matrix()
    Z = agglom.create_dendrogram(dist_matrix,
                                 method='average',
                                 metric='weighted_levenshtein')
    # Set the link color palette to black
    hierarchy.set_link_color_palette(['k'])
    a = hierarchy.dendrogram(Z,
                             labels=agglom.variants,
                             color_threshold=np.inf,
                             leaf_rotation=0)
    
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylabel('Distance',labelpad=13,fontsize=17)
    plt.tight_layout()
    plt.savefig('temp/dendrograms/example_agglomerative_wl.png', dpi=400)
    
    df_clus = agglom.run( 
                         n_clusters=3,
                         max_size_perc=0.4
                        )
    
    df_clus_valid = agglom.validate(log_valid)
    
    

    print()
    # plt.show(block=True)