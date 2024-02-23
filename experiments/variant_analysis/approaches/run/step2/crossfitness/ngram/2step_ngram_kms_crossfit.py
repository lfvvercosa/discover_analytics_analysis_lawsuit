from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy 
import matplotlib.pyplot as plt
import numpy as np
import boto3
import requests

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.CustomAgglomClust \
    import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust
import models.creation.read_and_write.s3_handle as s3_handle

import pandas as pd


def rename_columns(df_gram):
    map_name = {n:str(n) for n in df_gram.columns}
    df_gram = df_gram.rename(columns=map_name)


    return df_gram
    

def cashe_df_grams(log, ngram, min_percent, max_percent):
    cashed_dfs = {}
    print('cashing df-grams...')

    for n in ngram:
        cashed_dfs[n] = {}
        for min_perc in min_percent:
            cashed_dfs[n][min_perc] = {}
            for max_perc in max_percent:
                cashed_dfs[n][min_perc][max_perc] = {}
                # convert to df
                df = convert_to_dataframe(log)

                # create n-gram
                split_join = SplitJoinDF(df)
                traces = split_join.split_df()
                ids = split_join.ids
                ids_clus = [l[0] for l in ids]
                df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                # get ground-truth
                utils = Utils()
                y_true = utils.get_ground_truth(ids_clus)

                df_gram = create_n_gram(df_clus, 'case:concept:name', 'concept:name', n)
                df_gram = rename_columns(df_gram)
                df_gram.index = pd.Categorical(df_gram.index, ids_clus)
                df_gram = df_gram.sort_index()

                preProcess = PreProcessClust()
                rem_cols = preProcess.selectColsByFreq(min_perc, max_perc, df_gram)
                df_gram = df_gram.drop(columns=rem_cols)
                df_gram = df_gram.drop(columns=['n_gram','result'], errors='ignore')

                # normalize n-gram
                df_gram_norm = df_gram.copy()

                for c in df_gram_norm.columns:
                    if df_gram_norm[c].max() != df_gram_norm[c].min():
                        df_gram_norm[c] = (df_gram_norm[c] - df_gram_norm[c].min())/\
                            (df_gram_norm[c].max() - df_gram_norm[c].min())

                df_gram_norm = df_gram_norm.round(4)

                cashed_dfs[n][min_perc][max_perc] = df_gram_norm.copy()


    return cashed_dfs


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp5/exp5.xes'
    log = xes_importer.apply(log_path)
    exp_backlog = 'experiments/variant_analysis/exp5/results/2step_ngram_kms_crossf.txt'
    bucket = 'luiz-doutorado-projetos2'
    filename = 'variant_analysis/exp5/results/2step_ngram_kms_crossf.txt'

    params_agglom = {}
    params_agglom['custom_distance'] = 'alignment_between_logs'
    params_agglom['DEBUG'] = True
    # params_agglom['AWS_bucket'] = bucket

    # clusters = [4,5,6,7,8,9,10]
    # ngram = [1,2]
    # min_percents = [0, 0.1, 0.2, 0.3]
    # max_percents = [1, 0.9, 0.8, 0.7]
    # reps = 3

    clusters = [20]
    ngram = [2]
    min_percents = [0.15]
    max_percents = [1]
    reps = 1

    best_ARI = -1
    best_ARI_list = None
    best_Vm = -1
    best_clusters = -1
    best_ngram = -1
    best_min_percents = -1
    best_max_percents = -1
    best_df_distrib_1 = None
    best_df_distrib_2 = None
    best_dict_var = None
    best_dict_gd = None
    best_dict_gd2 = None


    total = len(clusters) * len(ngram) * len(min_percents) * \
            len(max_percents) * reps
    count = 0

    cashed_dfs = cashe_df_grams(log, ngram, min_percents, max_percents)

    for n_clusters in clusters:
        print('total clusters: ', n_clusters)
        for n in ngram:
            print('ngram: ',n)
            for min_perc in min_percents:
                for max_perc in max_percents:
                    ari_avg = 0
                    vm_avg = 0

                    if count % 10 == 0:
                        print('progress: ' + str(round((count/total)*100,2)) + '%\n')

                    count += 1
                    ARI_list = []

                    for i in range(reps):

                        df = convert_to_dataframe(log)
                        split_join = SplitJoinDF(df)
                        traces = split_join.split_df()
                        ids = split_join.ids
                        ids_clus = [l[0] for l in ids]
                        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                        df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                        df_gram_norm = cashed_dfs[n][min_perc][max_perc]

                        # apply first-step clustering (KMeans)
                        model = KMeans(n_clusters=n_clusters)
                        model.fit(df_gram_norm)
                        cluster_labels = list(model.labels_)

                        centroids = model.cluster_centers_
                        centroids = np.around(centroids, decimals=4)
                        df_centroids = pd.DataFrame(centroids, 
                                                    index=list(range(n_clusters)),
                                                    columns=df_gram_norm.columns)
                        # df_centroids.to_csv('temp/df_centroids.csv', sep='\t')

                        # print(list(set(cluster_labels)))
                        # print(cluster_labels)

                        # get performance by adjusted rand-score metric
                        utils = Utils()
                        y_pred = cluster_labels
                        y_true = utils.get_ground_truth(ids_clus)

                        if i == reps - 1:
                            # get df-log with cluster labels
                            split_join.join_df(cluster_labels)
                            df = split_join.df
                            dict_labels = {'index':ids_clus, 
                                           'cluster_label':cluster_labels}
                            df_labels = pd.DataFrame.from_dict(dict_labels)
                            df_labels = df_labels.set_index('index')
                            df_gram_norm = df_gram_norm.join(df_labels, how='left')

                            # get distribution of traces cluster per variant
                            stats = StatsClust()
                            df_distrib = stats.get_distrib(df, df_ids)

                            # get variants by cluster
                            dict_var = stats.get_variants_by_cluster(traces, 
                                                                     cluster_labels)

                            # get variants ground truth by cluster
                            dict_gd = stats.get_ground_truth_by_cluster(dict_var, 
                                                                        traces, 
                                                                        y_true)

                        
                        ARI = adjusted_rand_score(y_true, y_pred)
                        ari_avg += ARI
                        ARI_list.append(ARI)
                    
                        Vm = v_measure_score(y_true, y_pred, beta=0.2)
                        vm_avg += Vm
                    
                    ari_avg /= reps
                    vm_avg /= reps

                    if vm_avg > best_Vm:
                        print('best_Vm: ', vm_avg)
                        # best_ARI = ari_avg
                        # best_ARI_list = ARI_list.copy()
                        best_Vm = vm_avg
                        best_clusters = n_clusters
                        best_ngram = n
                        best_min_percents = min_perc
                        best_max_percents = max_perc
                        best_df_distrib_1 = df_distrib.copy()
                        best_dict_var = dict_var.copy()
                        best_dict_gd = dict_gd.copy()

    print('best params')
    print('best_ARI: ' + str(best_ARI))
    print('best_VM: ' + str(best_Vm))
    print('best_clusters: ' + str(best_clusters))
    print('best_ngram: ' + str(best_ngram))
    print('best_min_percents: ' + str(best_min_percents))
    print('best_max_percents: ' + str(best_max_percents))

    # get dendrogram-list using cross-fitness
    agglomClust = CustomAgglomClust()
    dend = agglomClust.agglom_fit(df, params_agglom)

    # print('dend: ', dend)

    if dend:
        Z = agglomClust.gen_Z(dend)

        # get best number of clusters
        t = max(Z[:,2])

        # get variants by cluster
        dict_var = stats.\
            get_variants_by_cluster(traces, cluster_labels)

        min_perc_dendro = 0.3
        max_perc_dendro = 0.9
        step_perc = 0.025
        perc = min_perc_dendro


        while perc <= max_perc_dendro:
            labels = hierarchy.fcluster(Z=Z, 
                                        t=perc*t, 
                                        criterion='distance')
            labels = [l-1 for l in labels]

            # get performance by adjusted rand-score metric
            
            y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)


            ARI = adjusted_rand_score(y_true, y_pred)

            if ARI > best_ARI:
                # get variants by cluster
                best_dict_var = stats.get_variants_by_cluster(traces, y_pred)
                print()

                # get variants ground truth by cluster
                best_dict_gd2 = stats.get_ground_truth_by_cluster(best_dict_var, 
                                                                traces, 
                                                                y_true)
                
                df = convert_to_dataframe(log)
                split_join = SplitJoinDF(df)
                traces = split_join.split_df()
                ids = split_join.ids
                ids_clus = [l[0] for l in ids]
                df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

                # get df-log with cluster labels
                split_join.join_df(y_pred)
                df = split_join.df

                # get distribution of traces cluster per variant
                stats = StatsClust()
                df_distrib = stats.get_distrib(df, df_ids)

                best_Vm = v_measure_score(y_true, y_pred)
                best_ARI = ARI
                best_perc = perc
                best_y_pred = y_pred.copy()
                best_df_distrib_2 = df_distrib.copy()
                # best_dict_gd = dict_gd.copy()
                best_dend = dend.copy()


            perc += step_perc


    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(best_ARI) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write('V-measure (Vm): ' + str(best_Vm) + '\n\n')
        f.write('best_clusters: ' + str(best_clusters) + '\n\n')
        f.write('best_ngram: ' + str(best_ngram) + '\n\n')
        f.write('best_min_percents: ' + str(best_min_percents) + '\n\n')
        f.write('best_max_percents: ' + str(best_max_percents) + '\n\n')
        f.write('best df_distrib fase 1: \n\n')
        f.write(best_df_distrib_1.to_string(header=True, index=True) + '\n\n')
        f.write('best df_distrib fase 2: \n\n')
        f.write(best_df_distrib_2.to_string(header=True, index=True) + '\n\n')
        f.write('dict_var: \n\n')
        f.write(str(best_dict_var) + '\n\n')
        f.write('dict_gd fase 2: ' + str(best_dict_gd2) + '\n\n')
        f.write('dict_gd: ' + str(best_dict_gd) + '\n\n')
        f.write('dend: ' + str(best_dend) + '\n\n')


    if 'AWS_bucket' in params_agglom:

        # Open a file: file
        file = open(exp_backlog, mode='r')
        
        # read all lines at once
        content = file.read()
        
        # close the file
        file.close()

        s3_handle.write_to_s3(bucket = bucket, 
                            filename = filename, 
                            file_content = content)
        
        print('wrote file to s3!')

        # shutdown ec2 instance

        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
        instance_id = response.text

        ec2 = boto3.resource('ec2')
        instance = ec2.Instance(instance_id)

        print('id: ' + str(instance))
        print('shutdown: ' + str(instance.terminate()))

    print('done!')

