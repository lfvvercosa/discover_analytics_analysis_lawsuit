from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from scipy.cluster import hierarchy 
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import pandas as pd
import boto3
import time
import requests
import numpy as np
from scipy.spatial.distance import hamming
from scipy.spatial.distance import cosine
from sklearn import preprocessing

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.DBScanClust \
    import DBScanClust
from experiments.variant_analysis.approaches.core.CustomAgglomClust \
    import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust

import utils.read_and_write.s3_handle as s3_handle


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

                # df_gram.to_csv('temp/df_gram.csv', sep='\t')

                # normalize n-gram
                # df_gram_norm = df_gram.copy()

                # for c in df_gram_norm.columns:
                #     if df_gram_norm[c].max() != df_gram_norm[c].min():
                #         df_gram_norm[c] = (df_gram_norm[c] - df_gram_norm[c].min())/\
                #             (df_gram_norm[c].max() - df_gram_norm[c].min())

                # df_gram_norm = df_gram_norm.round(4)

                # normalize n-gram

                # print(df_gram)
                # print(df_gram.dtypes)

                x = df_gram.values #returns a numpy array
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(x)
                df_gram_norm = pd.DataFrame(x_scaled)

                df_gram_norm = df_gram_norm.round(4)

                cashed_dfs[n][min_perc][max_perc] = df_gram_norm.copy()


    return cashed_dfs


def run_dbscan_with_params(n, min_perc, max_perc, eps, df_clus, ids_clus, metr):
    df_gram_norm = cashed_df_grams[n][min_perc][max_perc]

    # apply first-step clustering (DBScan) 
    min_samples = 1
    
    model = DBSCAN(eps = eps, min_samples = min_samples, metric=metr)
    cluster_labels = model.fit_predict(df_gram_norm)


    return adjusted_rand_score(y_true, cluster_labels), df_gram_norm, cluster_labels


def get_eps_by_metric(metric, p1, p2):
    if metric == 'hamming':
        return hamming(p1, p2)
    elif metric == 'cosine':
        return cosine(p1, p2)
    elif metric == 'euclidean':
        p1 = np.array(p1)
        p2 = np.array(p2)

        return np.linalg.norm(p1 - p2)


def find_eps(n_rows, p, metric, n):
    changes = max(1, int(n_rows * p))
    ones = n_rows - changes
    zeroes = n_rows - ones

    p1 = [1] * n_rows
    p2 = [1] * ones + [0] * zeroes
    
    eps = get_eps_by_metric(metric, p1, p2)
    eps_list = []

    for i in range(n):
        eps_list.append(round(eps + (i*0.15)*eps,4))

    
    return eps_list


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp2/p1_v2v4v5.xes'
    # exp_backlog = 'experiments/variant_analysis/exp3/exp3_leven_dbs.txt'
    exp_backlog = {}
    bucket = 'luiz-doutorado-projetos2'
    filename = 'experiments/variant_analysis/test/results/2step_2gram_dbs_crossf_log_1.txt'
    content = ""

    params_agglom = {}
    params_agglom['custom_distance'] = 'alignment_between_logs'
    # params_agglom['AWS_bucket'] = bucket
    # params_agglom['AWS_filename'] = \
    #     'variant_analysis/test/progress_exp5_2gram_dbs_crossf_1.txt'


    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # optimize params

    # n_grams = [1,2]
    # minimum_percents = [0, 0.05, 0.1]
    # maximum_percents = [1, 0.95, 0.9]
    # epsilons = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5]

    n_grams = [2]
    minimum_percents = [0]
    maximum_percents = [1]
    # epsilons = [0.01, 0.1, 0.25]
    # epsilons = [0.025]
    n_eps = 1
    p_eps = 0.02

    metrics = ['euclidean']
    # metrics = ['euclidean', 'hamming', 'cosine']

    best_n = 1
    best_min_perc = 0
    best_max_perc = 0.9
    best_eps = 0.3
    best_Vm = -1
    best_ARI = -1
    best_perc = -1
    best_y_pred = []
    best_df_distrib = None
    best_dict_gd = None
    best_dend = None

    total = len(minimum_percents) * len(maximum_percents) * \
            len(n_grams) * n_eps
    count = 0

    # convert to df
    df = convert_to_dataframe(log)

    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    # get ground-truth
    utils = Utils()
    stats = StatsClust()
    y_true = utils.get_ground_truth(ids_clus)
    start = time.time()

    cashed_df_grams = cashe_df_grams(log, 
                                     n_grams, 
                                     minimum_percents, 
                                     maximum_percents)


    for n in n_grams:
        for min_perc in minimum_percents:
            for max_perc in maximum_percents:
                for metr in metrics:
                    n_rows = len(cashed_df_grams[n][min_perc][max_perc].columns)
                    epsilons = find_eps(n_rows,p_eps,metr,n_eps)
                    # epsilons = [2.8169]

                    for eps in epsilons:

                        print('ngram: ',n)
                        print('min_perc: ',min_perc)
                        print('max_perc: ',max_perc)
                        print('eps: ',eps)
                        print('metric: ',metr)

                        count += 1

                        print('progress: ' + str(round((count/total)*100,2)) + '%\n')

                        # convert to df
                        df = convert_to_dataframe(log)
                        
                        split_join = SplitJoinDF(df)
                        traces = split_join.split_df()
                        ids = split_join.ids
                        ids_clus = [l[0] for l in ids]
                        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                        df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                        # get ground-truth
                        utils = Utils()
                        y_true = utils.get_ground_truth(ids_clus)

                        ARI,_,cluster_labels = run_dbscan_with_params(n, 
                                                                    min_perc, 
                                                                    max_perc, 
                                                                    eps,
                                                                    df_clus,
                                                                    ids_clus,
                                                                    metr)

                        # get df-log with cluster labels
                        split_join.join_df(cluster_labels)
                        df = split_join.df
                        dict_labels = {'index':ids_clus, 'cluster_label':cluster_labels}
                        df_labels = pd.DataFrame.from_dict(dict_labels)
                        df_labels = df_labels.set_index('index')

                        # get distribution of traces cluster per variant
                        df_distrib = stats.get_distrib(df, df_ids)

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

                                # get performance by adjusted rand-score metric
                                
                                y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

                                ARI = adjusted_rand_score(y_true, y_pred)

                                if ARI > best_ARI:

                                    # get variants ground truth by cluster
                                    dict_gd = stats.\
                                        get_ground_truth_by_cluster(dict_var, traces, y_true)
                                    
                                    best_Vm = v_measure_score(y_true, y_pred)
                                    best_ARI = ARI
                                    best_perc = perc
                                    best_y_pred = y_pred.copy()
                                    best_min_perc = min_perc
                                    best_max_perc = max_perc
                                    best_eps = eps
                                    best_n = n
                                    best_df_distrib = df_distrib.copy()
                                    best_dict_gd = dict_gd.copy()
                                    best_dend = dend.copy()


                                perc += step_perc

    end = time.time()
    
    # write results to s3

    content = 'time: ' + str(round(end - start,4)) + '\n\n'
    content += 'Percent dendrogram cut: ' + str(best_perc) + '\n\n'
    content += 'ARI: ' + str(best_ARI) + '\n\n'
    content += 'V-measure: ' + str(best_ARI) + '\n\n'
    content += 'Best n: ' + str(best_n) + '\n\n'
    content += 'Min-perc: ' + str(best_min_perc) + '\n\n'
    content += 'Max-perc: ' + str(best_max_perc) + '\n\n'
    content += 'eps: ' + str(best_eps) + '\n\n'
    content += 'traces: ' + str(traces) + '\n\n'
    content += 'y_true: ' + str(y_true) + '\n\n'
    content += 'y_pred: ' + str(best_y_pred) + '\n\n'
    content += 'df_distrib: ' + df_distrib.to_string() + '\n\n'
    content += 'dict_gd: ' + str(dict_gd) + '\n\n'
    content += 'dend: ' + str(best_dend) + '\n\n'

    if 'AWS_bucket' in params_agglom:
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
    
    else:
        with open(filename, 'w') as f:
            f.write(content)

    print('done!')

    
