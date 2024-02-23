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


def run_dbscan_with_params(n, min_perc, max_perc, eps, df_clus, ids_clus):
    # create n-gram
    df_gram = create_n_gram(df_clus, 'case:concept:name', 'concept:name', n)
    df_gram.index = pd.Categorical(df_gram.index, ids_clus)
    df_gram = df_gram.sort_index()

    preProcess = PreProcessClust()
    rem_cols = preProcess.selectColsByFreq(min_perc, max_perc, df_gram)
    df_gram = df_gram.drop(columns=rem_cols)

    # normalize n-gram
    df_gram_norm = df_gram.copy()

    for c in df_gram_norm.columns:
        if df_gram_norm[c].max() != df_gram_norm[c].min():
            df_gram_norm[c] = (df_gram_norm[c] - df_gram_norm[c].min())/\
                (df_gram_norm[c].max() - df_gram_norm[c].min())

    df_gram_norm = df_gram_norm.round(4)

    # apply first-step clustering (DBScan) 
    min_samples = 1
    
    model = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean')
    cluster_labels = model.fit_predict(df_gram_norm)


    return adjusted_rand_score(y_true, cluster_labels), df_gram_norm, cluster_labels


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp2/p1_v1v2.xes'
    # exp_backlog = 'experiments/variant_analysis/exp3/exp3_leven_dbs.txt'
    exp_backlog = {}
    bucket = 'luiz-doutorado-projetos2'
    filename = 'experiments/variant_analysis/exp5/2step_2gram_dbs_crossf_log.txt'
    content = ""

    params_agglom = {}
    params_agglom['custom_distance'] = 'alignment_between_logs'
    # params_agglom['AWS_bucket'] = bucket
    # params_agglom['AWS_filename'] = \
    #     'variant_analysis/exp5/progress_exp4_2gram_dbs_crossf.txt'


    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # optimize params

    n_grams = [1,2]
    minimum_percents = [0, 0.1, 0.2]
    maximum_percents = [1, 0.9, 0.8]
    epsilons = [0.5, 0.75, 1, 1.5, 2, 2.5]

    # n_grams = [1]
    # minimum_percents = [0]
    # maximum_percents = [0.9]
    # epsilons = [0.3]

    best_n = 1
    best_min_perc = 0
    best_max_perc = 0.9
    best_eps = 0.3
    best_ARI = -1
    best_ARI = -1

    total = len(minimum_percents) * len(maximum_percents) * \
            len(epsilons) * len(n_grams)
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
    y_true = utils.get_ground_truth(ids_clus)


    for n in n_grams:
        for min_perc in minimum_percents:
            for max_perc in maximum_percents:
                for eps in epsilons:

                    count += 1

                    if count % 10 == 0:
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

                    ARI,_,_ = run_dbscan_with_params(n, 
                                                     min_perc, 
                                                     max_perc, 
                                                     eps,
                                                     df_clus,
                                                     ids_clus)

                    if ARI > best_ARI:
                        print('best Vm: ' + str(ARI))
                        best_ARI = ARI
                        best_n = n
                        best_min_perc = min_perc
                        best_max_perc = max_perc
                        best_eps = eps

    # run with best params
    best_ARI, df_gram_norm, cluster_labels = \
        run_dbscan_with_params(best_n, 
                               best_min_perc, 
                               best_max_perc, 
                               best_eps,
                               df_clus,
                               ids_clus)

    # get df-log with cluster labels
    split_join.join_df(cluster_labels)
    df = split_join.df
    dict_labels = {'index':ids_clus, 'cluster_label':cluster_labels}
    df_labels = pd.DataFrame.from_dict(dict_labels)
    df_labels = df_labels.set_index('index')
    df_gram_norm = df_gram_norm.join(df_labels, how='left')

    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(traces, cluster_labels)
    print()

    # get variants ground truth by cluster
    dict_gd = stats.get_ground_truth_by_cluster(dict_var, traces, y_true)

    # get time needed
    start = time.time()    

    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get dendrogram-list using cross-fitness
    agglomClust = CustomAgglomClust()   
    dend = agglomClust.agglom_fit(df, params_agglom)

    end = time.time()

    Z = agglomClust.gen_Z(dend)

    hierarchy.dendrogram(Z)
    plt.show(block=True)
    # plt.close()

    # get best number of clusters
    t = max(Z[:,2])

    min_perc = 0.4
    max_perc = 0.9
    step_perc = 0.025
    perc = min_perc

    best_ARI = -1
    best_ARI = -1
    best_perc = -1
    best_y_pred = []
    

    while perc <= max_perc:
        labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')

        # get performance by adjusted rand-score metric
        
        y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

        ARI = adjusted_rand_score(y_true, y_pred)
        ARI = v_measure_score(y_true, y_pred)

        if ARI > best_ARI:
            best_ARI = ARI
            best_ARI = ARI
            best_perc = perc
            best_y_pred = y_pred.copy()

        perc += step_perc
       
    best_ARI = round(best_ARI, 4)
    best_ARI = round(best_ARI, 4)

    # write results to s3

    content = 'time: ' + str(round(end - start,4)) + '\n\n'
    content += 'Percent dendrogram cut: ' + str(best_perc) + '\n\n'
    content += 'ARI: ' + str(best_ARI) + '\n\n'
    content += 'V-measure: ' + str(best_ARI) + '\n\n'
    content += 'traces: ' + str(traces) + '\n\n'
    content += 'y_true: ' + str(y_true) + '\n\n'
    content += 'y_pred: ' + str(best_y_pred) + '\n\n'
    content += 'df_distrib: ' + df_distrib.to_string() + '\n\n'
    content += 'dict_gd: ' + str(dict_gd) + '\n\n'
    content += 'dend: ' + str(dend) + '\n\n'

    if 'AWS_bucket' in params_agglom:
        s3_handle.write_to_s3(bucket = bucket, 
                            filename = filename, 
                            file_content = content)
    else:
        with open(filename, 'w') as f:
            f.write(content)

    print('done!')

    # shutdown ec2 instance

    response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    instance_id = response.text

    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)

    print('id: ' + str(instance))
    print('shutdown: ' + str(instance.terminate()))
