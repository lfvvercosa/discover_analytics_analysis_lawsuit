from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
import pandas as pd
from statistics import mean

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
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
    df_gram_norm = (df_gram - df_gram.min())/\
                (df_gram.max() - df_gram.min())
    df_gram_norm = df_gram_norm.round(4)

    # apply first-step clustering (DBScan) 
    min_samples = 1
    
    model = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean')
    cluster_labels = model.fit_predict(df_gram_norm)


    return adjusted_rand_score(y_true, cluster_labels), \
           df_gram_norm, \
           cluster_labels


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp5/exp5.xes'
    # exp_backlog = 'experiments/variant_analysis/exp3/exp3_leven_dbs.txt'
    filename = 'experiments/variant_analysis/exp5/1step_ngram_dbs.txt'
    content = ""

    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # optimize params

    n_grams = [1,2]
    minimum_percents = [0, 0.1, 0.2, 0.3]
    maximum_percents = [1, 0.9, 0.8, 0.7]
    epsilons = [0.3, 0.6, 1, 2]
    simus = [0,1,2]

    # n_grams = [1]
    # minimum_percents = [0]
    # maximum_percents = [1]
    # epsilons = [0.5]

    best_ARI = [-1,-1,-1]
    best_n = 1 
    best_min_perc = 0
    best_max_perc = 0.9
    best_eps = 0.3

    total = len(minimum_percents) * \
            len(maximum_percents) * \
            len(epsilons) * \
            len(n_grams) * \
            len(simus)
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
                    
                    ARI_list = []

                    for s in simus:
                        count += 1

                        if count % 10 == 0:
                            print('progress: ' + \
                                  str(round((count/total)*100,2)) + '%\n')

                        # convert to df
                        df = convert_to_dataframe(log)
                        
                        split_join = SplitJoinDF(df)
                        traces = split_join.split_df()
                        ids = split_join.ids
                        ids_clus = [l[0] for l in ids]
                        df_ids = pd.DataFrame.\
                                    from_dict({'case:concept:name':ids_clus})
                        df_clus = df.merge(df_ids, 
                                           on='case:concept:name',
                                           how='inner')

                        # get ground-truth
                        utils = Utils()
                        y_true = utils.get_ground_truth(ids_clus)

                        ARI,_,_ = run_dbscan_with_params(n, 
                                                        min_perc, 
                                                        max_perc, 
                                                        eps,
                                                        df_clus,
                                                        ids_clus)
                        
                        ARI_list.append(ARI)

                    if mean(ARI_list) > mean(best_ARI):

                        best_ARI = ARI_list.copy()
                        best_n = n
                        best_min_perc = min_perc
                        best_max_perc = max_perc
                        best_eps = eps

                        print('best ARI: ' + str(best_ARI))
                        print('best_n: ' + str(best_n))
                        print('best_min_perc: ' + str(best_min_perc))
                        print('best_max_perc: ' + str(best_max_perc))
                        print('best_eps: ' + str(best_eps))
    
    # run with best params
    ARI, df_gram_norm, cluster_labels = \
        run_dbscan_with_params(best_n, 
                               best_min_perc, 
                               best_max_perc, 
                               best_eps,
                               df_clus,
                               ids_clus)

    best_Vm = v_measure_score(y_true, cluster_labels)

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

    with open(filename, 'w') as f:
        f.write('Adjusted Rand Score (ARI) All: ' + str(best_ARI) + '\n\n')
        f.write('Adjusted Rand Score (ARI) Mean: ' + str(mean(best_ARI)) + '\n\n')
        f.write('V-measure (ARI): ' + str(best_Vm) + '\n\n')
        f.write('eps: ' + str(best_eps) + '\n\n')
        f.write('best_n: ' + str(best_n) + '\n\n')
        f.write('best_min_perc: ' + str(best_min_perc) + '\n\n')
        f.write('best_max_perc: ' + str(best_max_perc) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write(df_distrib.to_string(header=True, index=True) + '\n\n')
        f.write('dict_var: \n\n')
        f.write(str(dict_var) + '\n\n')

    print('done!')
