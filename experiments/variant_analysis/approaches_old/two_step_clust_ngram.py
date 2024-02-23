from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy 
import matplotlib.pyplot as plt

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust

import pandas as pd


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/v2v4v5v6v7v8.xes'
    log = log_p1v1 = xes_importer.apply(log_path)
    exp_backlog = 'experiments/variant_analysis/exp2/ngram2_dbs_conf2.txt'

    # convert to df
    df = convert_to_dataframe(log)

    # create n-gram
    n = 2
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    df_gram = create_n_gram(df_clus, 'case:concept:name', 'concept:name', n)
    df_gram.index = pd.Categorical(df_gram.index, ids_clus)
    df_gram = df_gram.sort_index()

    # remove infrequent cols
    min_perc = 0.1
    max_perc = 0.9

    preProcess = PreProcessClust()
    rem_cols = preProcess.selectColsByFreq(min_perc, max_perc, df_gram)
    df_gram = df_gram.drop(columns=rem_cols)

    # df_gram.to_csv('temp/df_gram.csv', sep='\t')

    # normalize n-gram
    df_gram_norm = (df_gram - df_gram.min())/\
                   (df_gram.max() - df_gram.min())
    df_gram_norm = df_gram_norm.round(4)

    # apply first-step clustering (DBScan) 
    eps = 1.25
    min_samples = 1
    
    model = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean')
    cluster_labels = model.fit_predict(df_gram_norm)

    # apply first-step clustering (KMeans)
    # model = KMeans(n_clusters=6)
    # model.fit(df_gram_norm)
    # cluster_labels = list(model.labels_)

    print(list(set(cluster_labels)))
    print(cluster_labels)

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

    # get centroids of each cluster
    # df_centroids = df_gram_norm.groupby('cluster_label').mean()

    # get dendrogram using centroids
    agglomClust = CustomAgglomClust()
    # agglomClust.gen_dendrogram(df_centroids)

    # get dendrogram-list using cross-fitness
    dend = agglomClust.agglom_fit(df)
    Z = agglomClust.gen_Z(dend)

    hierarchy.dendrogram(Z)
    plt.show(block=True)
    plt.close()

    # get best number of clusters
    t = max(Z[:,2])
    perc = 0.7
    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')

    # get performance by adjusted rand-score metric
    utils = Utils()
    y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)
    y_true = utils.get_ground_truth(ids_clus)

    ARI = adjusted_rand_score(y_true, y_pred)
    print()

    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(ARI) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write(df_distrib.to_string(header=True, index=True) + '\n\n')
        f.write('dict_var: \n\n')
        f.write(str(dict_var) + '\n\n')
        f.write('dend: \n\n')
        f.write(str(dend) + '\n\n')

    print('done!')

