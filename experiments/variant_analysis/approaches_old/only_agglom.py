from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy 
import matplotlib.pyplot as plt

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust

import pandas as pd
import numpy as np


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp2/p1_v1v2.xes'
    log = xes_importer.apply(log_path)
    exp_backlog = 'experiments/variant_analysis/exp2/only_agglom_cross_fit.txt'
    params_agglom = {}

    # convert to df
    df = convert_to_dataframe(log)

    # get traces
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    
    # make each cluster individual to apply only agglomerative
    size = len(traces)
    cluster_labels = list(range(0,size))
    cluster_labels = np.array(cluster_labels)

    print(list(set(cluster_labels)))
    print(cluster_labels)

    # get df-log with cluster labels
    split_join.join_df(cluster_labels)
    df = split_join.df
    dict_labels = {'index':ids_clus, 'cluster_label':cluster_labels}
    
    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(traces, cluster_labels)
    print()

    # get dendrogram-list using cross-fitness
    agglomClust = CustomAgglomClust()
    dend = agglomClust.agglom_fit(df, params_agglom)
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

