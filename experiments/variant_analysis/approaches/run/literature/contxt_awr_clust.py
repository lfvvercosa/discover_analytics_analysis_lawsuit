from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy 

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust
import utils.read_and_write.s3_handle as s3_handle
from experiments.variant_analysis.approaches.core.context_aware_clust.\
     ContextAgglomClust import ContextAgglomClustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests
import boto3


if __name__ == '__main__':
    log_path = 'xes_files/test_variants/exp4/exp4.xes'
    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # convert to df
    df = convert_to_dataframe(log)

    # get variants
    split_join = SplitJoinDF(df)
    variants = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    # get ground-truth
    utils = Utils()
    stats = StatsClust()
    y_true = utils.get_ground_truth(ids_clus)

    context_aware_clust = ContextAgglomClustering(log, variants)
    Z = context_aware_clust.cluster()

    best_percent = context_aware_clust.get_best_cutting_point(Z, 
                                                              y_true,
                                                              variants)

    t = max(Z[:,2])
    labels = hierarchy.fcluster(Z=Z, t=best_percent*t, criterion='distance')
    labels = [l-1 for l in labels]

    # get df-log with cluster labels
    split_join.join_df(labels)
    df = split_join.df

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(variants, labels)
    # y_pred = utils.get_agglom_labels_by_trace(variants, dict_var, labels)
    df_distrib = stats.get_distrib(df, df_ids)

    ARI = adjusted_rand_score(y_true, labels)
    Vm = v_measure_score(y_true, labels)

    hierarchy.dendrogram(Z, color_threshold=best_percent*t)
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    plt.show(block=True)
    plt.savefig('temp/dendro_lev_context.png', dpi=400)


    print()
