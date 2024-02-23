from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.FindClustersHier import FindClustersHierarchy

import pandas as pd
from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score


if __name__ == '__main__':
    dend = [([0], [1], 0.9091), ([2], [3], 0.9091), ([5], [8], 0.9091), ([9], [10], 0.9091), ([12], [13], 0.9091), ([14], [15], 0.9091), ([16], [17], 0.9091), ([4], [11], 0.8571), ([18], [19], 0.8571), ([20], [21], 0.8571), ([0, 1], [2, 3], 0.8091), ([0, 1, 2, 3], [9, 10], 0.8091), ([12, 13], [14, 15], 0.8091), ([0, 1, 2, 3, 9, 10], [12, 13, 14, 15], 0.8091), ([5, 8], [6], 0.7842), ([4, 11], [16, 17], 0.7639), ([0, 1, 2, 3, 9, 10, 12, 13, 14, 15], [4, 11, 16, 17], 0.7446), ([18, 19], [20, 21], 0.6905), ([0, 1, 2, 3, 9, 10, 12, 13, 14, 15, 4, 11, 16, 17], [5, 8, 6], 0.5994), ([0, 1, 2, 3, 9, 10, 12, 13, 14, 15, 4, 11, 16, 17, 5, 8, 6], [18, 19, 20, 21], 0.4764), ([0, 1, 2, 3, 9, 10, 12, 13, 14, 15, 4, 11, 16, 17, 5, 8, 6, 18, 19, 20, 21], [7], 0.3447)]
    # y_true = [12, 12, 12, 12, 11, 12, 12, 11, 12, 12, 12, 12, 11, 12, 12, 11]
    # traces = ['ABCDE', 'ABCDEF', 'ABCED', 'ABCEDF', 'ABD', 'ABDCE', 'ABDCEF', 'ABDF', 'ACBDE', 'ACBDEF', 'ACBED', 'ACBEDF', 'ACE', 'ACEBD', 'ACEBDF', 'ACEF']
    
    # load event-log
    log_path = 'xes_files/test_variants/exp2/p1_v2v4v5.xes'
    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # convert to df
    df = convert_to_dataframe(log)

    # extract variants from log
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

    agglomClust = CustomAgglomClust()   
    utils = Utils()
    stats = StatsClust()
    
    perc = 0.475
    Z = agglomClust.gen_Z(dend)
    

    t = max(Z[:,2])

    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')
    labels = [l - 1 for l in labels]
    dict_var = stats.get_variants_by_cluster(traces, labels)
    y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)


    # get df-log with cluster labels
    split_join.join_df(labels)
    df = split_join.df

    df_distrib = stats.get_distrib(df, df_ids)


    print(df_distrib)
    print()

    n = 3
    # min_size = int(len(traces)/5)
    min_size = 3

    fit_markov = FindClustersHierarchy()
    fit_markov.get_n_clusters_hier(Z, n, min_size, log, traces)

    print()