from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.DBScanLevWeight import DBScanLevWeight
from experiments.clustering.StatsClust import StatsClust

import pandas as pd


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/p1_v2v4v5v6v7.xes'
    log = xes_importer.apply(log_path)

    # convert to df
    df = convert_to_dataframe(log)

    # extract variants from log
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

    # apply first-step clustering (DBScan)
    eps = 1
    min_samples = 2
    weight_parallel = 1
    weight_new_act = 1
    weight_substitute = 1

    dbscan = DBScanLevWeight(traces,
                             log,
                             weight_parallel,
                             weight_new_act,
                             weight_substitute,
                             )
    labels = dbscan.cluster(eps, min_samples)

    print(list(set(labels)))

    # get df-log with cluster labels
    split_join.join_df(labels)
    df = split_join.df

    print()

    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(traces, labels)
    print()

    

