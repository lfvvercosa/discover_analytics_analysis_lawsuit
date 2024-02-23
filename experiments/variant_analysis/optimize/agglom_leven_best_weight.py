from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.AgglomClust import AglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.LevWeight import LevWeight


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/v2v4v5v6v7v8.xes'
    exp_backlog = 'experiments/variant_analysis/exp2/leven_weight_agglom.txt'
    log = xes_importer.apply(log_path)

    weight_parallel = [0, 0.5, 1, 1.5, 2]
    weight_new_act = [1, 1.5, 2, 2.5]
    weight_subst = [0.5, 1, 1.5, 2, 2.5]
    percentage = [0.2, 0.3, 0.4, 0.5, 0.6]
    best_ARI = 0
    best_wp = -1
    best_wn = -1
    best_ws = -1
    best_perc = -1


    for wp in weight_parallel:
        for wn in weight_new_act:
            for ws in weight_subst:
                for perc in percentage:

                    # convert to df
                    df = convert_to_dataframe(log)

                    # extract variants from log
                    split_join = SplitJoinDF(df)
                    traces = split_join.split_df()
                    ids = split_join.ids
                    ids_clus = [l[0] for l in ids]
                    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

                    # cluster using only agglommerative with levenshtein distance
                    levWeight = LevWeight(traces,
                                          log,
                                          weight_parallel = wp,
                                          weight_new_act = wn,
                                          weight_substitute = ws,
                                         )
                    agglomClust = AglomClust(traces, levWeight.lev_metric_weight)
                    Z = agglomClust.cluster('average')
                    t = max(Z[:,2])

                    # hierarchy.dendrogram(Z, color_threshold=perc*t)
                    # plt.show(block=True)
                    # plt.close()

                    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')
                    labels = [l-1 for l in labels]

                    # get df-log with cluster labels
                    split_join.join_df(labels)
                    df = split_join.df

                    # get distribution of traces cluster per variant
                    # stats = StatsClust()
                    # df_distrib = stats.get_distrib(df, df_ids)

                    # get variants by cluster
                    # dict_var = stats.get_variants_by_cluster(traces, labels)
                    
                    # get performance by adjusted rand-score metric
                    utils = Utils()
                    y_pred = labels
                    y_true = utils.get_ground_truth(ids_clus)

                    ARI = adjusted_rand_score(y_true, y_pred)

                    if ARI > best_ARI:
                        best_ARI = ARI
                        best_wp = wp
                        best_wn = wn
                        best_ws = ws
                        best_perc = perc

                        print('ARI: ' + str(best_ARI))

                    
    print('Best ARI: ' + str(best_ARI))
    print('best_wp: ' + str(best_wp))
    print('best_wn: ' + str(best_wn))
    print('best_ws: ' + str(best_ws))
    print('best_perc: ' + str(best_perc))

    print('done!')