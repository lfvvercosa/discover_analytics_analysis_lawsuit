from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.AgglomClust import AglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.LevWeight import LevWeight


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp3/exp3.xes'
    exp_backlog = 'experiments/variant_analysis/exp3/results/1step_agglom_lev_par.txt'
    log = xes_importer.apply(log_path)
    content = ""

    weight_parallel = [1]
    weight_new_act = [1]
    weight_subst = [1]
    percentage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_ARI = 0
    best_wp = -1
    best_wn = -1
    best_ws = -1
    best_perc = -1

    total = len(weight_parallel) * len(weight_new_act) * len(weight_subst) * \
            len(percentage)
    count = 0

    # get time needed
    start = time.time()

    for wp in weight_parallel:
        for wn in weight_new_act:
            for ws in weight_subst:
                for perc in percentage:

                    count += 1

                    if count % 10 == 0:
                        print('progress: ' + str(round((count/total)*100,2)) + '%\n')

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
                    
                    # get performance by adjusted rand-score metric
                    utils = Utils()
                    y_pred = labels
                    y_true = utils.get_ground_truth(ids_clus)
                    # y_true = [0 if x < 20 else 1 for x in y_true]

                    ARI = adjusted_rand_score(y_true, y_pred)

                    if ARI > best_ARI:
                        best_ARI = ARI
                        best_wp = wp
                        best_wn = wn
                        best_ws = ws
                        best_perc = perc

                        print('ARI: ' + str(best_ARI))
                        print('best_wp: ' + str(best_wp))
                        print('best_wn: ' + str(best_wn))
                        print('best_ws: ' + str(best_ws))
                        print('best_perc: ' + str(best_perc) + '\n')

    end = time.time()

                    
    print('Best ARI: ' + str(best_ARI))
    print('best_wp: ' + str(best_wp))
    print('best_wn: ' + str(best_wn))
    print('best_ws: ' + str(best_ws))
    print('best_perc: ' + str(best_perc))

    # cluster using only agglommerative with levenshtein distance
    levWeight = LevWeight(traces,
                          log,
                          weight_parallel=best_wp,
                          weight_new_act=best_wn,
                          weight_substitute=best_ws,
                         )
    agglomClust = AglomClust(traces, levWeight.lev_metric_weight)
    Z = agglomClust.cluster('average')
    t = max(Z[:,2])

    hierarchy.dendrogram(Z,color_threshold=max(Z[:,2])*best_perc)
    # plt.show(block=True)
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    plt.savefig('temp/dendro_lev_par.png', dpi=400)
    plt.close()

    labels = hierarchy.fcluster(Z=Z, t=best_perc*t, criterion='distance')
    labels = [l-1 for l in labels]

    # get performance by adjusted rand-score metric
    utils = Utils()
    y_pred = labels
    y_true = utils.get_ground_truth(ids_clus)
    # y_true = [0 if x < 20 else 1 for x in y_true]

    best_ARI = adjusted_rand_score(y_true, y_pred)
    best_Vm = v_measure_score(y_true, y_pred)

    # get df-log with cluster labels
    split_join.join_df(labels)
    df = split_join.df

    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(traces, labels)

    # get variants ground truth by cluster
    dict_gd = stats.get_ground_truth_by_cluster(dict_var, traces, y_true)

    content = 'time: ' + str(round(end - start,4)) + '\n\n'
    content += 'Percent dendrogram cut: ' + str(best_perc) + '\n\n'
    content += 'ARI: ' + str(best_ARI) + '\n\n'
    content += 'V-measure: ' + str(best_Vm) + '\n\n'
    content += 'wp: ' + str(best_wp) + '\n\n'
    content += 'wn: ' + str(best_wn) + '\n\n'
    content += 'ws: ' + str(best_ws) + '\n\n'
    content += 'traces: ' + str(traces) + '\n\n'
    content += 'y_true: ' + str(y_true) + '\n\n'
    content += 'y_pred: ' + str(labels) + '\n\n'
    content += 'df_distrib: ' + df_distrib.to_string() + '\n\n'
    content += 'dict_gd: ' + str(dict_gd) + '\n\n'
    content += 'Z: ' + str(Z) + '\n\n'

    with open(exp_backlog, 'w') as f:
        f.write(content)

    print('done!')