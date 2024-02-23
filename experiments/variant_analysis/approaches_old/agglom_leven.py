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


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/v2v4v5v6v7v8.xes'
    exp_backlog = 'experiments/variant_analysis/exp2/leven_agglom.txt'
    log = xes_importer.apply(log_path)

    # convert to df
    df = convert_to_dataframe(log)

    # extract variants from log
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

    # cluster using only agglommerative with levenshtein distance
    agglomClust = AglomClust(traces)
    Z = agglomClust.cluster('weighted')
    t = max(Z[:,2])
    perc = 0.35

    hierarchy.dendrogram(Z, color_threshold=perc*t)
    plt.show(block=True)
    plt.close()

    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')
    labels = [l-1 for l in labels]

    # get df-log with cluster labels
    split_join.join_df(labels)
    df = split_join.df

    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(traces, labels)
    
    # get performance by adjusted rand-score metric
    utils = Utils()
    y_pred = labels
    y_true = utils.get_ground_truth(ids_clus)

    ARI = adjusted_rand_score(y_true, y_pred)

    # traces ids
    trace_id = {}

    for i in range(len(traces)):
        trace_id[i] = traces[i]

    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(ARI) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write(df_distrib.to_string(header=True, index=True) + '\n\n')
        f.write('dict_var: \n\n')
        f.write(str(dict_var) + '\n\n')
        f.write('dend: \n\n')
        f.write(str(Z) + '\n\n')
        f.write('traces: \n\n')
        f.write(str(trace_id) + '\n\n')

    print('done!')
