from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import pandas as pd

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.DBScanClust import DBScanClust
from experiments.variant_analysis.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust



if __name__ == '__main__':
    # load event-log
    # log_path = 'xes_files/test_variants/v2v4v5v6v7v8.xes'
    log_path = 'xes_files/test_variants/exp3/exp3.xes'
    exp_backlog = 'experiments/variant_analysis/exp3/exp3_leven_dbs.txt'
    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # convert to df
    df = convert_to_dataframe(log)

    # extract variants from log
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

    # get ground-truth
    utils = Utils()
    y_true = utils.get_ground_truth(ids_clus)

    # apply first-step clustering (DBScan)
    eps = 2
    min_samples = 1
    dbscan = DBScanClust(traces)
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

    # get variants ground truth by cluster
    dict_gd = stats.get_ground_truth_by_cluster(dict_var, traces, y_true)

    # test
    df_distrib.to_csv('temp/df_distrib.csv', sep='\t')
    
    with open('temp/dict_gd.txt', 'w') as f:
        f.write(str(dict_gd))  

    with open('temp/dict_var.txt', 'w') as f:
        f.write(str(dict_var)) 

    # get dendrogram-list using cross-fitness
    agglomClust = CustomAgglomClust()
    dend = agglomClust.agglom_fit(df)
    Z = agglomClust.gen_Z(dend)

    # hierarchy.dendrogram(Z)
    # plt.show(block=True)
    # plt.close()

    # get best number of clusters
    t = max(Z[:,2])
    perc = 0.7
    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')

    # get performance by adjusted rand-score metric
    y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)
    

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


