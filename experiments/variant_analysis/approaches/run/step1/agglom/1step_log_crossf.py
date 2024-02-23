from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy 

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.CustomAgglomClust \
    import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust
import utils.read_and_write.s3_handle as s3_handle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests
import boto3



if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp2/p1_v2v4v5.xes'

    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    exp_backlog = {}
    bucket = 'luiz-doutorado-projetos2'
    filename = 'experiments/variant_analysis/test/results/1step_log_crossf.txt'
    filename_dend = 'experiments/variant_analysis/test/results/dendrograms/' + \
                    '1step_log_crossf.png'
    content = ""

    params_agglom = {}
    params_agglom['custom_distance'] = 'alignment_between_logs'
    # params_agglom['DEBUG'] = True

    # params_agglom['AWS_bucket'] = bucket
    # params_agglom['AWS_filename'] = 'variant_analysis/exp5/' + \
    #     'progress_exp5_1step_log_crossf.txt'


    # convert to df
    df = convert_to_dataframe(log)

    # get traces
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    # get ground-truth
    utils = Utils()
    y_true = utils.get_ground_truth(ids_clus)

    # make each cluster individual to apply only agglomerative
    size = len(traces)
    cluster_labels = list(range(0,size))
    cluster_labels = np.array(cluster_labels)

    # print(list(set(cluster_labels)))
    # print(cluster_labels)

    # get df-log with cluster labels
    split_join.join_df(cluster_labels)
    df = split_join.df
    df_variants = split_join.join_df_uniques(cluster_labels)

    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(traces, cluster_labels)
    print()

    # get time needed
    start = time.time()

    # get dendrogram-list using cross-fitness
    agglomClust = CustomAgglomClust()
    dend = agglomClust.agglom_fit(df_variants, params_agglom)
    Z = agglomClust.gen_Z(dend)

    end = time.time()

    t = max(Z[:,2])

    min_perc = 0.3
    max_perc = 0.9
    step_perc = 0.025
    perc = min_perc

    best_ARI = -1
    best_Vm = -1
    best_perc = -1
    best_y_pred = []
    

    while perc <= max_perc:
        labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')

        # get performance by adjusted rand-score metric
        
        y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

        ARI = adjusted_rand_score(y_true, y_pred)
        Vm = v_measure_score(y_true, y_pred)

        if ARI > best_ARI:
            best_ARI = ARI
            best_Vm = Vm
            best_perc = perc
            best_y_pred = y_pred.copy()

        perc += step_perc
       
    best_ARI = round(best_ARI, 4)
    best_Vm = round(best_Vm, 4)
    best_perc = round(best_perc, 4)

    # write results to s3

    content = 'time: ' + str(round(end - start,4)) + '\n\n'
    content += 'Percent dendrogram cut: ' + str(best_perc) + '\n\n'
    content += 'ARI: ' + str(best_ARI) + '\n\n'
    content += 'V-measure: ' + str(best_Vm) + '\n\n'
    content += 'traces: ' + str(traces) + '\n\n'
    content += 'y_true: ' + str(y_true) + '\n\n'
    content += 'y_pred: ' + str(best_y_pred) + '\n\n'
    content += 'df_distrib: ' + df_distrib.to_string() + '\n\n'
    content += 'dend: ' + str(dend) + '\n\n'

    t = max(Z[:,2])

    hierarchy.dendrogram(Z, color_threshold=best_perc*t)
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    plt.savefig(filename_dend, dpi=400)
    # plt.show(block=True)
    plt.close()

    if 'AWS_bucket' in params_agglom:

        s3_handle.write_to_s3(bucket = bucket, 
                            filename = filename, 
                            file_content = content)

        print('done!')

        # shutdown ec2 instance

        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
        instance_id = response.text

        ec2 = boto3.resource('ec2')
        instance = ec2.Instance(instance_id)

        print('id: ' + str(instance))
        print('shutdown: ' + str(instance.terminate()))
    else:
        with open(filename, 'w') as f:
            f.write(content)


    print('done!')

