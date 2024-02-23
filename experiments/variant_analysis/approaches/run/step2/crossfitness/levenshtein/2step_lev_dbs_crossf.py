from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import pandas as pd
import boto3
import time
import requests

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.DBScanClust import DBScanClust
from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
import utils.read_and_write.s3_handle as s3_handle



if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp5/exp5.xes'
    # exp_backlog = 'experiments/variant_analysis/exp3/exp3_leven_dbs.txt'
    exp_backlog = {}
    bucket = 'luiz-doutorado-projetos2'
    filename = 'experiments/variant_analysis/exp5/results/2step_leven_dbs_cross_fit_1.txt'
    content = ""

    params_agglom = {}
    params_agglom['custom_distance'] = 'alignment_between_logs'
    params_agglom['AWS_bucket'] = bucket
    params_agglom['AWS_filename'] = 'variant_analysis/exp5/progress_exp4_leven_dbs.txt'


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

    # get time needed
    start = time.time()

    # apply first-step clustering (DBScan)
    eps = 3
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

    # get dendrogram-list using cross-fitness
    agglomClust = CustomAgglomClust()   
    dend = agglomClust.agglom_fit(df, params_agglom)

    end = time.time()

    Z = agglomClust.gen_Z(dend)

    # hierarchy.dendrogram(Z)
    # plt.show(block=True)
    # plt.close()

    # get best number of clusters
    t = max(Z[:,2])

    min_perc = 0.3
    max_perc = 0.9
    step_perc = 0.025
    perc = min_perc

    best_ARI = -1
    best_Vm = -1
    best_perc = -1
    best_y_pred = []
    best_dict_gd = None
    

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

            # get variants by cluster
            dict_var_temp = stats.get_variants_by_cluster(traces, y_pred)
            print()

            # get variants ground truth by cluster
            best_dict_gd = stats.get_ground_truth_by_cluster(dict_var_temp, 
                                                             traces, 
                                                             y_true)

        perc += step_perc
       
    best_ARI = round(best_ARI, 4)
    best_Vm = round(best_Vm, 4)

    # write results to s3

    content = 'time: ' + str(round(end - start,4)) + '\n\n'
    content += 'Percent dendrogram cut: ' + str(best_perc) + '\n\n'
    content += 'ARI: ' + str(best_ARI) + '\n\n'
    content += 'V-measure: ' + str(best_Vm) + '\n\n'
    content += 'traces: ' + str(traces) + '\n\n'
    content += 'y_true: ' + str(y_true) + '\n\n'
    content += 'y_pred: ' + str(best_y_pred) + '\n\n'
    content += 'df_distrib: ' + df_distrib.to_string() + '\n\n'
    content += 'dict_gd: ' + str(dict_gd) + '\n\n'
    content += 'best_dict_gd: ' + str(best_dict_gd) + '\n\n'
    content += 'dend: ' + str(dend) + '\n\n'

    if 'AWS_bucket' in params_agglom:
        s3_handle.write_to_s3(bucket = bucket, 
                            filename = filename, 
                            file_content = content)
        
        print('wrote file to s3!')

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

    




