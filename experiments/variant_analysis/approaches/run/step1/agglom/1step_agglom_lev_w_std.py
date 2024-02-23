import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import pandas as pd
import boto3
import time
import requests

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.variant_analysis.approaches.core.metrics.MarkovMeasures import MarkovMeasures
from experiments.variant_analysis.approaches.core.LevWeight import LevWeight
from experiments.variant_analysis.approaches.core.AgglomClust import AglomClust

import utils.read_and_write.s3_handle as s3_handle



def calc_overall_fitness_perc(df_labels,
                             cluster_labels, 
                             min_pts, 
                             weight_clus, 
                             s,
                             k):

    count = Counter(cluster_labels)
    bigger_clusters = [c for c in count.most_common() if c[1] >= min_pts]
    total = sum([c[1] for c in bigger_clusters])
    weight_fit = 0

    for c in bigger_clusters:
        df_log = df_labels[df_labels['cluster_label'] == c[0]]
        log = pm4py.convert_to_event_log(df_log)
        markov_fitness = MarkovMeasures(log, k)
        fit = markov_fitness.get_fitness_gaussian(n=s)
        weight_fit += fit * c[1]

    weight_fit /= total
    weight_fit -= weight_clus * len(bigger_clusters)


    return weight_fit


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp4/exp4.xes'
    exp_backlog = {}
    bucket = 'luiz-doutorado-projetos2'
    local_file = 'experiments/variant_analysis/exp4/results/1step_agglom_lev_w_std.txt'
    filename = 'variant_analysis/exp4/results/1step_agglom_lev_w_std.txt'
    content = ""

    params_agglom = {}
    params_agglom['AWS_bucket'] = bucket
    params_agglom['AWS_filename'] = \
        'variant_analysis/exp4/results/1step_agglom_lev_w_std.txt'


    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # convert to df
    df = convert_to_dataframe(log)
    
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    # get ground-truth
    utils = Utils()
    y_true = utils.get_ground_truth(ids_clus)

    # optimize params

    minimum_pts = [1, 3, 5, 7]
    weight_cluster_number = [0, 0.01, 0.001]
    std = [0.2, 0.3, 0.4]
    k_markov = [1, 2, 3]

    # minimum_pts = [1]
    # weight_cluster_number = [0]
    # mean_perc = [0.2]
    # k_markov = [1]

    weight_parallel = [0, 0.5, 1, 1.5, 2, 2.5]
    weight_new_act = [1, 1.5, 2, 2.5, 3]
    weight_subst = [0.5, 1, 1.5, 2, 2.5]
    percentage = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # weight_parallel = [0]
    # weight_new_act = [1]
    # weight_subst = [0.5]
    # percentage = [0.2]


    total = len(minimum_pts) * len(weight_cluster_number) * \
            len(std) * len(k_markov) * len(weight_parallel) * \
            len(weight_new_act) * len(weight_subst) * len(percentage)
    

    best_min_pts = -1
    best_weight_cluster_number = -1
    best_standard_deviations = -1
    best_k_markov = -1
    ALL_best_Fit = -1
    ALL_ARI = -1

    count = 0
    firstTime = True

    for min_pts in minimum_pts:
        for weight_clus in weight_cluster_number:
            for s in std:
                for k in k_markov:
                   
                    best_wp = -1
                    best_wn = -1
                    best_ws = -1
                    best_perc = -1
                    best_Fit = -1
                    best_ARI = -1

                    for wp in weight_parallel:
                        for wn in weight_new_act:
                            for ws in weight_subst:
                                for perc in percentage:

                                    if count % 10 == 0:
                                        progress = str(round((count/total)*100,2)) + '%\n'

                                        s3_handle.write_to_s3(params_agglom['AWS_bucket'],
                                          params_agglom['AWS_filename'],
                                          progress
                                         )
                                        print('progress: ' + progress)

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

                                    labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')
                                    labels = [l-1 for l in labels]

                                    # get df with cluster labels
                                    df_labels = pd.DataFrame.from_dict({'case:concept:name':ids_clus,
                                                                        'cluster_label':labels})
                                    df_labels = df_clus.merge(df_labels, on='case:concept:name', how='inner')
                                    Fit = calc_overall_fitness_perc( 
                                                               df_labels,
                                                               labels,
                                                               min_pts,
                                                               weight_clus,
                                                               s,
                                                               k
                                                             )

                                    if Fit > best_Fit:
                                        best_Fit = Fit
                                        best_wp = wp
                                        best_wn = wn
                                        best_ws = ws
                                        best_perc = perc
                                        best_ARI = adjusted_rand_score(y_true, labels)

                                    count += 1

                    if firstTime:
                        openMode = 'w'
                        firstTime = False
                    else:
                        openMode = 'a+'

                    with open(local_file, openMode) as f:
                        f.write('min_pts: ' + str(min_pts) + '\n')
                        f.write('weight_clus: ' + str(weight_clus) + '\n')
                        f.write('std: ' + str(perc) + '\n')
                        f.write('k: ' + str(k) + '\n')
                        f.write('best_wp: ' + str(best_wp) + '\n')
                        f.write('best_wn: ' + str(best_wn) + '\n')
                        f.write('best_ws: ' + str(best_ws) + '\n')
                        f.write('best_perc: ' + str(best_perc) + '\n')

                        f.write('Fit Markov Std: ' + str(best_Fit) + '\n')
                        f.write('ARI: ' + str(best_ARI) + '\n\n')

                    if best_Fit > ALL_best_Fit:
                        ALL_best_Fit = best_Fit
                        best_min_pts = min_pts
                        best_weight_cluster_number = weight_clus
                        best_standard_deviations = perc
                        best_k_markov = k
                        ALL_ARI = best_ARI

                        print('all_best_fit: ' + str(round(ALL_best_Fit,2)))


    with open(local_file, 'a+') as f:
        f.write('#### Best Config #### \n')
        f.write('min_pts: ' + str(best_min_pts) + '\n')
        f.write('weight_clus: ' + str(best_weight_cluster_number) + '\n')
        f.write('std: ' + str(best_standard_deviations) + '\n')
        f.write('k: ' + str(best_k_markov) + '\n')
        f.write('Fit Markov Std: ' + str(ALL_best_Fit) + '\n')
        f.write('ARI: ' + str(ALL_ARI) + '\n')


    with open(local_file, 'r') as file:
        content = file.read()

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

    