from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd
import boto3
import requests
import time
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from statistics import mean

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.utils.Utils import Utils
from experiments.variant_analysis.approaches.core.DBScanLevWeight import DBScanLevWeight
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity

import utils.read_and_write.s3_handle as s3_handle


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp5/exp5.xes'
    exp_backlog = 'experiments/variant_analysis/exp5/results/1step_lev_w_dbs.txt'
    log = xes_importer.apply(log_path)
    
    use_AWS = True
    bucket = 'luiz-doutorado-projetos2'
    filename = 'variant_analysis/exp5/results/1step_lev_w_dbs.txt'

    simus = 3

    # weight_parallel = [0, 0.5, 1, 1.5, 2, 2.5]
    # weight_new_act = [0, 0.5, 1, 1.5, 2, 2.5]
    # weight_subst = [0, 0.5, 1, 1.5, 2, 2.55]
    # eps_list = [1, 2, 3, 4, 5, 6, 7, 8]

    weight_parallel = [0]
    weight_new_act = [2.5]
    weight_subst = [2.55]
    eps_list = [6]

    best_ARI = [-1,-1,-1]
    best_Vm = float('-inf')
    best_eps = -1
    best_wp = -1
    best_wn = -1
    best_ws = -1
    best_distrib = None
    best_dict_var = None

    count_runs = 0
    total = len(weight_parallel) * \
            len(weight_new_act) * \
            len(weight_subst) * \
            len(eps_list) * \
            simus
    
    start = time.time()

    ## Calculate fitness and complexity
    fit_complex = FindFitnessComplexity()
    k_markov = 2
    
    for wp in weight_parallel:
        for wn in weight_new_act:
            for ws in weight_subst:
                for eps in eps_list:
                    count_simu = 0
                    list_ARI = []

                    while count_simu < simus:
                        count_simu += 1
                        count_runs += 1

                        if count_runs % 10 == 0:
                            print('progress: ' + \
                                str(round((count_runs/total)*100,2)) + '%\n')
                    
                        # convert to df
                        df = convert_to_dataframe(log)

                        # extract variants from log
                        split_join = SplitJoinDF(df)
                        traces = split_join.split_df()
                        ids = split_join.ids
                        ids_clus = [l[0] for l in ids]
                        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

                        # apply clustering (DBScan)
                        min_samples = 1
                        dbscan = DBScanLevWeight(traces,
                                                 log,
                                                 wp,
                                                 wn,
                                                 ws)
                        labels = dbscan.cluster(eps, min_samples)

                        # print(list(set(labels)))

                        # get df-log with cluster labels
                        split_join.join_df(labels)
                        df = split_join.df
                        df_variants = split_join.join_df_uniques(labels)

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
                        Vm = v_measure_score(y_true, y_pred)
                    
                        list_ARI.append(ARI)

                        clusters_number = 5
                        variants_number = len(df_variants['case:concept:name'].drop_duplicates())
                        min_size = int(variants_number/clusters_number)
                        k_markov = 2

                        logs = fit_complex.find_best_match_clusters(clusters_number,
                                                                    min_size,
                                                                    log,
                                                                    traces,
                                                                    labels)
                        fit, complex = fit_complex.get_metrics(logs, k_markov)


                    if mean(list_ARI) > mean(best_ARI):
                        print('best ARI: ' + str(ARI))
                        print('best eps: ', eps)
                        print('best wp: ', wp)
                        print('best wn: ', wn)
                        print('best ws: ', ws)
                        print()

                        best_ARI = list_ARI.copy()
                        best_Vm = Vm
                        best_eps = eps
                        best_distrib = df_distrib
                        best_dict_var = dict_var
                        best_wp = wp
                        best_wn = wn
                        best_ws = ws

    end = time.time()

    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(best_ARI) + '\n\n')
        f.write('V-measure (ARI): ' + str(best_Vm) + '\n\n')
        f.write('eps: ' + str(best_eps) + '\n\n')
        f.write('weight_parallel: ' + str(best_wp) + '\n\n')
        f.write('weight_new_act: ' + str(best_wn) + '\n\n')
        f.write('weight_subst: ' + str(best_ws) + '\n\n')
        f.write('time: ' + str(end - start) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write(best_distrib.to_string(header=True, index=True) + '\n\n')
        f.write('dict_var: \n\n')
        f.write(str(best_dict_var) + '\n\n')


    if use_AWS:
        with open(exp_backlog, 'r') as file:
            content = file.read()

        s3_handle.write_to_s3(bucket = bucket, 
                            filename = filename, 
                            file_content = content)

        print('done!')

        # shutdown ec2 instance

        response = requests.\
            get('http://169.254.169.254/latest/meta-data/instance-id')
        instance_id = response.text

        ec2 = boto3.resource('ec2')
        instance = ec2.Instance(instance_id)

        print('id: ' + str(instance))
        print('shutdown: ' + str(instance.terminate()))
        
        print('done!')
