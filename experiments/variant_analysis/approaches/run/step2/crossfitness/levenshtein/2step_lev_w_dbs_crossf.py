from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd
import boto3
import requests
import time
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from scipy.cluster import hierarchy 
from statistics import mean

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.utils.Utils import Utils
from experiments.variant_analysis.approaches.core.DBScanLevWeight import DBScanLevWeight
from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.clustering.StatsClust import StatsClust
import utils.read_and_write.s3_handle as s3_handle


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp5/exp5.xes'
    exp_backlog = 'experiments/variant_analysis/exp5/results/2step_lev_w_dbs_crossf.txt'
    log = xes_importer.apply(log_path)
    
    use_AWS = True
    bucket = 'luiz-doutorado-projetos2'
    filename = 'variant_analysis/exp5/results/2step_lev_w_dbs_crossf.txt'

    params_agglom = {}
    params_agglom['custom_distance'] = 'alignment_between_logs'
    params_agglom['DEBUG'] = True
    params_agglom['AWS_bucket'] = bucket

    simus = 1

    weight_parallel = [2]
    weight_new_act = [2.5]
    weight_subst = [2.5]
    eps_list = [5]

    best_ARI = [-1,-1,-1]
    best_Vm = float('-inf')
    best_eps = -1
    best_wp = -1
    best_wn = -1
    best_ws = -1
    best_distrib = None
    best_distrib2 = None
    best_dict_var = None
    best_dict_gd = None

    count_runs = 0
    total = len(weight_parallel) * \
            len(weight_new_act) * \
            len(weight_subst) * \
            len(eps_list) * \
            simus
    
    start = time.time()
    
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

                        # get distribution of traces cluster per variant
                        stats = StatsClust()
                        df_distrib = stats.get_distrib(df, df_ids)

                        # get performance by adjusted rand-score metric
                        utils = Utils()
                        y_pred = labels
                        y_true = utils.get_ground_truth(ids_clus)
                        
                        # get variants by cluster
                        dict_var = stats.get_variants_by_cluster(traces, labels)

                        dict_gd = stats.get_ground_truth_by_cluster(dict_var, 
                                                                    traces, 
                                                                    y_true)


                        ARI = adjusted_rand_score(y_true, y_pred)
                        Vm = v_measure_score(y_true, y_pred)
                    
                        list_ARI.append(ARI)

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
                        best_distrib = df_distrib.copy()
                        best_dict_var = dict_var.copy()
                        best_dict_gd = dict_gd.copy()
                        best_wp = wp
                        best_wn = wn
                        best_ws = ws


    # get dendrogram-list using cross-fitness
    agglomClust = CustomAgglomClust()   
    dend = agglomClust.agglom_fit(df, params_agglom)

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
        labels = [l-1 for l in labels]

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

            # get variants ground truth by cluster
            best_dict_gd2 = stats.get_ground_truth_by_cluster(dict_var_temp, 
                                                             traces, 
                                                             y_true)
            
            df = convert_to_dataframe(log)
            split_join = SplitJoinDF(df)
            traces = split_join.split_df()
            ids = split_join.ids
            ids_clus = [l[0] for l in ids]
            df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

            # get df-log with cluster labels
            split_join.join_df(y_pred)
            df = split_join.df

            # get distribution of traces cluster per variant
            stats = StatsClust()
            best_distrib2 = stats.get_distrib(df, df_ids)

        perc += step_perc
       
    best_ARI = round(best_ARI, 4)
    best_Vm = round(best_Vm, 4)

    end = time.time()


    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(best_ARI) + '\n\n')
        f.write('V-measure (ARI): ' + str(best_Vm) + '\n\n')
        f.write('eps: ' + str(best_eps) + '\n\n')
        f.write('weight_parallel: ' + str(best_wp) + '\n\n')
        f.write('weight_new_act: ' + str(best_wn) + '\n\n')
        f.write('weight_subst: ' + str(best_ws) + '\n\n')
        f.write('time: ' + str(end - start) + '\n\n')
        f.write('df_distrib_1: \n\n')
        f.write(best_distrib.to_string(header=True, index=True) + '\n\n')
        f.write('df_distrib_2: \n\n')
        f.write(best_distrib2.to_string(header=True, index=True) + '\n\n')
        f.write('dict_gd2: \n\n')
        f.write(str(best_dict_gd2) + '\n\n')
        f.write('dict_gd: ' + str(best_dict_gd) + '\n\n')
        f.write('dend: ' + str(dend) + '\n\n')


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
