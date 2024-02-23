from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pm4py
import sys
import pandas as pd
from pathlib import Path
import models.lgbm as lgbm

from core.my_create_features import group_infrequent
from core.my_create_features import apply_one_hot_encoder
from core.my_create_features import rename_clus_col
import core.my_loader as my_loader

from clustering.actitrac.ActiTraCConnector import ActiTracConnector

def create_path_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__": 
    base_path = 'dataset/'
    log_path = 'dataset/tribunais_trabalho/TRT_mini2.xes'
    dataset_path = 'dataset/tribunais_trabalho/dataset_trt_model.csv'
    DEBUG = True
    use_cash = False

    if len(sys.argv) > 1:
        use_cash = eval(sys.argv[1])

    if len(sys.argv) > 2:
        number_cores = eval(sys.argv[2])
    else:
        number_cores = 4

    number_of_clusters_params = [5, 10, 20]
    target_fit_params = [0.9, 1]
    is_greedy_params = [True, False]
    dist_greed_params = [0.1, 0.25]
    heu_miner_threshold_params = [0.3, 0.5, 0.7]
    heu_miner_long_dist_params = [False, True]
    heu_miner_rel_best_thrs_params = [0.05, 0.025]

    total_runs = len(target_fit_params) * \
                 len(is_greedy_params) * \
                 len(dist_greed_params) * \
                 len(number_of_clusters_params)
    count_runs = 0

    best_mse = float('inf')
    best_r2 = float('-inf')
    best_params_and_result = {}

    # jar_path = 'temp/actitrac/actitrac.jar'
    jar_path = 'temp/actitrac/actitrac2.jar'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    random_seed = 3
    splits_kfold = 10
    number_cores = 4
    test_size = 0.2

    actitrac = ActiTracConnector(jar_path)
    
    params = {}
    params['boosting_type'] = 'dart'
    params['learning_rate'] = 0.1
    params['n_estimators'] = 100

    df = pd.read_csv(dataset_path, sep='\t')
    
    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    df_log = convert_to_dataframe(log)

    df = df[df['case:concept:name'].isin(df_log['case:concept:name'])]


    df = df[[
        'case:concept:name',
        'CASE:COURT:CODE',
        'CLA_EXECUCAO_FISCAL_1116',
        'MOV_CONCLUSAO_51',
        'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO',
        'MOVEMENTS_COUNT',
        'TOTAL_OFFICIAL',
        'MOV_DESARQUIVAMENTO_893',
        'TOTAL_MAGISTRATE',
        gt
    ]]

    cols_run = [c for c in df.columns if c != 'case:concept:name']

    # Run LGBM only with very basic dataset
    print('Run LGBM only with very basic dataset')

    params_and_result, _ = lgbm.train_and_run_lgbm(df[cols_run],
                                                   gt, 
                                                   [gt], 
                                                   params, 
                                                   random_seed, 
                                                   splits_kfold,
                                                   number_cores,
                                                   test_size)
    
    print(params_and_result)

    for number_of_clusters in number_of_clusters_params:
        for target_fit in target_fit_params:
            for is_greedy in is_greedy_params:
                for dist_greed in dist_greed_params:
                    for heu_miner_thres in heu_miner_threshold_params:
                        for heu_miner_long_dist in heu_miner_long_dist_params:
                            for heu_miner_rel_best_thrs in heu_miner_rel_best_thrs_params:
                                print('progress: ' + str(round(count_runs/total_runs,2)))

                                min_clus_size = round((1/(number_of_clusters)),5)
                                infreq = (1/number_of_clusters)/10

                                saving_path = 'temp/actitrac/trt/'
                                create_path_dir(saving_path)

                                if use_cash:
                                    df_clus = pd.read_csv('temp/df_clus_temp.csv', sep='\t')
                                else:   
                                    df_clus = actitrac.run(number_of_clusters,
                                                                    is_greedy,
                                                                    dist_greed,
                                                                    target_fit,
                                                                    min_clus_size,
                                                                    heu_miner_thres,
                                                                    heu_miner_long_dist,
                                                                    heu_miner_rel_best_thrs,
                                                                    0.1,
                                                                    log_path,
                                                                    saving_path,
                                                                    is_return_clusters=True)
                                    
                                    if DEBUG:
                                        df_check = df_clus.groupby('case:lawsuit:cluster_act').\
                                                    agg(count=('case:lawsuit:cluster_act','count'))
                                        df_check = df_check.sort_values('count', ascending=False)
                                        print(df_check)
                                        df_check.to_csv('temp/df_clus_temp.csv', sep='\t', index=False)

                                
                                df_clus = df_clus.rename(columns={'case:lawsuit:cluster_act':
                                                                'CASE:LAWSUIT:CLUSTER_ACT'})
                                df_clus = rename_clus_col(df_clus, 
                                                        'CASE:LAWSUIT:CLUSTER_ACT', 
                                                        'CLUS_ACT_')
                                df_clus = group_infrequent(df_clus, 
                                                            'CASE:LAWSUIT:CLUSTER_ACT', 
                                                            infreq, 
                                                            'CLUS_ACT_')
                                df_clus = apply_one_hot_encoder(df_clus, 
                                                                'CASE:LAWSUIT:CLUSTER_ACT')

                                # df['case:concept:name'] = df['case:concept:name'].astype(int)
                                # df_clus['case:concept:name'] = \
                                    #df_clus['case:concept:name'].astype(int)

                                df_run = df.merge(df_clus, on='case:concept:name', how='left')
                                df_run = df_run.drop(columns='case:concept:name')

                                # Make target to last column
                                cols = list(df_run.columns)
                                cols.remove(gt)
                                cols.append(gt)

                                df_run = df_run[cols]
                                

                                params_and_result, _ = lgbm.train_and_run_lgbm(df_run, 
                                                                            gt, 
                                                                            [gt], 
                                                                            params, 
                                                                            random_seed, 
                                                                            splits_kfold,
                                                                            number_cores,
                                                                            test_size)

                                mse = params_and_result['test_perf']['MSE']

                                if mse < best_mse:
                                    params_and_result['n_clusters'] = number_of_clusters
                                    params_and_result['target_fit'] = target_fit
                                    params_and_result['heu_miner_thres'] = heu_miner_thres
                                    params_and_result['heu_miner_long_dist'] = \
                                        heu_miner_long_dist
                                    params_and_result['heu_miner_rel_best_thrs'] = \
                                        heu_miner_rel_best_thrs

                                    best_mse = mse
                                    best_params_and_result = params_and_result

                                    print('Current best r2: ' + \
                                            str(params_and_result['test_perf']['R2']))

                                count_runs += 1

    print('### Best params and result:\n\n')
    print(best_params_and_result)