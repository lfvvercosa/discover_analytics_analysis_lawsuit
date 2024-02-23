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

from clustering.agglomerative.Agglomerative import Agglomerative


def create_path_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sample_traces(df, p):
    df_ids = df.drop_duplicates('case:concept:name')
    df_ids = df_ids.sample(frac=p)

    
    return df[df['case:concept:name'].isin(df_ids['case:concept:name'])]


if __name__ == "__main__": 
    base_path = 'dataset/'
    log_path = 'dataset/tribunais_trabalho/TRT_mini.xes'
    dataset_path = 'dataset/tribunais_trabalho/dataset_trt_model.csv'
    DEBUG = False
    use_cash = False

    if len(sys.argv) > 1:
        use_cash = eval(sys.argv[1])

    if len(sys.argv) > 2:
        number_cores = eval(sys.argv[2])
    else:
        number_cores = 4

    # n_of_clusters_params = [5, 10, 15, 20, 25, 30, 35]
    n_of_clusters_params = [25]
    method_params = ['single','complete','average']
    max_size_par = [0.2]

    total_runs = len(method_params) * \
                 len(n_of_clusters_params)
    count_runs = 0

    best_mse = float('inf')
    best_r2 = float('-inf')
    best_params_and_result = {}

    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    random_seed = 3
    splits_kfold = 10
    number_cores = 4
    test_size = 0.2

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
    

    for n_clusters in n_of_clusters_params:
        for method in method_params:
                infreq = (1/n_clusters)/10
                print('progress: ' + str(round(count_runs/total_runs,2)))

                if use_cash:
                    df_clus = pd.read_csv('temp/df_clus_temp.csv', sep='\t')
                else:
                    agglom = Agglomerative(df_log)
                    df_clus = agglom.run(method, 
                                         'levenshtein', 
                                         n_clusters,
                                         max_size_par[0])
                        
                    if DEBUG:
                        if df_clus is not None:
                            df_clus.to_csv('temp/df_clus_temp.csv', sep='\t', index=False)

                if df_clus is None:
                    continue

                df_clus = df_clus.rename(columns={'cluster_label':
                                                    'CASE:LAWSUIT:CLUSTER_AGG'})
                df_clus = rename_clus_col(df_clus, 
                                         'CASE:LAWSUIT:CLUSTER_AGG', 
                                         'CLUS_AGG_')
                df_clus = group_infrequent(df_clus, 
                                            'CASE:LAWSUIT:CLUSTER_AGG', 
                                            infreq, 
                                            'CLUS_AGG_')
                df_clus = apply_one_hot_encoder(df_clus, 
                                                'CASE:LAWSUIT:CLUSTER_AGG')

                # df['case:concept:name'] = df['case:concept:name'].astype(int)
                # df_clus['case:concept:name'] = df_clus['case:concept:name'].astype(int)

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
                    params_and_result['n_clusters'] = n_clusters
                    params_and_result['method'] = method

                    best_mse = mse
                    best_params_and_result = params_and_result

                    print('Current best r2: ' + \
                            str(params_and_result['test_perf']['R2']))

                count_runs += 1

    print('### Best params and result:\n\n')
    print(best_params_and_result)