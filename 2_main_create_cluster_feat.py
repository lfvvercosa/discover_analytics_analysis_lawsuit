from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe
import pm4py
import time

import statistics as stats
import sys
import os
import pandas as pd
import numpy as np
import models.lgbm as lgbm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import core.my_loader as my_loader
from core import my_create_features
from clustering.boa.Kmeans import Kmeans
from clustering.agglomerative.Agglomerative import Agglomerative
from clustering.actitrac.ActiTraCConnector import ActiTracConnector
from core.my_create_features import group_infrequent
from core.my_create_features import apply_one_hot_encoder
from core.my_create_features import rename_clus_col
from core.my_create_features import map_infrequent
from core.my_utils import equalize_dfs
from models.DatasetSplit import DatasetSplit


def process_dfs_clus(df_clus, df_clus_valid, col, thres):
    map_values = map_infrequent(df_clus, col, thres)
    df_clus[col] = df_clus[col].map(map_values)
    df_clus_valid[col] = df_clus_valid[col].map(map_values)
    df_clus = apply_one_hot_encoder(df_clus, col)
    df_clus_valid = apply_one_hot_encoder(df_clus_valid, col)

    df_clus_valid = equalize_dfs(df_clus, df_clus_valid)


    return df_clus, df_clus_valid


def process_dfs_clus_target(df_clus, df_clus_valid, df_train, col, thres, gt):
    map_values = map_infrequent(df_clus, col, thres)
    df_clus[col] = df_clus[col].map(map_values)
    df_clus_valid[col] = df_clus_valid[col].map(map_values)

    df_work = df_clus.merge(df_train[['case:concept:name',gt]])
    df_work = df_work.groupby(col).agg(mean=(gt,'mean'),
                                             median=(gt,'median'),
                                             std=(gt,'std'),
                                             count=(gt,'count'),
                                            )
    df_work = df_work.sort_values('median', ascending=True)
    df_work['new_order'] = np.arange(len(df_work.index))
    df_work = df_work[['new_order']]
    map_values = df_work.to_dict()['new_order']

    df_clus[col] = df_clus[col].replace(map_values)
    df_clus_valid[col] = df_clus_valid[col].replace(map_values)

    # Normalize min-max
    max_val = df_clus[col].max()
    min_val = df_clus[col].min()

    df_clus[col] = (df_clus[col] - min_val) / (max_val - min_val) 
    df_clus_valid[col] = (df_clus_valid[col] - min_val) / \
                         (max_val - min_val) 

    
    return df_clus, df_clus_valid


def process_dfs_clus_ohe(df_clus, df_clus_valid):
    df_clus = apply_one_hot_encoder(df_clus, 'cluster_label')
    df_clus_valid = apply_one_hot_encoder(df_clus_valid, 'cluster_label')
    df_clus_valid = equalize_dfs(df_clus, df_clus_valid)


    return df_clus, df_clus_valid


def equalize_dfs(df_ref, df):
        only_in_df_ref = [c for c in df_ref.columns if c not in df.columns]
        only_in_df = [c for c in df.columns if c not in df_ref.columns]

        if len(only_in_df) > 0:
            raise Exception('invalid cluster in df_test')

        for c in only_in_df_ref:
            df[c] = 0

        
        return df

def obtain_cluster_feat(algorithm, 
                        config, 
                        log_train, 
                        log_valid, 
                        idx, 
                        df_model_train,
                        is_ohe):
    if algorithm == 'kmeans':
        technique = Kmeans(log_train,
                           config['n_clusters'],
                           config['n_gram'],
                           config['min_max_perc'],
                           config['kms_algorithm'])
        par = {config['n_gram']:[config['min_max_perc']]}
        cashed_df = technique.cashe_df_grams(log_train, par)
        
        df_clus = technique.run(
                                cashed_df,
                            )
        df_ref = cashed_df[config['n_gram']][config['min_max_perc']]
        df_clus_valid = technique.validate(log_valid, df_ref)
    
    elif algorithm == 'agglom':
        df_log = pm4py.convert_to_dataframe(log_train)

        technique = Agglomerative(df_log,
                                  config['method'], 
                                  config['metric'],
                                  config['ins_del_weight'],
                                  config['trans_weight'],
                                  config['subst_weight'],
                                  is_map_act=True
                                 )
        params_config = (idx, 
                         config['metric'], 
                         config['ins_del_weight'], 
                         config['trans_weight'],
                         config['subst_weight'],
                        )
        
        if not params_config in cashe:
            dist_matrix = np.copy(technique.create_dist_matrix())
            cashe[params_config] = technique.save_cash(config['cash_path'],
                                                       params_config,
                                                       dist_matrix)
            print('### cashing...')
        else:
            dist_matrix = technique.retrieve_cash(config['cash_path'],
                                                  params_config)
            print('### retrieving cash...')
        
        # dist_matrix = cashe[params_config]
        df_clus = technique.run(
                                config['n_clusters'],
                                config['max_size_perc'],
                                dist_matrix
                               )
        df_clus_valid = technique.validate(log_valid)
    
    elif algorithm == 'actitrac':
        technique = ActiTracConnector(
                        config['is_greedy'],
                        config['dist_greed'],
                        config['target_fit'],
                        config['min_clus_size'],
                        config['heu_miner_threshold'],
                        config['heu_miner_long_dist'],
                        config['heu_miner_rel_best_thrs'],
                        config['heu_miner_and_thrs'],
                        config['include_external'],
                    )
        
        technique.remove_file_if_exists(config['log_path_train'])
        technique.remove_file_if_exists(config['log_path_valid'])

        technique.save_log(log_train, config['log_path_train'])
        technique.save_log(log_valid, config['log_path_valid'])

        df_log_train = pm4py.convert_to_dataframe(log_train)
        traces_train = get_readable_traces(df_log_train)

        df_log_valid = pm4py.convert_to_dataframe(log_valid)
        traces_valid = get_readable_traces(df_log_valid)

        df_clus = technique.run(
                                config['n_clusters'],
                                config['log_path_train'],
                                config['saving_path_train'],
                                True
                               )

        print('### saving df_clus...')
        df_clus.to_csv('temp/df_clus.csv', sep='\t')

        df_clus_valid = technique.validate(
                                    config['log_path_valid'],
                                    config['saving_path_valid'],
                                    config['saving_path_train']
                                        )

    if df_clus is None:
        return pd.DataFrame([]),pd.DataFrame([])

    infreq = (1/config['n_clusters'])/2
    
    if not is_ohe:
        df_clus, df_clus_valid = process_dfs_clus_target(
                                        df_clus, 
                                        df_clus_valid,
                                        df_model_train,
                                        'cluster_label',
                                        infreq,
                                        gt
                                )
    else:
        df_clus, df_clus_valid = process_dfs_clus_ohe(
                                    df_clus, 
                                    df_clus_valid,
                                )

    # df_clus, df_clus_valid = process_dfs_clus(df_clus, 
    #                                           df_clus_valid, 
    #                                           'cluster_label',
    #                                           infreq,
    #                                          )


    return df_clus, df_clus_valid


def get_model_predic(df_model_train, df_model_test, gt, params):
    X_train, y_train = get_feat_gt(df_model_train, 
                                             [gt], 
                                              gt)
    X_test, y_test = get_feat_gt(df_model_test, 
                                        [gt], 
                                        gt)
    y_pred = lgbm.run_lgbm(X_train, 
                           y_train, 
                           X_test, 
                           params)
    

    return y_test, y_pred


def get_config_params(algorithm, params):
    config_params = []

    if algorithm == 'kmeans':
        for n_clusters in params['n_clusters_params']:
            for kms_algorithm in params['kms_algorithm']:
                for n_gram in params['n_gram_params']:
                    for min_max_perc in params['n_gram_params'][n_gram]:
                        config_params.append({
                            'n_clusters':n_clusters,
                            'n_gram':n_gram,
                            'min_max_perc':min_max_perc,
                            'kms_algorithm':kms_algorithm,
                        })
    
    if algorithm == 'agglom':
        for n_clusters in params['n_clusters_params']:
            for method in params['method']:
                for max_perc in params['max_perc']:
                    for metric in params['metric']:
                        if metric == 'levenshtein':
                            config_params.append({
                                        'n_clusters':n_clusters,
                                        'method':method,
                                        'max_size_perc':max_perc,
                                        'metric':metric,
                                        'ins_del_weight':1,
                                        'trans_weight':1,
                                        'subst_weight':1,
                                        'cash_path':params['cash_path'],
                                    })
                            continue
                        else:
                            for ins_del_weight in params['ins_del_weight']:
                                for trans_weight in params['trans_weight']:
                                    for subst_weight in params['subst_weight']:
                                        
                                        if ins_del_weight == 2 and \
                                           trans_weight == 2 and \
                                           subst_weight == 8:
                                            continue
                                                

                                        config_params.append({
                                            'n_clusters':n_clusters,
                                            'method':method,
                                            'max_size_perc':max_perc,
                                            'metric':metric,
                                            'ins_del_weight':ins_del_weight,
                                            'trans_weight':trans_weight,
                                            'subst_weight':subst_weight,
                                            'cash_path':params['cash_path'],
                                        })

    if algorithm == 'actitrac':
        for n_clusters in params['n_clusters_params']:
            for target_fit in params['target_fit']:
                for is_greedy in params['is_greedy']:
                    for dist_greed in params['dist_greed']:
                        for min_size_factor in params['min_size_factor']:
                            for heu_miner_conf in params['heu_miner_config']:
                                for include_external in params['include_external']:

                                    min_clus_size = round((1/(min_size_factor * n_clusters)),5)
                                    hm_threshold = heu_miner_conf['heu_miner_threshold']
                                    hm_long_dist = heu_miner_conf['heu_miner_long_dist']
                                    hm_rel_best_thrs = heu_miner_conf['heu_miner_rel_best_thrs']
                                    hm_and_thrs = heu_miner_conf['heu_miner_and_thrs']

                                    config_params.append({
                                                'n_clusters':n_clusters,
                                                'target_fit':target_fit,
                                                'is_greedy':is_greedy,
                                                'dist_greed':dist_greed,
                                                'min_clus_size':min_clus_size,
                                                'heu_miner_threshold':hm_threshold,
                                                'heu_miner_long_dist':hm_long_dist,
                                                'heu_miner_rel_best_thrs':hm_rel_best_thrs,
                                                'heu_miner_and_thrs':hm_and_thrs,
                                                'saving_path_train':params['saving_path_train'],
                                                'saving_path_valid':params['saving_path_valid'],
                                                'log_path_train':params['log_path_train'],
                                                'log_path_valid':params['log_path_valid'],
                                                'include_external':include_external,
                                            })
                                


                            # for hm_threshold in params['heu_miner_threshold']:
                            #     for hm_long_dist in params['heu_miner_long_dist']:
                            #         for hm_rel_best_thrs in params['heu_miner_rel_best_thrs']:
                            #             for hm_and_thrs in params['heu_miner_and_thrs']:



    return config_params


def get_feat_gt(df, not_a_feature, gt):
    cols = list(df.columns)

    for f in not_a_feature:
        cols.remove(f)

    X = df[cols].to_numpy()
    y = df[[gt]].values.ravel()


    return X, y


def get_dir_path():
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory of the current file
    project_dir = os.path.dirname(current_file_path)


    return project_dir


def get_readable_traces(df_log):
    return df_log.groupby('case:concept:name').agg(trace=('concept:name',list))


def get_clus_params(n_clusters_params):
    clus_params = {}
    clus_params['n_clusters_params'] = n_clusters_params

    if algorithm == 'kmeans':

        # kms_algorithm = ['elkan','lloyd']
        kms_algorithm = ['elkan']
        n_gram_params = {
                            # 1:[(0,1), (0.1,0.9), (0.2,0.8)],
                            # 2:[(0.1,0.9)],
                            1:[(0,1)],
                        }
       
        clus_params['n_gram_params'] = n_gram_params
        clus_params['kms_algorithm'] = kms_algorithm
    
    elif algorithm == 'agglom':
        # method_params = ['single','average']
        # metric_params = ['weighted_levenshtein']
        # ins_del_weight_params = [1, 2]
        # trans_weight_params = [1, 2]
        # subst_weight_params = [4, 8]
        # max_perc_params = [0.1, 0.2]

        method_params = ['single']
        metric_params = ['weighted_levenshtein']
        ins_del_weight_params = [4]
        trans_weight_params = [2]
        subst_weight_params = [4]
        max_perc_params = [0.2]

        clus_params['method'] = method_params
        clus_params['metric'] = metric_params
        clus_params['max_perc'] = max_perc_params
        clus_params['ins_del_weight'] = ins_del_weight_params
        clus_params['trans_weight'] = trans_weight_params
        clus_params['subst_weight'] = subst_weight_params

        project_dir = get_dir_path()

        clus_params['cash_path'] = project_dir + '/temp' + global_temp_folder + \
                                         '/agglom/'



    elif algorithm == 'actitrac':

        clus_params['target_fit'] = [1, 0.9]
        clus_params['is_greedy'] = [True]
        clus_params['dist_greed'] = [0.025]
        # clus_params['heu_miner_threshold'] = [0.9,0.000009]
        # clus_params['heu_miner_long_dist'] = [True]
        # clus_params['heu_miner_rel_best_thrs'] = [0.05, 0.005]
        # clus_params['heu_miner_and_thrs'] = [0.1,0.0001]
        clus_params['min_size_factor'] = [0.5,1]
        clus_params['heu_miner_config'] = [
            # {'heu_miner_threshold':0.9,
            #  'heu_miner_long_dist':True,
            #  'heu_miner_rel_best_thrs':0.05,
            #  'heu_miner_and_thrs':0.1
            # },
            {'heu_miner_threshold':0.09,
             'heu_miner_long_dist':True,
             'heu_miner_rel_best_thrs':0.005,
             'heu_miner_and_thrs':0.01
            },
            {'heu_miner_threshold':0.009,
             'heu_miner_long_dist':True,
             'heu_miner_rel_best_thrs':0.0005,
             'heu_miner_and_thrs':0.001
            },
            {'heu_miner_threshold':0.0009,
             'heu_miner_long_dist':True,
             'heu_miner_rel_best_thrs':0.00005,
             'heu_miner_and_thrs':0.0001
            },
        ]
        clus_params['include_external'] = [False,True]

        # clus_params['target_fit'] = [0.9]
        # clus_params['is_greedy'] = [True]
        # clus_params['dist_greed'] = [0.25]
        # clus_params['heu_miner_threshold'] = [0.9]
        # clus_params['heu_miner_long_dist'] = [True]
        # clus_params['heu_miner_rel_best_thrs'] = [0.05]
        # clus_params['heu_miner_and_thrs'] = [0.1]

        project_dir = get_dir_path()
        clus_params['log_path_train'] = project_dir + '/temp' + global_temp_folder + \
                                         '/actitrac/train/log_train.xes'
        clus_params['log_path_valid'] = project_dir + '/temp' + global_temp_folder + \
                                         '/actitrac/valid/log_valid.xes'
        clus_params['saving_path_train'] = project_dir + '/temp' + global_temp_folder + \
                                         '/actitrac/train/results/'
        clus_params['saving_path_valid'] = project_dir + '/temp' + global_temp_folder + \
                                         '/actitrac/valid/results/'
    
    else:
        raise Exception('invalid algorithm choice!')
    

    return clus_params


def get_clustering_best_params(
                               df_log,
                               df,
                               gt,
                               random_seed,
                               splits_kfold,
                               test_size,
                               algorithm,
                               n_clusters_params,
                               params_lgbm
                           ):

    # cols_run = [c for c in df.columns if c != 'case:concept:name']

    # Run LGBM only with very basic dataset
    # print('Run LGBM only with very basic dataset')

    # params_and_result, _ = lgbm.train_and_run_lgbm(df[cols_run],
    #                                                gt, 
    #                                                [gt], 
    #                                                params, 
    #                                                random_seed, 
    #                                                splits_kfold,
    #                                                number_cores,
    #                                                test_size)
    
    # print(params_and_result)

    config_params = [] 
    
    # n_clusters_params = [10]
    technique = None
    clus_params = get_clus_params(n_clusters_params)


    ###### Agglomerative params #######

    # t1 = 0
    # t2 = 100
    # t3 = 1000
 
    # bins = pd.IntervalIndex.from_tuples([
    #                                      (t1,t2),
    #                                      (t2,t3),
    #                                     ]
    #                                    )
    # labels = [0,1]

    dataset_split = DatasetSplit()

    # X_train_, X_test_, y_train_, y_test_ = dataset_split.\
    #     strat_train_test_split(df, 
    #                            None, 
    #                            None, 
    #                            gt, 
    #                            test_size, 
    #                            random_seed)
    
    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
        train_test_split(df, gt, test_size, random_seed)

        # strat_train_test_split(df, 
        #                        bins, 
        #                        labels, 
        #                        gt, 
        #                        test_size, 
        #                        random_seed)
    # 

    # skf = dataset_split.strat_kfold(X_train_,
    #                                 y_train_,
    #                                 splits_kfold, 
    #                                 random_seed)

    skf = dataset_split.kfold(X_train_,
                              y_train_,
                              splits_kfold, 
                              random_seed)
    skf_index = list(skf)

    best_mse = float('inf')
    best_r2 = float('-inf')
    best_mae = float('inf')
    best_config = None

    config_params = get_config_params(algorithm, clus_params)
    

    for config in config_params:
        if DEBUG:
            print('configuration: ' + str(config))
        
        mse = {'clus':[], 'base':[]}
        r2 = {'clus':[], 'base':[]}
        mae = {'clus':[], 'base':[]}
        count = 0

        for idx,(train_index, test_index) in enumerate(skf_index):

            # print('running kfold: ' + str(count) + '/' + str(splits_kfold))
            count += 1

            # X_trainkf, X_validkf = X_train_[train_index,:-1], X_train_[test_index,:-1]
            # y_trainkf, y_validkf = X_train_[train_index,-1], X_train_[test_index,-1]

            X_trainkf, X_validkf = X_train_[train_index], X_train_[test_index]
            y_trainkf, y_validkf = y_train_[train_index], y_train_[test_index]

            df_log_train = df_log[df_log['case:concept:name'].isin(X_trainkf[:,0])]
            log_train = pm4py.convert_to_event_log(df_log_train)

            df_log_valid = df_log[df_log['case:concept:name'].isin(X_validkf[:,0])]
            log_valid = pm4py.convert_to_event_log(df_log_valid)
            
            # df_model_train = df[df['case:concept:name'].isin(X_trainkf[:,0])].\
            #                     drop(columns='cat')
            # df_model_valid = df[df['case:concept:name'].isin(X_validkf[:,0])].\
            #                     drop(columns='cat')

            df_model_train = df[df['case:concept:name'].isin(X_trainkf[:,0])]
            df_model_valid = df[df['case:concept:name'].isin(X_validkf[:,0])]

            df_clus, df_clus_valid = obtain_cluster_feat(algorithm, 
                                                         config,
                                                         log_train,
                                                         log_valid,
                                                         idx,
                                                         df_model_train,
                                                         global_is_ohe)

            if df_clus.empty or df_clus_valid.empty:
                continue
            
            df_model_train_clus = df_model_train.merge(df_clus, 
                                                       on='case:concept:name',
                                                       how='inner').\
                                                    set_index('case:concept:name')
            
            df_model_valid_clus = df_model_valid.merge(df_clus_valid, 
                                                  on='case:concept:name',
                                                  how='inner').\
                                                    set_index('case:concept:name')
            
            df_model_train = df_model_train.set_index('case:concept:name')
            df_model_valid = df_model_valid.set_index('case:concept:name')

            if len(df_model_train_clus.index) != len(X_trainkf):
                raise Exception('invalid df_model size')

            if len(df_model_valid_clus.index) != len(X_validkf):
                raise Exception('invalid df_model size')
            
            y_valid_clus, y_pred_clus = get_model_predic(df_model_train_clus, 
                                                         df_model_valid_clus,
                                                         gt,
                                                         params_lgbm)
            
            y_valid, y_pred = get_model_predic(df_model_train, 
                                               df_model_valid,
                                               gt,
                                               params_lgbm)

            mse['clus'].append(mean_squared_error(y_valid_clus, y_pred_clus))
            mae['clus'].append(mean_absolute_error(y_valid_clus, y_pred_clus))
            r2['clus'].append(r2_score(y_valid_clus, y_pred_clus))

            mse['base'].append(mean_squared_error(y_valid, y_pred))
            mae['base'].append(mean_absolute_error(y_valid, y_pred))
            r2['base'].append(r2_score(y_valid, y_pred))

        mean_mse_clus = stats.mean(mse['clus'])
        mean_mae_clus = stats.mean(mae['clus'])
        mean_r2_clus = stats.mean(r2['clus'])

        mean_mse = stats.mean(mse['base'])
        mean_mae = stats.mean(mae['base'])
        mean_r2 = stats.mean(r2['base'])

        if mean_mse_clus < best_mse:
            best_mse = mean_mse_clus
            best_mae = mean_mae_clus
            best_r2 = mean_r2_clus
            best_config = config

            print('best r2: ' + str(best_r2))
            print('best config: ' + str(best_config))
            print()
            print('base r2: ' + str(mean_r2))

            with open('temp/results_' + algorithm + '_' + str(global_temp_folder) + \
                      '.txt','a+') as f:
                f.write('best r2: ' + str(best_r2) + '\n')
                f.write('base r2: ' + str(mean_r2) + '\n')
                f.write('best config: ' + str(best_config) + '\n\n')


    
    print('best parameters: ' + str(best_config))
    print('best mse: ' + str(best_mse))
    print('best r2: ' + str(best_r2))

    print('base performance:')
    print('best mse: ' + str(mean_mse))
    print('best r2: ' + str(mean_r2))


    return best_config


def get_clustering_features(best_params, 
                            df,
                            df_log,
                            random_seed,
                            test_size,
                            is_ohe
                            ):
    dataset_split = DatasetSplit()
    
    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
        strat_train_test_split(df, 
                               None, 
                               None, 
                               gt, 
                               test_size, 
                               random_seed,
                               'cat')
    
    X_train, X_test = X_train_[:,:-1], X_test_[:,:-1]
    y_train, y_test = X_train_[:,-1], X_test_[:,-1]

    df_log_train = df_log[df_log['case:concept:name'].isin(X_train[:,0])]
    df_log_test = df_log[df_log['case:concept:name'].isin(X_test[:,0])]

    log_train = pm4py.convert_to_event_log(df_log_train)
    log_test = pm4py.convert_to_event_log(df_log_test)

    df_model_train = df[df['case:concept:name'].isin(X_train[:,0])].\
                        drop(columns='cat')

    df_clus, df_clus_test = obtain_cluster_feat(algorithm, 
                                                best_params,
                                                log_train,
                                                log_test,
                                                -1,
                                                df_model_train,
                                                is_ohe)

    
    return df_clus, df_clus_test, list(X_train[:,0]), list(X_test[:,0])


def run_with_clust_feat(df, 
                         df_clus, 
                         df_clus_test, 
                         ids_train, 
                         ids_test,
                         params_lgbm):
    
    df_model_train = df[df['case:concept:name'].isin(ids_train)].\
                        drop(columns='cat')
    df_model_test = df[df['case:concept:name'].isin(ids_test)].\
                        drop(columns='cat')
   

    df_model_train_clus = df_model_train.merge(df_clus, 
                                            on='case:concept:name',
                                            how='inner').\
                                            set_index('case:concept:name')
    df_model_test_clus = df_model_test.merge(df_clus_test, 
                                            on='case:concept:name',
                                            how='inner').\
                                            set_index('case:concept:name')
    
    df_model_train = df_model_train.set_index('case:concept:name')
    df_model_test = df_model_test.set_index('case:concept:name')

    if len(df_model_train_clus.index) != len(ids_train):
        raise Exception('invalid df_model size')

    if len(df_model_test_clus.index) != len(ids_test):
        raise Exception('invalid df_model size')
    
    y_test_clus, y_pred_clus = get_model_predic(df_model_train_clus, 
                                                df_model_test_clus,
                                                gt,
                                                params_lgbm)
    
    # y_test, y_pred = get_model_predic(df_model_train, 
    #                                   df_model_test,
    #                                   gt,
    #                                   params_lgbm)

    print('### test performance with cluster features ###')
    print('mean squared error: ' + str(mean_squared_error(y_test_clus, y_pred_clus)))
    print('mean absolute error: ' +  str(mean_absolute_error(y_test_clus, y_pred_clus)))
    print('r2-score: ' + str(r2_score(y_test_clus, y_pred_clus)))

    # print('### test performance without cluster features ###')
    # print(mean_squared_error(y_test, y_pred))
    # print(mean_absolute_error(y_test, y_pred))
    # print(r2_score(y_test, y_pred))


if __name__ == "__main__":
    
    # log_path = 'clustering/test/test_cluster_feat.xes'
    # dataset_path = 'clustering/test/df_test_cluster_feat.csv'
    # splits_kfold = 3
    # test_size = 0.25
    
    algorithm = 'kmeans'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    cashe = {}
    number_cores = 4
    DEBUG = True
    random_seed = 3
    splits_kfold = 4
    test_size = 0.2
    is_merge_clus = False
    global_is_ohe = False
    global_temp_folder = ''
    best_params = None
    n_clusters_params = [40]

    log_path = 'dataset/tribunais_trabalho/TRT_micro.xes'
    dataset_path = 'dataset/tribunais_trabalho/dataset.csv'
    out_path = 'dataset/tribunais_trabalho/cluster_' + algorithm + '.csv'

    best_params = None

    sys.argv.append(4)
    sys.argv.append('actitrac')

    if not is_merge_clus:
        if len(sys.argv) > 1:
            number_cores = int(sys.argv[1])
        else:
            raise Exception('please provide number of cores to be used')
        
        if len(sys.argv) > 2:
            algorithm = sys.argv[2]
        else:
            raise Exception('please provide clustering algorithm')

        if algorithm == 'agglom':
            project_dir = get_dir_path()
            cash_path = project_dir + '/temp' + global_temp_folder + \
                                      '/agglom/'
            
            if not os.path.exists(cash_path):
                os.makedirs(cash_path)
    
            best_params = {
                'n_clusters': 40, 
                'method': 'average', 
                'max_size_perc': 0.2, 
                'metric': 'weighted_levenshtein', 
                'ins_del_weight': 1, 
                'trans_weight': 1, 
                'subst_weight': 4, 
                'cash_path': cash_path
            }

        elif algorithm == 'kmeans':
            best_params = {
                'n_clusters':60,
                'n_gram':1,
                'min_max_perc':(0,1),
                'kms_algorithm':'elkan',
            }

        elif algorithm == 'actitrac':
            project_dir = get_dir_path()
            cash_path = project_dir + '/temp' + global_temp_folder + \
                                      '/actitrac/'
            sav_path_train = cash_path + 'train/results/'
            sav_path_valid = cash_path + 'valid/results/'
            log_path_train = cash_path + 'train/log_train.xes'
            log_path_valid = cash_path + 'valid/log_valid.xes'

            if not os.path.exists(sav_path_train):
                os.makedirs(sav_path_train)

            if not os.path.exists(sav_path_valid):
                os.makedirs(sav_path_valid)

            best_params = {
                'n_clusters': 25, 
                'target_fit': 1, 
                'is_greedy': True, 
                'dist_greed': 0.025, 
                'min_clus_size': 0.02, 
                'heu_miner_long_dist': True, 
                'include_external': False,
                'heu_miner_threshold': 0.09, 
                'heu_miner_rel_best_thrs': 0.005, 
                'heu_miner_and_thrs': 0.01, 
                'saving_path_train': sav_path_train,
                'saving_path_valid': sav_path_valid,
                'log_path_train': log_path_train,
                'log_path_valid': log_path_valid,
            }

        best_mse = float('inf')
        best_r2 = float('-inf')
        best_params_and_result = {}
        
        params_lgbm_main = {}
        params_lgbm_main['boosting_type'] = 'dart'
        params_lgbm_main['learning_rate'] = 0.1
        params_lgbm_main['n_estimators'] = 300

        df_main = pd.read_csv(dataset_path, sep='\t')
        df_main['case:concept:name'] = df_main['case:concept:name'].astype(str)
        
        log = xes_importer.apply(log_path, 
                                variant=xes_importer.Variants.LINE_BY_LINE)
        df_log_main = convert_to_dataframe(log)
        df_log_main = df_log_main.sort_values(['case:concept:name','time:timestamp'])
        
        cols = list(df_log_main.columns)
        cols.remove('case:concept:name')
        cols = ['case:concept:name'] + cols
        df_log_main = df_log_main[cols]

        df_main = df_main[df_main['case:concept:name'].\
                        isin(df_log_main['case:concept:name'])]
        
        df_main = df_main[[
            'case:concept:name',
            # 'CASE:COURT:CODE',
            # 'CLASSE_PROCESSUAL',
            # 'MOV_CONCLUSAO_51',
            # 'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO',
            # 'MOVEMENTS_COUNT',
            # 'TOTAL_OFFICIAL',
            # 'MOV_DESARQUIVAMENTO_893',
            # 'TOTAL_MAGISTRATE',
            gt
        ]]

        print('df_main size: ' + str(len(df_main.index)))

        start = time.time()

        if best_params is None:
            best_params = get_clustering_best_params(
                                                    df_log_main,
                                                    df_main,
                                                    gt,
                                                    random_seed,
                                                    splits_kfold,
                                                    test_size,
                                                    algorithm,
                                                    n_clusters_params,
                                                    params_lgbm_main
                                                    )
        
        df_clus, df_clus_test, ids_train, ids_test = \
            get_clustering_features(best_params, 
                                    df_main,
                                    df_log_main,
                                    random_seed,
                                    test_size,
                                    global_is_ohe
                                )
        
        print('size df_clus: ' + str(len(df_clus.index)))
        print('size df_clus_test: ' + str(len(df_clus_test.index)))

        run_with_clust_feat(df_main, 
                            df_clus, 
                            df_clus_test, 
                            ids_train, 
                            ids_test,
                            params_lgbm_main)

        end = time.time()

        hours = round((end - start) / 3600,3)

        print('simulation duration (h): ' + str(hours))

        df_clus = pd.concat([df_clus,df_clus_test])
        df_clus['cluster_label'] = df_clus['cluster_label'].round(4)
        df_clus.to_csv(out_path, sep='\t', index=False)

        print('saved features!')
        
    else:
        techs = ['kmeans','agglom','actitrac']
        df_all = None
        out_path = 'dataset/tribunais_trabalho/cluster_feat_all.csv'

        for t in techs:
            path = 'dataset/tribunais_trabalho/cluster_'+ t +'.csv'
            name = t[:3].upper()
            df = pd.read_csv(path, sep='\t')
            df = df.rename(columns={'cluster_label':'CLUS_'+name})

            if df_all is None:
                df_all = df
            else:
                print('df_all before: ' + str(len(df_all.index)) + ' lines')

                df_all = df_all.merge(df, on='case:concept:name', how='inner')

                print('df_all after: ' + str(len(df_all.index)) + ' lines')

        df_all = df_all.round(4)
        df_all.to_csv(out_path, sep='\t', index=False)

        print('merged all cluster features!')

    print('done!')