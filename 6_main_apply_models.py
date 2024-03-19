from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from core.my_visual import gen_bar_plot
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from scipy import stats
import pandas as pd
import numpy as np
import re
import boto3
import sys
import matplotlib.pyplot as plt
import seaborn as sns

import models.svr as svr
import models.lgbm as lgbm
import models.adaboost as ada
import models.naive as naive
import models.linear as linear


def apply_best_features(df):
    top_10_gain = [
        'CLUS_AGG',
        'CLUS_KME',
        'NUMERO_TRT',
        'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO',
        'PROCESSO_DIGITAL',
        'MOV_DESARQUIVAMENTO_893',
        'TOTAL_ASSUNTOS_DISTINTOS',
        'MOV_ENTREGA_EM_CARGA_VISTA_493',
        'TOTAL_MAGISTRATE',
        'MOVEMENTS_COUNT',
    ]
    
    top_10_split = [
        'NUMERO_TRT',
        'CLUS_KME',
        'CLASSE_PROCESSUAL',
        'CLUS_AGG',
        'MOVEMENTS_COUNT',
        'TOTAL_OFFICIAL',
        'MOV_CONCLUSAO_51',
        'PROCESSO_DIGITAL',
        'MOV_EXPEDICAO_DE_DOCUMENTO_60',
        'CLUS_ACT',
    ]

    my_params = {'params':{}}
    my_params['params']['boosting_type'] = 'dart'
    my_params['params']['learning_rate'] = 0.2
    my_params['params']['n_estimators'] = 600
    my_params['params']['importance_type'] = 'split'


    # get best params
    params, _ = lgbm.train_and_run_lgbm(df[top_10_gain + [gt]], 
                                                    gt, 
                                                    [gt], 
                                                    my_params, 
                                                    random_seed, 
                                                    4,
                                                    4,
                                                    0.2)
    
    params_lgbm = {
        'params':{
            'boosting_type': params['params']['boosting_type'],
            'learning_rate': params['params']['learning_rate'],
            'n_estimators': params['params']['n_estimators'],
            'importance_type': 'gain',
        }
    }

    
    results = {'gain':{}, 'split':{}}
    feat = []
    count = 1

    for f in top_10_gain:
        feat.append(f)
        params_and_result, _ = lgbm.train_and_run_lgbm(df[feat + [gt]], 
                                                       gt, 
                                                       [gt], 
                                                       params_lgbm, 
                                                       random_seed, 
                                                       4,
                                                       4,
                                                       0.2)
        results['gain'][count] = \
            round(params_and_result['training_perf']['R2_avg'],4)
        count += 1

    feat = []
    count = 1

    params_lgbm['params']['importance_type'] = 'split'

    for f in top_10_split:
        feat.append(f)
        params_and_result, _ = lgbm.train_and_run_lgbm(df[feat + [gt]] , 
                                                       gt, 
                                                       [gt], 
                                                       params_lgbm, 
                                                       random_seed, 
                                                       4,
                                                       4,
                                                       0.2)
        results['split'][count] = \
            round(params_and_result['training_perf']['R2_avg'],4)
        count += 1

    with open('temp/feat_import_only_best.txt', 'w') as f:
        f.write(str(params_lgbm['params']) + '\n\n')
        f.write(str(results))


def apply_groups_with_best_model(df, gt):

    regexp = re.compile('^MOV_')
    mov_cols = [c for c in df.columns if regexp.search(c)]

    regexp = re.compile('^CLUS_')
    clus_cols = [c for c in df.columns if regexp.search(c)]

    count_cols = ['TOTAL_MAGISTRATE', 'TOTAL_OFFICIAL', 'MOVEMENTS_COUNT']

    reg_cols = list(df.columns)
    reg_cols = [c for c in reg_cols if c not in mov_cols and \
                                       c not in count_cols and \
                                       c not in clus_cols and 
                                       c != gt]


    features_group = {
        'REGULAR': reg_cols,
        'CLUSTERING': clus_cols,
        'MOVEMENTS': mov_cols + count_cols,
    }

    # params_lgbm = None
    params_lgbm = {'params':{}}
    params_lgbm['params']['boosting_type'] = 'dart'
    params_lgbm['params']['learning_rate'] = 0.2
    params_lgbm['params']['n_estimators'] = 600
    params_lgbm['params']['importance_type'] = 'split'

    
    inc_groups = []
    groups_name = []

    ### Get performance by groups incrementally

    for group in features_group:
        inc_groups.append(features_group[group])
        groups_name.append(group)
        cols = []

        print('groups: ' + str(groups_name))

        for g in inc_groups:
            cols += g
        
        cols += [gt]

        params_and_results, df_feat_import = \
                            lgbm.train_and_run_lgbm(df[cols], 
                                                    gt, 
                                                    [gt], 
                                                    params_lgbm, 
                                                    random_seed, 
                                                    splits_kfold,
                                                    number_cores,
                                                    test_size)

        print(params_and_results)
        print(df_feat_import)

        with open(out_results_inc, 'a+') as f:
            f.write('groups: ' + str(groups_name) + '\n\n')
            f.write(str(params_and_results) + '\n\n')
            f.write('feature importance: \n\n')
            f.write(str(df_feat_import) + '\n\n')

    if DEBUG:
        df_feat_import.to_csv('temp/lgbm_feat_import.csv', sep='\t')


def apply_discrete_results(df, params_lgbm):
    t1 = 0
    t2 = 365*2
    t3 = 365*5
    t4 = 365*8
    t5 = 365*100

    bins = pd.IntervalIndex.from_tuples([(t1,t2),
                                         (t2,t3),
                                         (t3,t4),
                                         (t4,t5),
                                         ]
                                        )
    # labels = ['very fast','fast','medium','slow','very slow']
    labels = [0,1,2,3]

    params_lgbm['discrete'] = {'bins':bins, 'labels':labels}

    df_results = lgbm.train_and_run_lgbm(df, 
                                         gt, 
                                         [gt], 
                                         params_lgbm, 
                                         random_seed, 
                                         splits_kfold,
                                         number_cores,
                                         test_size)
    
    res = precision_recall_fscore_support(df_results['cat_y_test'],
                                          df_results['cat_y_pred'], 
                                          average='weighted'
                                         )

    df_test = df_results.copy()
    df_test['cat_1_test'] = np.where(df_test['cat_y_test'] != 1, 0, 1)
    df_test['cat_1_pred'] = np.where(df_test['cat_y_pred'] != 1, 0, 1)
    confusion_mat_cat_0 = confusion_matrix(df_test['cat_1_test'], df_test['cat_1_pred'])
    precision_cat_0 = confusion_mat_cat_0[1, 1] / \
                     (confusion_mat_cat_0[1, 1] + confusion_mat_cat_0[0, 1])
    print("Precision (cat_0):", precision_cat_0)

    print()

    confusion_mat = confusion_matrix(df_results['cat_y_test'], 
                                            df_results['cat_y_pred'])
    print(confusion_mat)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print()


if __name__ == '__main__':
    DEBUG = True
    out_results = 'dataset/tribunais_trabalho/results_by_feat_group.txt'
    out_results_inc = 'dataset/tribunais_trabalho/results_by_feat_group_inc.txt'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    random_seed = 3
    splits_kfold = 10
    number_cores = 4
    test_size = 0.2
    dataset_path = 'dataset/tribunais_trabalho/dataset_model.csv'
    test_best_features = False
    test_group_features = False
    test_discrete = False

    if len(sys.argv) > 1:   
        list_algo = eval(sys.argv[1])
    else:
        list_algo = ['lgbm']

    if len(sys.argv) > 2:
        if sys.argv[4] == 'group_features':
            test_group_features = True
        elif sys.argv[4] == 'best_features':
            test_best_features = True
        elif sys.argv[4] == 'discrete':
            test_discrete = True

    df = pd.read_csv(dataset_path, sep='\t')

    regexp = re.compile('EXTRAJUDICIAL')
    jud_cols = [c for c in df.columns if regexp.search(c)]

    regexp = re.compile('SUSPENSOS')
    susp_cols = [c for c in df.columns if regexp.search(c)]

    regexp = re.compile('RECURSOS')
    rec_cols = [c for c in df.columns if regexp.search(c)]

    rem_cols = jud_cols + susp_cols + rec_cols
    rem_cols += [
        'case:concept:name',
        'CASE:LAWSUIT:PERCENT_KEY_MOV',
        'TAXA_DE_CONGESTIONAMENTO_LIQUIDA',
        'TAXA_DE_CONGESTIONAMENTO_TOTAL',
        'ESTOQUE',
        'INDEX',
        'CASE:COURT:CODE',
        'MOV_ATO_ORDINATORIO_11383',
        'MOV_DISTRIBUICAO_26',
        'MOV_DECISAO_3',
        'MOV_AUDIENCIA_970',
        'ASSU__PARTES_E_PROCURADORES',
    ]

    df = df[[c for c in df.columns if c not in rem_cols]]
    df = df.drop_duplicates()
   
    print('type list_algo: ' + str(type(list_algo)))
    print('list_algo: ' + str(list_algo))

    if DEBUG:
        print('### Total records: ' + str(len(df.index)))
        print('### Total features: ' + str(len(df.columns) - 1))
    

    if DEBUG:
        print('#####################')
        print('#### apply Naive ####')
        print('#####################\n')

    naive.run_naive(df, splits_kfold)    

    if DEBUG:
        print('#############################')
        print('## apply Linear Regression ##')
        print('#############################\n')

    linear.run_linear_regression(df.copy(), 
                                 splits_kfold,
                                 random_seed,
                                 test_size
                                 )
    
    params_lgbm = {'params':{}}
    params_lgbm['params']['boosting_type'] = 'dart'
    params_lgbm['params']['learning_rate'] = 0.2
    params_lgbm['params']['n_estimators'] = 600
    params_lgbm['params']['importance_type'] = 'split'

    if test_best_features:
        apply_best_features(df)
    elif test_group_features:
        apply_groups_with_best_model(df, gt)
    elif test_discrete:
        apply_discrete_results(df, params_lgbm)
    else:
        if 'lgbm' in list_algo:

            if DEBUG:
                print('#####################')
                print('#### apply LGBM ####')
                print('#####################\n')

            

            params_and_results, df_import = lgbm.train_and_run_lgbm(df.copy(), 
                                                                    gt, 
                                                                    [gt], 
                                                                    params_lgbm, 
                                                                    random_seed, 
                                                                    splits_kfold,
                                                                    number_cores,
                                                                    test_size)
            
            print('### LGBM Results ###')
            print('### Training Performance ###')
            print('R2_avg: ' + str(params_and_results['training_perf']['R2_avg']))
            print('R2_std: ' + str(params_and_results['training_perf']['R2_std']))
            print('MAE_avg: ' + str(params_and_results['training_perf']['MAE_avg']))
            print('MAE_std: ' + str(params_and_results['training_perf']['MAE_std']))
            print('MSE_avg: ' + str(params_and_results['training_perf']['MSE_avg']))
            print('MSE_std: ' + str(params_and_results['training_perf']['MSE_std']))

            print('### Test Performance ###')
            print('R2: ' + str(params_and_results['test_perf']['R2']))
            print('MAE: ' + str(params_and_results['test_perf']['MAE']))
            print('MSE: ' + str(params_and_results['test_perf']['MSE']))

            print('### Feature Importance ###')
            print(df_import)

            # print(params_and_results)
            df_import.to_csv('temp/lgbm_feat_import.csv', sep='\t')
            
            print('### Feat Import ###')
            print(df_import)
            
        if 'svr' in list_algo:
            if DEBUG:
                print('#####################')
                print('##### apply SVR #####')
                print('#####################\n')

            # params = None
            params = {}
            params['C'] = 4096
            params['kernel'] = 'rbf'
            params['epsilon'] = 0.1

            best_params_model, df_import = svr.run_svr(df.copy(), 
                                                        gt, 
                                                        params, 
                                                        random_seed, 
                                                        splits_kfold, 
                                                        test_size)
            print(best_params_model)
            df_import.to_csv('dataset/tribunais_trabalho/statistics/svr_feat_import.csv', sep='\t')
  
    print('done!')