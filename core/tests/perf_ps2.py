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
    log_path = 'dataset/tribunais_eleitorais/tre-ne.xes'
    dataset_path_v2 = 'dataset/tribunais_eleitorais/dataset_tre-ne_v2.csv'
    dataset_path_p2 = 'dataset/tribunais_eleitorais/dataset_tre-ne_p2.csv'
    DEBUG = True
    use_cash = False

    if len(sys.argv) > 1:
        use_cash = True

    best_msle = float('inf')
    best_r2 = float('-inf')
    best_params_and_result = {}

    jar_path = 'temp/actitrac/actitrac.jar'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    random_seed = 3
    splits_kfold = 10
    number_cores = 4

    actitrac = ActiTracConnector(jar_path)
    
    params = {}
    params['boosting_type'] = 'dart'
    params['learning_rate'] = 0.1
    params['n_estimators'] = 300

    df_v2 = pd.read_csv(dataset_path_v2, sep='\t')
    df_p2 = pd.read_csv(dataset_path_p2, sep='\t')
    feats = [
        'case:concept:name',
        'CASE:COURT:CODE',
        'PROCESSO_DIGITAL',
        'UF_ALAGOAS',
        'UF_CEARA',
        'UF_OUTRO',
        'UF_PARAIBA',
        'UF_PERNAMBUCO',
        'UF_PIAUI',
        'UF_RIO_GRANDE_DO_NORTE',
        'UF_SERGIPE',
    ]
    feats_mov = [c for c in df_p2.columns if 'MOV_' in c]
    feats_cla = [c for c in df_p2.columns if 'CLA_' in c]
    feats_assu = [c for c in df_p2.columns if 'ASSU_' in c]

    feats_p2 = feats + feats_mov + feats_cla + feats_assu
    cols = feats_p2 + [gt]
    
    df_p2 = df_p2[cols]
    df_p2 = df_p2[df_p2['case:concept:name'].isin(df_v2['case:concept:name'])]
    df_p2 = df_p2.drop(columns='case:concept:name')

    feats_mov = [c for c in df_v2.columns if 'MOV_' in c]
    feats_cla = [c for c in df_v2.columns if 'CLA_' in c]
    feats_assu = [c for c in df_v2.columns if 'ASSU_' in c]
    feats_v2 = feats + feats_mov + feats_cla + feats_assu
    cols = feats_v2 + [gt]

    df_v2 = df_v2[cols]
    df_v2 = df_v2.drop(columns='case:concept:name')

    # Run LGBM only with very basic dataset
    print('Run LGBM only with very basic dataset (P2)')

    params_and_result = lgbm.train_and_run_lgbm(df_p2, 
                                      gt, 
                                      [gt], 
                                      params, 
                                      random_seed, 
                                      splits_kfold,
                                      number_cores)
    
    print(params_and_result)

    # Run LGBM only with very basic dataset
    print('Run LGBM only with very basic dataset (V2)')

    params_and_result = lgbm.train_and_run_lgbm(df_v2, 
                                      gt, 
                                      [gt], 
                                      params, 
                                      random_seed, 
                                      splits_kfold,
                                      number_cores)
    
    print(params_and_result)
    


    


