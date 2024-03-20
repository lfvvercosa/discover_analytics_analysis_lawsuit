from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe
import pm4py

import pandas as pd

import core.my_loader as my_loader
from core import my_create_features


if __name__ == "__main__":
    base_path = 'dataset/'
    log_path = 'dataset/tribunais_trabalho/TRT.xes'
    congest_path = 'dataset/proc_taxa_congestionamento_ujs.csv'
    pend_path = 'dataset/proc_pendentes_serie_ujs.csv'
    ibge_path = 'dataset/ibge_info_municipio.csv'
    out_path = 'dataset/tribunais_trabalho/dataset_raw.csv'
    clus_path = 'dataset/tribunais_trabalho/cluster_feat_all.csv'
    DEBUG = True

    df_congest = pd.read_csv(congest_path, sep='\t')
    df_pend = pd.read_csv(pend_path, sep='\t')
    df_ibge = pd.read_csv(ibge_path, sep='\t')
    df_code_subj = my_loader.load_df_subject(base_path)
    df_code_mov = my_loader.load_df_movements(base_path)
    df_clus = pd.read_csv(clus_path, sep='\t')

    ngram = 1
    min_perc = 0.05
    max_perc = 0.95
    level_subject = 1

    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    df_log = convert_to_dataframe(log)

    print('creating dataset...')

    # Create feature dataset
    df_feat = my_create_features.create_features(df_log,
                                                 df_code_subj,
                                                 df_code_mov,
                                                 df_pend,
                                                 df_congest,
                                                 df_ibge,
                                                 df_clus,
                                                 ngram,
                                                 min_perc,
                                                 max_perc,
                                                 level_subject)

    
    df_feat.to_csv(out_path, sep='\t', index=False)
    
    print('dataset created!')

