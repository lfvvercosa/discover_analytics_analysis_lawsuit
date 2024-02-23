from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe
import pm4py

import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

import core.my_loader as my_loader
import core.my_stats as my_stats
from core import my_create_features


if __name__ == "__main__":
    court_type = 'tribunais_trabalho'
    input_path = 'dataset/tribunais_trabalho/dataset_trt_raw.csv'
    # input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    stats_path = 'dataset/' + court_type + '/statistics/'
    target = 'total_time'
    col_key_mov = 'case:lawsuit:percent_key_mov'
    # target = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    # col_key_mov = 'CASE:LAWSUIT:PERCENT_KEY_MOV'

    Path(stats_path + 'numeric/').mkdir(parents=True, exist_ok=True)

    df_feat = pd.read_csv(input_path, sep='\t')
    

    # Check percentile of numeric columns
    categoric_cols = [
        'case:lawsuit:type',
        'case:digital_lawsuit',
        'Município',
        'Classificação',
        'Classificação da unidade',
        'Tipo',
        'Justiça',
        'Tribunal',
        'Tipo de unidade',
        'Unidade Judiciária',
        'Município sede',
    ]
    # categoric_cols = [
    #     'CLASSE_PROCESSUAL',
    #     # 'CLASSIFICACAO_DA_UNIDADE',
    #     # 'TIPO',
    #     # 'TRIBUNAL',
    #     'NUMERO_TRT',
    #     'PROCESSO_DIGITAL',
    # ]

    if 'case:concept:name' in df_feat.columns:
        df_feat = df_feat.drop(columns='case:concept:name')

    df_feat = df_feat.drop_duplicates()
    numeric_cols = [c for c in df_feat.columns if c not in categoric_cols]

    # Create path
    Path(stats_path + 'numeric/').mkdir(parents=True, exist_ok=True)
    
    # Number of rows and columns
    print('df_feat shape: ' + str(df_feat.shape))

    # Target distribution
    print('save histogram of target')
    s_plot = df_feat[target]
    # s_plot /= 365
    color1 = (0.16696655132641292, 0.48069204152249134, 0.7291503267973857)
    my_axes = s_plot.hist(bins=30,
                          edgecolor='black',
                          fill=True,
                          color=color1)
    my_axes.set_xlabel('Time (days)', labelpad=14, fontsize=14)
    my_axes.set_ylabel('Count', labelpad=14, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim(0, 5000)
    plt.grid(False)
    # plt.figure(figsize=(8, 6))
    # plt.title('Distribution of Proceedings Total Time')
    plt.savefig(stats_path + 'distrib_target.png', 
                bbox_inches='tight',
                dpi=500,
                )

    # Check presence of key movements
    my_stats.percent_key_mov(df_feat, [246,22,848])


    # Nulls percentage
    percent_missing = df_feat.isnull().sum() * 100 / len(df_feat)
    df_null = pd.DataFrame({'column_name': df_feat.columns,
                            'percent_missing': percent_missing})
    
    df_null.to_csv(stats_path + 'nulls.csv', sep='\t', index=False)

    df_percentile = df_feat[numeric_cols].quantile([.05,.25,.5,.75,.95], 
                                                   interpolation='nearest').T
    df_percentile.to_csv(stats_path + 'percentiles.csv', sep='\t', index=True)
    
    # Check distribution of categoric columns
    # Check correlation of categoric columns with target
    for c in categoric_cols:
        df_distrib = my_stats.get_categoric_distrib(df_feat, c, target)
        print(df_distrib)

    regexp = re.compile('^ASSU_')
    subj_cols = [c for c in df_feat.columns if regexp.search(c)]
    # dfs = []

    print('## Distribution "Assuntos"')
    
    for c in subj_cols:
        df_temp = df_feat.groupby(c).agg(count=(c,'count'),
                                         mean=(target,'mean'),
                                         std=(target,'std'))

        print(df_temp)
        # df_temp = df_temp.reset_index(drop=False)
        # df_temp[c] = c + '_' + df_temp[c].astype(str)
        # df_temp = df_temp.rename(columns={c:'ASSUNTO'})
        # dfs.append(df_temp)

    print('## Distribution "Movimentos"')
    regexp = re.compile('^MOV_')
    mov_cols = [c for c in df_feat.columns if regexp.search(c)]
    
    df_work = df_feat[mov_cols + [target]]
    # df_work[df_work[mov_cols] > 1] = 1

    for c in mov_cols:
        df_temp = df_work.groupby(c).agg(count=(c,'count'),
                                         mean=(target,'mean'),
                                         std=(target,'std'))

        print(df_temp)

    print('\n## Distribution "Clustering \n"')
    
    regexp = re.compile('^CLUS_')
    mov_cols = [c for c in df_feat.columns if regexp.search(c)]
    
    df_work = df_feat[mov_cols + [target]]
    # df_work[df_work[mov_cols] > 1] = 1

    for c in mov_cols:
        df_temp = df_work.groupby(c).agg(count=(c,'count'),
                                         mean=(target,'mean'),
                                         std=(target,'std'))

        print(df_temp)

    # Check distribution of position of key 
    print('## Distribution of position of key movement')
    print(my_stats.key_mov_distrib(df_feat, col_key_mov, target))

    # Check correlation of numeric cols
    df_numeric_corr = df_feat[numeric_cols].corr(method='pearson')
    df_numeric_corr.to_csv(stats_path + 'corr_pearson.csv', sep='\t', index=True)

    df_numeric_corr = df_feat[numeric_cols].corr(method='spearman')
    df_numeric_corr.to_csv(stats_path + 'corr_spearman.csv', sep='\t', index=True)

    print('done!')
