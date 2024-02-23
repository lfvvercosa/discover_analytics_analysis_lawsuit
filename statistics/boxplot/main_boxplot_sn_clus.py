import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pm4py
import statistics as stats
import sys
from sklearn.preprocessing import MinMaxScaler
from pm4py.objects.log.importer.xes import importer as xes_importer
from collections import Counter

from core import my_loader
from core import my_create_features


FAST = 0
MEDIUM = 1
SLOW = 2


def filter_cluster(df_log, df_feat, cluster):
    df_selec = df_feat[df_feat['cluster_label'] == cluster]['case:concept:name']
    df = df_log[df_log['case:concept:name'].isin(df_selec)]
    
    
    return df


def get_condition(df_feat, clus, cat):
    if cat == FAST:
        cond = (df_feat['total_time'] <= (365*2)) & \
               (df_feat['cluster_label'] == clus)
    elif cat == MEDIUM:
        cond = (df_feat['total_time'] > (365*2)) & \
               (df_feat['total_time'] <= (365*7)) & \
               (df_feat['cluster_label'] == clus)
    elif cat == SLOW:
        cond = (df_feat['total_time'] > (365*7)) & \
               (df_feat['cluster_label'] == clus)
    

    return cond


if __name__ == "__main__":
    tech = 'kmeans'
    my_path = 'dataset/tribunais_trabalho/cluster_'+ tech + '.csv'
    df_path = 'dataset/tribunais_trabalho/dataset_model.csv'
    metric = 'median'
    gt = "TEMPO_PROCESSUAL_TOTAL_DIAS"
    gtNewName = 'Time (days)'
    out_path = 'dataset/tribunais_trabalho/statistics/corr_'+ tech + '.png'

    n = 10

    if len(sys.argv) > 1:   
        cond_cat = int(sys.argv[1])
    else:
        cond_cat = MEDIUM

    df_cluster = pd.read_csv(my_path, sep='\t')
    df_feat = pd.read_csv(df_path, sep='\t')

    df_feat = df_feat[['case:concept:name'] + [gt]]
    df_feat = df_feat.merge(df_cluster, on='case:concept:name', how='left')
    # df_feat = my_create_features.rename_mov_cols(df_feat, df_code_mov, 1)

    


    df_size = df_feat.groupby('cluster_label').agg(count=('case:concept:name','count'))
    df_size = df_size.sort_values('count', ascending=False)
    df_size = df_size.head(n)

    df_box = df_feat[df_feat['cluster_label'].isin(list(df_size.index))]
    df_box['cluster_label'] = df_box['cluster_label'].astype('category').cat.codes
    df_box = df_box.rename(columns={gt:gtNewName, 'cluster_label':'Cluster'})
   
    sns.set(font_scale=1.9)
    sns.boxplot(
        x = 'Cluster',
        y = gtNewName,
        showmeans=True,  # Show means as diamonds
        data=df_box,
        palette="Blues",
    )

    plt.xlabel('Cluster', labelpad=20)
    plt.ylabel(gtNewName, labelpad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400)


    print('done!')


   