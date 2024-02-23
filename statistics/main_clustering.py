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
    my_path = 'dataset/tribunais_trabalho/cluster_agglom.csv'
    df_path = 'dataset/tribunais_trabalho/dataset_raw.csv'
    log_path = 'dataset/tribunais_trabalho/TRT_mini.xes'
    metric = 'median'
    perc = 0.3

    n = 10

    if len(sys.argv) > 1:   
        cond_cat = int(sys.argv[1])
    else:
        cond_cat = MEDIUM

    df_code_mov = my_loader.load_df_movements('dataset/')
    df_cluster = pd.read_csv(my_path, sep='\t')
    df_feat = pd.read_csv(df_path, sep='\t')
    log = xes_importer.apply(log_path, 
                            variant=xes_importer.Variants.LINE_BY_LINE)
    df_log = pm4py.convert_to_dataframe(log)
    # df_log = my_create_features.rename_movs(df_log, df_code_mov)

    regexp = re.compile('^MOV_')
    mov_cols = [c for c in df_feat.columns if regexp.search(c)]

    df_feat = df_feat[mov_cols + ['case:concept:name'] + ['total_time']]
    df_feat = df_feat.merge(df_cluster, on='case:concept:name', how='left')
    # df_feat = my_create_features.rename_mov_cols(df_feat, df_code_mov, 1)


    df_size = df_feat.groupby('cluster_label').agg(count=('case:concept:name','count'))
    df_size = df_size.sort_values('count', ascending=False)
    df_size = df_size.head(n)

    l = list(df_size.index)
    l.sort()
    medium_value = l[int(len(l)/2)]
    larger_value = df_size.index.max() 
    smaller_value = df_size.index.min() 

    cond_fast = get_condition(df_feat, smaller_value, FAST)
    cond_medium = get_condition(df_feat, medium_value, MEDIUM)
    cond_slow = get_condition(df_feat, larger_value, SLOW)

    perc_fast = len(df_feat[cond_fast]) / \
                len(df_feat[df_feat['cluster_label'] == smaller_value]) * 100
    perc_medium = len(df_feat[cond_medium]) / \
                len(df_feat[df_feat['cluster_label'] == medium_value]) * 100
    perc_slow = len(df_feat[cond_slow]) / \
                len(df_feat[df_feat['cluster_label'] == larger_value]) * 100


    print(perc_fast)
    print(perc_medium)
    print(perc_slow)

    if cond_cat == FAST:
        cond = cond_fast
        save_name = 'temp/main_bottlenecks_fast.png'
    elif cond_cat == MEDIUM:
        cond = cond_medium
        save_name = 'temp/main_bottlenecks_medium.png'
    elif cond_cat == SLOW:
        cond = cond_slow
        save_name = 'temp/main_bottlenecks_slow.png'

    conds = [(cond_fast,'temp/main_bottlenecks_fast.png'), 
             (cond_medium,'temp/main_bottlenecks_medium.png'), 
             (cond_slow,'temp/main_bottlenecks_slow.png')]

    for t in conds:
        print('processing cond ' + str(t[1]) + '...')
        cond = t[0]
        save_name = t[1]

        df_cat = df_feat[cond]
        df_log_curr = df_log[df_log['case:concept:name'].isin(list(df_cat['case:concept:name']))]
        dfg, sa, ea = pm4py.discover_dfg(df_log_curr)
        perf_dfg, sa, ea = pm4py.discover_performance_dfg(df_log_curr)

        my_counter = Counter(dfg)
        most_common = my_counter.most_common(int(len(my_counter) * perc)) 
        most_common = [c[0] for c in most_common]

        perf_dfg_filt = {k:v[metric] for k,v in perf_dfg.items() if k in most_common}
        perf_dfg_filt_sorted = dict(sorted(perf_dfg_filt.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))
        ten_pairs = list(perf_dfg_filt_sorted.items())[:10]
        
        x = [str(pair[0]) for pair in ten_pairs]
        # x = list(range(10))
        y = [pair[1] for pair in ten_pairs]

        plt.bar(x, y)
        plt.xlabel('DFG')
        plt.ylabel('Performance')
        plt.title('Top 10 DFGs by Performance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(save_name)

    print('done!')
    

    






