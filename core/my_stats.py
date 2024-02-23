import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def is_in_previous_loop(pos, loop_acts):
    for p in pos:
        if p in loop_acts:
            return True
    
    return False


def loops_by_trace(l, only_consecutive = False):
    size = len(l)
    count_loops = 0
    loop_acts = []
   

    for i in range(size):
        loop = [l[i]]

        for j in range(i+1,size):
            k = 0
            pos = []

            while j + k < size and \
                  k < len(loop) and \
                  l[j + k] == loop[k]:
                        pos.append(j + k)
                        k += 1

            if only_consecutive:
                if len(pos) == len(loop):
                    if not is_in_previous_loop(pos, loop_acts):
                        count_loops += 1
                        loop_acts += pos
            elif len(pos) > 0:
                if not is_in_previous_loop(pos, loop_acts):
                    count_loops += 1
                    loop_acts += pos

            if j + k < size:
                loop.append(l[j + k])


    return count_loops


def get_loops_by_cluster(df_mov, id_col, act_col, c1, c2, only_consec):
    print('### get_loops_by_cluster')
    
    # df_work = df_mov.groupby(id_col).agg(act)

    df_work = df_mov.groupby(id_col).agg({act_col:list, 'cluster_label':min})
    df_work['loop_count'] = df_work.\
                            apply(lambda df_work: loops_by_trace(
                                                    df_work[act_col],
                                                    only_consec
                                                                ), axis=1
                                 )
    
    s_c1 = df_work[df_work['cluster_label'].isin(c1[1])]['loop_count']
    s_c2 = df_work[df_work['cluster_label'].isin(c2[1])]['loop_count']


    return {'' + str(c1[0]) : s_c1, 
            '' + str(c2[0]) : s_c2}


def get_traces_duration(df_mov, id='id', time='movimentoDataHora', measure='W'):
    df_work = df_mov.groupby(id).\
        agg({time:['min','max']})
    df_work['duration'] = (df_work[(time, 'max')] - \
                           df_work[(time, 'min')])\
                           / np.timedelta64(1, measure)

    
    return df_work[['duration']]


def get_number_of_movs(df_gram, 
                       df, 
                       class_list, 
                       thrs_dur_min,
                       thrs_dur_max, 
                       clus_list):
    cols = list(df_gram.columns)
    cols.remove('cluster_label')

    df_sum = df_gram[cols]
    df_sum = df_sum.sum(axis=1).to_frame(name='total_movs')
    df_sum = df_sum.join(df[['classeProcessual','duration','cluster_label']])
    df_sum = df_sum[df_sum['classeProcessual'].isin(class_list)]
    df_sum = df_sum[df_sum['cluster_label'].isin(clus_list)]
    
    if thrs_dur_min:
        df_sum = df_sum[df_sum['duration'] > thrs_dur_min]

    if thrs_dur_max:
        df_sum = df_sum[df_sum['duration'] < thrs_dur_max]
    
    return df_sum['total_movs']


def get_class_in_clusters(df, my_classes, clusters):
    df_work = df[df['cluster_label'].isin(clusters)]
    # df_work = df_work[df_work['classeProcessual'].isin(my_classes)]

    df_work = df_work.groupby(['classeProcessual'])['processoNumero'].count()

    df_work = df_work.sort_values()

                            
    print(df_work.groupby(['classeProcessual','cluster_label'])\
                          ['processoNumero'].count())

    print()


def get_most_freq_val(s, n):
    return s.value_counts()[:n].index.tolist()


def get_percent_traces_by_clusters(perc_cluster, cluster_labels):
    new_cluster_labels = [c for c in cluster_labels if c != -1]
    cluster_count = Counter(new_cluster_labels)
    most_common = cluster_count.most_common()

    n = int(perc_cluster * len(most_common))
    total_n = 0
    total = 0

    for i in range(len(most_common)):
        if i < n:
            total_n += most_common[i][1]
        
        total += most_common[i][1]

    print('percent clusters: ' + str(perc_cluster))
    print('number of clusters: ' + str(n))
    print('percent traces: ' + str(total_n/total))
    print()


def get_attribute_level(breadscrum, desired_level):
    breadscrum = breadscrum.split(':')

    if len(breadscrum) > desired_level:
        return int(breadscrum[desired_level])
    else:
        return -1


def get_attribute_traces(df_code, df_traces, level, code_col, name_col):
    
    temp = df_code[[code_col,'breadscrum']]
    df_traces = df_traces.merge(temp, on=[code_col], how='left')

    if level is not None:
        df_traces['code_new_level'] = df_traces.apply(lambda df: \
            get_attribute_level(
                df['breadscrum'],
                level), axis=1)
    else:
         df_traces['code_new_level'] = df_traces[code_col]
    
    df_traces = df_traces.groupby('code_new_level').\
                    agg(count=(code_col, 'count'))

    df_traces = df_traces.sort_values('count', ascending=False)
    df_traces[code_col] = df_traces.index 

    temp = df_code[[code_col,name_col]]
    df_traces = df_traces.merge(temp, on=[code_col], how='left')

    total = df_traces['count'].sum()
    df_traces['Percent'] = (df_traces['count']/total).round(2)


    return df_traces[[name_col, code_col, 'Percent']]


def format_class(d, col, val, classes):
    f = {col:[]}
    
    for v in val:
        for c in classes:
            if c not in f:
                f[c] = []
            if v not in f[col]:
                f[col].append(v)

            if (v,c) in d:
                f[c].append(d[(v,c)])
            else:
                f[c].append(0)

    
    return pd.DataFrame.from_dict(f)


def get_class_perc(df):
    df['TOTAL'] = df.sum(axis=1)

    for c in df.columns:
        df['PERC_' + str(c)] = df[c]/df['TOTAL']

    
    return df


def get_categoric_distrib(df_feat, col, target):
    df_time = df_feat.groupby(col).\
                      agg(COUNT=(col,'count'), 
                          MEAN_TIME=(target,'mean'),
                          STD_TIME=(target,'std'))
    df_time = df_time.sort_values(by='MEAN_TIME', ascending=False)
    df_time = df_time.round(2)
    df_time = df_time[['MEAN_TIME','STD_TIME','COUNT']]

    # df_time = df_time.drop(columns='COUNT')
    # df_class = df_class.drop(columns='PERC_TOTAL')

    # df_distrib = df_time.merge(df_class, on=df_time.index, how='inner')
    # df_distrib = df_distrib.rename(columns={'key_0':col})

    return df_time


def categorics_distrib(df_feat, cols, out_path):
    for c in cols:
        print('distribution for feature ' + c)
        df_distrib = get_categoric_distrib(df_feat, 
                                           c,
                                           'TEMPO_PROCESSUAL_TOTAL_DIAS')

        df_distrib.to_csv(out_path +'/distrib_' + c + '.csv', 
                        sep='\t', 
                        index=True)
        

def numerics_distrib(df_feat, cols, out_path):
    for c in cols:
        print('distribution for feature ' + c)
        my_axes = df_feat[c].hist()
        my_axes.set_xlabel('Valor')
        my_axes.set_ylabel('Quantidade')
        plt.title('DISTRIB ' + c)
        plt.savefig(out_path + 'numeric/distrib_'+ c +'.png', 
                    bbox_inches='tight')
        plt.figure().clear()


def key_mov_distrib(df_feat, col, target):
    print('Key mov distribution')
    df_work = df_feat[[col,target]]
    df_work = df_work.round(1)
    df_work = df_work.groupby(col).agg(count=(col,'count'),
                                       mean=(target,'mean'),
                                       std=(target,'std'))

    return df_work


def time_distrib(df_feat, out_path):
    col = 'TEMPO_PROCESSUAL_TOTAL_DIAS'

    my_axes = df_feat[col].hist(bins=40)
    my_axes.set_xlabel('Valor')
    my_axes.set_ylabel('Quantidade')
    plt.title('DISTRIB ' + col)
    plt.savefig(out_path + 'numeric/distrib_'+ col.lower() +'.png', 
                bbox_inches='tight')
    plt.figure().clear()


def frequency_start_end_movs(df_mov):
    df_temp = df_mov.groupby('case:concept:name').agg(
                        mov_first=('concept:name','first'),
                        mov_last=('concept:name','last')
                    )
    
    print('frequency start movement:')
    df_start = df_temp.groupby('mov_first').agg(count=('mov_first','count'))
    df_start = df_start.sort_values('count', ascending=False)
    print(df_start)
    print()
    
    print('frequency end movement:')
    df_end = df_temp.groupby('mov_last').agg(count=('mov_last','count'))
    df_end = df_end.sort_values('count', ascending=False)
    print(df_end)
    print()


def percent_key_mov(df_feat, movs):
    print('Total traces: ' + str(len(df_feat.index)))
    
    for m in movs:
        col = [c for c in df_feat.columns if str(m) == c[c.rfind('_')+1:]]

        if col:
            col = col[0]
            print('Total ' + col + ' : ' + str(len(df_feat[df_feat[col] > 0])))

    for i in range(len(movs)):

        if i + 1 == len(movs):
            j = 0
        else:
            j = i + 1

        mov1 = movs[i]
        mov2 = movs[j]

        col1 = [c for c in df_feat.columns if str(mov1) == c[c.rfind('_')+1:]]        
        col2 = [c for c in df_feat.columns if str(mov2) == c[c.rfind('_')+1:]]

        if col1 and col2:
            col1 = col1[0]
            col2 = col2[0]

            print('Total ' + col1 + ' and ' + col2 + ': ' + \
                str(len(df_feat[(df_feat[col1] > 0) & (df_feat[col2] > 0)])))
        
    for i in range(len(movs)-2):
        mov1 = movs[i]
        mov2 = movs[i+1]
        mov3 = movs[i+2]

        col1 = [c for c in df_feat.columns if str(mov1) == c[c.rfind('_')+1:]]        
        col2 = [c for c in df_feat.columns if str(mov2) == c[c.rfind('_')+1:]]
        col3 = [c for c in df_feat.columns if str(mov3) == c[c.rfind('_')+1:]]

        if col1 and col2 and col3:
            col1 = col1[0]
            col2 = col2[0]
            col3 = col3[0]

            print('Total ' + col1 + ',' + col2 + ' and ' + col3 + ': ' + \
                str(len(df_feat[(df_feat[col1] > 0) & (df_feat[col2] > 0) & \
                            (df_feat[col3] > 0)])))
            

def time_bottlenecks(df_mov, percent, bins):
    df_work = df_mov[['case:concept:name','concept:name','time:timestamp']]
    df_work['next'] = df_work.groupby('case:concept:name')\
                                    ['time:timestamp'].shift(-1)
    df_work['transition'] = df_work['next'] - df_work['time:timestamp']
    df_work = df_work.drop(columns='next')
    df_work = df_work.sort_values(['case:concept:name','transition'],ascending=False)
    df_work['count_by_trace'] =  df_work.groupby('case:concept:name').cumcount()

    df_temp = df_work.groupby('case:concept:name').agg(trace_perc=('case:concept:name',
                                                                   'count'))
    df_temp = (df_temp * percent).round(0)
    df_work = df_work.merge(df_temp, on='case:concept:name',how='left')
    df_work = df_work[df_work['count_by_trace'] <= df_work['trace_perc']]
    df_work = df_work.groupby('case:concept:name').agg(time_bottlenecks=('transition','sum'))

    df_time = df_mov.groupby('case:concept:name').agg(max_time=('time:timestamp','max'),
                                                       min_time=('time:timestamp','min'))
    df_time['total_time'] = df_time['max_time'] - df_time['min_time']
    df_time = df_time.drop(columns=['max_time','min_time'])
    df_time['category'] = pd.cut(df_time['total_time'], bins)

    df_work = df_work.merge(df_time, on='case:concept:name', how='left')
    df_work['percent_bottleneck'] = (df_work['time_bottlenecks']/df_work['total_time']).round(2)


    return df_work.groupby('category').agg(mean_bottleneck=('percent_bottleneck','mean'),
                                           std_bottleneck=('percent_bottleneck','std'),
                                           count=('category','count'))