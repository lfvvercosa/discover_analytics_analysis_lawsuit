import pandas as pd
import numpy as np


def print_stats_time(df):
    print()
    print('######## ' + str(df['DISCOVERY_ALG'].iloc[0]) + ' #######')
    print()

    print('time fitness_alignment: ' + str(round(df['fitness_time'].mean(),2)))
    print('std fitness_alignment: ' + str(round(df['fitness_time'].std(),2)))
    print()

    print('time subset 50%: ' + str(round(df['TIME_50_FREQ_FIT'].mean(),2)))
    print('std subset 50%: ' + str(round(df['TIME_50_FREQ_FIT'].std(),2)))
    print()

    print('time subset 25%: ' + str(round(df['TIME_25_FREQ_FIT'].mean(),2)))
    print('std subset 25%: ' + str(round(df['TIME_25_FREQ_FIT'].std(),2)))
    print()

    print('time new align: ' + str(round(df['TIME_ALIGN'].mean(),2)))
    print('std new align: ' + str(round(df['TIME_ALIGN'].std(),2)))
    print()


def print_stats_corr(df):
    print()
    print('######## ' + str(df['DISCOVERY_ALG'].iloc[0]) + ' #######')
    print()
    df_temp = df[['fitness_alignment',
                  'ALIGNMENTS_MARKOV_4_K1',
                  'FITNESS_50_FREQ',
                  'FITNESS_25_FREQ']]
    print(df_temp.corr(method='pearson')['fitness_alignment'])
    print()


align_path = 'experiments/features_creation/feat_markov/'+\
             'feat_markov_align_4_k_1.csv'

metrics_path = 'experiments/results/metrics_mixed_dataset.csv'
subset_etm_path = 'experiments/subset_conf/ETM.csv'
subset_imd_path = 'experiments/subset_conf/IMd.csv'
subset_imf_path = 'experiments/subset_conf/IMf.csv'


df_align = pd.read_csv(align_path, sep='\t')
df_metrics = pd.read_csv(metrics_path, sep=',')
df_subset_etm = pd.read_csv(subset_etm_path, sep=',')
df_subset_imd = pd.read_csv(subset_imd_path, sep=',')
df_subset_imf = pd.read_csv(subset_imf_path, sep=',')


df_metrics = df_metrics[['event_log',
                         'algorithm',
                         'fitness_time',
                         'fitness_alignment']]

df_metrics = df_metrics.rename(columns={'event_log':'EVENT_LOG',
                                        'algorithm':'DISCOVERY_ALG'})

df_subset_etm = df_subset_etm.rename(columns={'PETRI-NETS':'EVENT_LOG'})
df_subset_etm = df_subset_etm.rename(columns=lambda x: x.strip())
df_subset_etm['DISCOVERY_ALG'] = 'ETM'



df_subset_imd = df_subset_imd.rename(columns={'PETRI-NETS':'EVENT_LOG'})
df_subset_imd = df_subset_imd.rename(columns=lambda x: x.strip())
df_subset_imd['DISCOVERY_ALG'] = 'IMd'

df_subset_imf = df_subset_imf.rename(columns={'PETRI-NETS':'EVENT_LOG'})
df_subset_imf = df_subset_imf.rename(columns=lambda x: x.strip())
df_subset_imf['DISCOVERY_ALG'] = 'IMf'

df_metrics['EVENT_LOG'] = df_metrics['EVENT_LOG'].str.replace(r'\.xes\.gz$', '')
df_metrics['EVENT_LOG'] = df_metrics['EVENT_LOG'].str.replace(r'\.xes$', '')

print('total records: ' + str(len(df_metrics.index)))

df_time = df_align.merge(df_metrics,
                         how='inner',
                         on=['EVENT_LOG', 'DISCOVERY_ALG'])

print('total records: ' + str(len(df_time.index)))

df_time = df_time[df_time['fitness_alignment'] != -1]

print('total records: ' + str(len(df_time.index)))

df_etm = df_time[df_time['DISCOVERY_ALG'] == 'ETM']
df_imf = df_time[df_time['DISCOVERY_ALG'] == 'IMf']
df_imd = df_time[df_time['DISCOVERY_ALG'] == 'IMd']

df_etm = df_etm.merge(df_subset_etm,
                         how='inner',
                         on=['EVENT_LOG', 'DISCOVERY_ALG'])
df_imf = df_imf.merge(df_subset_imf,
                         how='inner',
                         on=['EVENT_LOG', 'DISCOVERY_ALG'])
df_imd = df_imd.merge(df_subset_imd,
                         how='inner',
                         on=['EVENT_LOG', 'DISCOVERY_ALG'])

df_etm['TIME_50_FREQ_FIT'] = df_etm['TIME_50_FREQ_FIT'].astype(float)
df_etm['TIME_25_FREQ_FIT'] = df_etm['TIME_25_FREQ_FIT'].astype(float)
df_etm['FITNESS_50_FREQ'] = df_etm['FITNESS_50_FREQ'].astype(float)
df_etm['FITNESS_25_FREQ'] = df_etm['FITNESS_25_FREQ'].astype(float)

df_imf['TIME_50_FREQ_FIT'] = df_imf['TIME_50_FREQ_FIT'].astype(float)
df_imf['TIME_25_FREQ_FIT'] = df_imf['TIME_25_FREQ_FIT'].astype(float)
df_imf['FITNESS_50_FREQ'] = df_imf['FITNESS_50_FREQ'].astype(float)
df_imf['FITNESS_25_FREQ'] = df_imf['FITNESS_25_FREQ'].astype(float)

df_imd['TIME_50_FREQ_FIT'] = df_imd['TIME_50_FREQ_FIT'].astype(float)
df_imd['TIME_25_FREQ_FIT'] = df_imd['TIME_25_FREQ_FIT'].astype(float)
df_imd['FITNESS_50_FREQ'] = df_imd['FITNESS_50_FREQ'].astype(float)
df_imd['FITNESS_25_FREQ'] = df_imd['FITNESS_25_FREQ'].astype(float)


print_stats_time(df_etm)
print_stats_time(df_imd)
print_stats_time(df_imf)
print_stats_corr(df_etm)
print_stats_corr(df_imd)
print_stats_corr(df_imf)









