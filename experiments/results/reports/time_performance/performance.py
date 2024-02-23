import pandas as pd
from pandas.api.types import is_numeric_dtype
from utils.creation.create_df_from_dict import df_from_dict


FIT_100 = 0
FIT_50_FREQ = 1
FIT_50_RAND = 2
FIT_25_FREQ = 3
FIT_25_RAND = 4
FIT_MARKOV = 5

PRE_100 = 6
PRE_50_FREQ = 7
PRE_50_RAND = 8
PRE_25_FREQ = 9
PRE_25_RAND = 10
PRE_MARKOV = 11


def remove_suffix_of_log(df, col='event_log'):
    df[col] = df[col].str.replace(r'\.xes\.gz$', '')
    df[col] = df[col].str.replace(r'\.xes$', '')

    return df


def keep_only_k3(df):
    aux_path = 'experiments/results/markov/k_3/df_markov_k_3.csv'
    df_aux = pd.read_csv(
                aux_path,
                sep='\t',
            )
    df_aux = df_aux[['EVENT_LOG', 'DISCOVERY_ALG']]
    df_aux = df_aux.rename(columns={
        'EVENT_LOG':'event_log', 
        'DISCOVERY_ALG':'algorithm',
    })
    df = df.merge(df_aux,
                  on=['EVENT_LOG', 'DISCOVERY_ALG'],
                  how='inner')
    
    return df


def create_df_all(k):
    k = str(k)
    path_metr = 'experiments/results/metrics_mixed_dataset.csv'
    path_metr_subset = 'experiments/subset_conf/results_subset_times.csv'
    path_model = 'experiments/results/markov/k_' + k + '/model_perf_k_' + \
                      k + '.csv'
    path_log = 'experiments/results/markov/k_' + k + '/log_perf_k_' + \
                      k + '.csv'
    out_path_comp = 'experiments/results/reports/performance/k_' + \
                      k + '.csv'

    df_subset = pd.read_csv(path_metr_subset, sep=',')
    df_metr = pd.read_csv(path_metr, sep=',')
    df_model = pd.read_csv(path_model, sep='\t')
    df_log = pd.read_csv(path_log, sep='\t')

    df_metr = keep_only_k3(df_metr)

    df_subset = df_subset.drop(['TIME_FITNESS_ALIGNMENT',
                                'TIME_PRECISION_ALIGNMENT'], axis=1)
    df_subset = df_subset.rename(columns={
                                    'EVENT_LOG':'event_log',
                                    'DISCOVERY_ALG':'algorithm',
                                    'TIME_50_FREQ_FIT ':'50_freq_fit',
                                    'TIME_50_FREQ_PREC ':'50_freq_pre',
                                    'TIME_50_RAND_FIT ':'50_rand_fit',
                                    'TIME_50_RAND_PREC ':'50_rand_pre',
                                    'TIME_25_FREQ_FIT ':'25_freq_fit',
                                    'TIME_25_FREQ_PREC ':'25_freq_pre',
                                    'TIME_25_RAND_FIT ':'25_rand_fit',
                                    'TIME_25_RAND_PREC ':'25_rand_pre',
                                })

    df_subset = remove_suffix_of_log(df_subset)
    df_metr = remove_suffix_of_log(df_metr)
    df_model = remove_suffix_of_log(df_model)
    df_log = remove_suffix_of_log(df_log)

    df_subset = df_subset.round(4)

    df_metr = df_metr[['event_log',
                       'algorithm',
                       'precision_time',
                       'fitness_time']]

    df_metr['align_time_fit'] = df_metr['fitness_time']
    df_metr['align_time_pre'] = df_metr['precision_time']

    df_metr = df_metr[['event_log',
                       'algorithm',
                       'align_time_fit',
                       'align_time_pre']]

    df_model = df_model[['event_log',
                         'algorithm',
                         'total_time']]
    
    df_log = df_log[['event_log',
                     'total_time']]
    

    df_model = df_model.rename(columns={'total_time':'model_time'})
    df_log = df_log.rename(columns={'total_time':'log_time'})
    df_markov = df_model.merge(df_log,
                               how='left',
                               on='event_log',
                              )
    df_markov['markov_fit_k_' + str(k)] = df_markov['model_time'] + df_markov['log_time']
    df_markov['markov_pre_k_' + str(k)] = df_markov['markov_fit_k_' + str(k)]

    df_markov = df_markov[['event_log',
                           'algorithm',
                           'markov_fit_k_' + str(k),
                           'markov_pre_k_' + str(k)]]

    df_perf = df_metr.merge(df_subset,
                            how='left',
                            on=['event_log','algorithm'],
                           )

    df_perf = df_perf.merge(df_markov,
                            how='left',
                            on=['event_log','algorithm'],
                           )

    df_perf.to_csv(out_path_comp, sep='\t', index=False)

    return df_perf


def get_perf_ratio(df_perf, k, timeout):
    stats = {
        'method':{},
        'mean_perc_all':{},
        'std_perc_all':{},
        'mean_perc_imf':{},
        'std_perc_imf':{},
        'mean_perc_imd':{},
        'std_perc_imd':{},
        'mean_perc_etm':{},
        'std_perc_etm':{},
    } 
    
    all_time_cols = fit_time_cols + pre_time_cols
    algs = ['IMf','IMd','ETM']

    for c in df_perf.columns:
        if is_numeric_dtype(df_perf[c]):
            df_perf = df_perf[df_perf[c] < timeout]
            df_perf = df_perf[df_perf[c] != -1]

    df_perf = df_perf.dropna()

    df_perf['highest_time_fit'] = df_perf[fit_time_cols].max(axis=1) 
    df_perf['highest_time_pre'] = df_perf[pre_time_cols].max(axis=1) 

    for c in fit_time_cols:
        df_perf[c] = df_perf[c] / df_perf['highest_time_fit']

    for c in pre_time_cols:
        df_perf[c] = df_perf[c] / df_perf['highest_time_pre']

    df_perf = df_perf.round(4)

    k = str(k)

    stats['method'][FIT_100] = 'align_time_fit'
    stats['method'][FIT_50_FREQ] = 'fit_50_freq'
    stats['method'][FIT_50_RAND] = 'fit_50_rand'
    stats['method'][FIT_25_FREQ] = 'fit_25_freq'
    stats['method'][FIT_25_RAND] = 'fit_25_rand'
    stats['method'][FIT_MARKOV] = 'markov_fit_k_' + str(k)
    stats['method'][PRE_100] = 'align_time_pre'
    stats['method'][PRE_50_FREQ] = 'pre_50_freq'
    stats['method'][PRE_50_RAND] = 'pre_50_rand'
    stats['method'][PRE_25_FREQ] = 'pre_25_freq'
    stats['method'][PRE_25_RAND] = 'pre_25_rand'
    stats['method'][PRE_MARKOV] = 'markov_pre_k_' + str(k)


    for c in all_time_cols:
        stats['mean_perc_all'][map_cols[c]] = df_perf[c].mean()
        stats['std_perc_all'][map_cols[c]] = df_perf[c].std()
    
    for a in algs:
        for c in all_time_cols:
            stats['mean_perc_' + a.lower()][map_cols[c]] = df_perf[df_perf['algorithm'] == a][c].mean()
            stats['std_perc_' + a.lower()][map_cols[c]] = df_perf[df_perf['algorithm'] == a][c].std()

    df_stats = pd.DataFrame.from_dict(stats)
    df_stats = df_stats.round(4)

    return df_stats


def get_perf(df_perf, k, timeout):
    
    stats = {
        'method':{},
        'timeouts':{},
        'mean_perc_all':{},
        'std_perc_all':{},
        'mean_perc_imf':{},
        'std_perc_imf':{},
        'mean_perc_imd':{},
        'std_perc_imd':{},
        'mean_perc_etm':{},
        'std_perc_etm':{},
    } 
    k = str(k)

    stats['method'][FIT_100] = 'align_time_fit'
    stats['method'][FIT_50_FREQ] = '50_freq_fit'
    stats['method'][FIT_50_RAND] = '50_rand_fit'
    stats['method'][FIT_25_FREQ] = '25_freq_fit'
    stats['method'][FIT_25_RAND] = '25_rand_fit'
    stats['method'][PRE_100] = 'align_time_pre'
    stats['method'][PRE_50_FREQ] = '50_freq_pre'
    stats['method'][PRE_50_RAND] = '50_rand_pre'
    stats['method'][PRE_25_FREQ] = '25_freq_pre'
    stats['method'][PRE_25_RAND] = '25_rand_pre'
    stats['method'][FIT_MARKOV] = 'markov_fit_k_' + str(k)
    stats['method'][PRE_MARKOV] = 'markov_pre_k_' + str(k)

    stats['timeouts'][FIT_100] = \
         df_perf['align_time_fit'][df_perf['align_time_fit'] >= timeout].count()
    stats['timeouts'][FIT_50_FREQ] = \
         df_perf['50_freq_fit'][df_perf['50_freq_fit'] == -1].count()
    stats['timeouts'][FIT_50_RAND] = \
         df_perf['50_rand_fit'][df_perf['50_rand_fit'] == -1].count()
    stats['timeouts'][FIT_25_FREQ] = \
         df_perf['25_freq_fit'][df_perf['25_freq_fit'] == -1].count()
    stats['timeouts'][FIT_25_RAND] = \
         df_perf['25_rand_fit'][df_perf['25_rand_fit'] == -1].count()
    stats['timeouts'][PRE_100] = \
         df_perf['align_time_pre'][df_perf['align_time_pre'] >= timeout].count()
    stats['timeouts'][PRE_50_FREQ] = \
         df_perf['50_freq_pre'][df_perf['50_freq_pre'] == -1].count()
    stats['timeouts'][PRE_50_RAND] = \
         df_perf['50_rand_pre'][df_perf['50_rand_pre'] == -1].count()
    stats['timeouts'][PRE_25_FREQ] = \
         df_perf['25_freq_pre'][df_perf['25_freq_pre'] == -1].count()
    stats['timeouts'][PRE_25_RAND] = \
         df_perf['25_rand_pre'][df_perf['25_rand_pre'] == -1].count()
    stats['timeouts'][FIT_MARKOV] = \
         df_perf['markov_fit_k_' + str(k)].isna().sum()
    stats['timeouts'][PRE_MARKOV] = \
         df_perf['markov_pre_k_' + str(k)].isna().sum()

    all_time_cols = fit_time_cols + pre_time_cols
    algs = ['IMf','IMd','ETM']

    for c in df_perf.columns:
        if is_numeric_dtype(df_perf[c]):
            df_perf = df_perf[df_perf[c] < timeout]
            df_perf = df_perf[df_perf[c] != -1]

    df_perf = df_perf.dropna()

    for c in all_time_cols:
        stats['mean_perc_all'][map_cols[c]] = df_perf[c].mean()
        stats['std_perc_all'][map_cols[c]] = df_perf[c].std()
    
    for a in algs:
        for c in all_time_cols:
            stats['mean_perc_' + a.lower()][map_cols[c]] = df_perf[df_perf['algorithm'] == a][c].mean()
            stats['std_perc_' + a.lower()][map_cols[c]] = df_perf[df_perf['algorithm'] == a][c].std()

    df_stats = pd.DataFrame.from_dict(stats)
    df_stats = df_stats.round(4)

    return df_stats


if __name__ == '__main__':
    k = 1
    timeout = 7200
    out_path = 'experiments/results/reports/performance/stats_k_' + \
                      str(k) + '.csv'
    out_path_ratio = 'experiments/results/reports/performance/stats_k_' + \
                      str(k) + '_percent.csv'

    map_cols = {
        'align_time_fit': FIT_100,
        '50_freq_fit': FIT_50_FREQ,
        '50_rand_fit': FIT_50_RAND,
        '25_freq_fit': FIT_25_FREQ,
        '25_rand_fit': FIT_25_RAND,
        'align_time_pre': PRE_100,
        '50_freq_pre': PRE_50_FREQ,
        '50_rand_pre': PRE_50_RAND,
        '25_freq_pre': PRE_25_FREQ,
        '25_rand_pre': PRE_25_RAND,
        'markov_fit_k_' + str(k): FIT_MARKOV,
        'markov_pre_k_' + str(k): PRE_MARKOV,
    }

    fit_time_cols = [
        'align_time_fit',
        '50_freq_fit',
        '50_rand_fit',
        '25_freq_fit',
        '25_rand_fit',
        'markov_fit_k_' + str(k),
    ]
    pre_time_cols = [
        'align_time_pre',
        '50_freq_pre',
        '50_rand_pre',
        '25_freq_pre',
        '25_rand_pre',
        'markov_pre_k_' + str(k),
    ]

    df_perf = create_df_all(k)
    df_stats = get_perf(df_perf, k, timeout)
    df_stats_ratio = get_perf_ratio(df_perf, k, timeout)

    df_stats.to_csv(out_path, sep='\t', index=False)
    df_stats_ratio.to_csv(out_path_ratio, sep='\t', index=False)

    print('done!')