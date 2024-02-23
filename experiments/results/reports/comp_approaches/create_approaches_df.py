import pandas as pd


ref_path = 'experiments/results/markov/k_1/df_markov_k_1.csv'
align_path = 'experiments/results/metrics_mixed_dataset.csv'
prop_path = 'experiments/features_creation/feat_markov/' + \
               'feat_markov_align_4_k_1.csv'
prop2_path = 'experiments/features_creation/feat_markov/' + \
               'feat_markov_align_5_k_1.csv'               
bl_path = 'experiments/features_creation/feat_markov/' + \
               'feat_markov_align_bl_k_1.csv'
subset_perf_path = 'experiments/features_creation/feat_subset/' + \
                    'results_subset_metrics.csv'
subset_time_path = 'experiments/features_creation/feat_subset/' + \
                    'results_subset_times.csv'


df_ref = pd.read_csv(ref_path, sep='\t')
df_ref = df_ref[['EVENT_LOG','DISCOVERY_ALG']]

df_align = pd.read_csv(align_path, sep=',')
df_align = df_align[[
                    'event_log',	
                    'algorithm',		
                    'fitness_alignment',		
                    'fitness_time',
                    ]]
df_align = df_align.rename(columns={'event_log':'EVENT_LOG',
                                    'algorithm':'DISCOVERY_ALG',
                                    'fitness_alignment':'ALIGN',
                                    'fitness_time':'TIME_ALIGN',
                                    })
df_align['EVENT_LOG'] = df_align['EVENT_LOG'].str.replace(r'\.xes\.gz$', '')
df_align['EVENT_LOG'] = df_align['EVENT_LOG'].str.replace(r'\.xes$', '')

df_prop = pd.read_csv(prop_path, sep='\t')
df_prop = df_prop.rename(columns={'TIME_ALIGN':'TIME_PROP',
                                  'ALIGNMENTS_MARKOV_4_K1':'PROP'})

df_prop2 = pd.read_csv(prop2_path, sep='\t')
df_prop2 = df_prop2.rename(columns={'TIME_ALIGN':'TIME_PROP2',
                                  'ALIGNMENTS_MARKOV_4_K1':'PROP2'})

df_bl = pd.read_csv(bl_path, sep='\t')
df_bl = df_bl.rename(columns={'TIME_ALIGN':'TIME_BL',
                              'ALIGNMENTS_MARKOV_4_K1':'BL'})

df_subset = pd.read_csv(subset_perf_path, sep=',')
df_subset = df_subset[[
                        'EVENT_LOG',
                        'DISCOVERY_ALG',
                        'FITNESS_50_FREQ',
                        'FITNESS_25_FREQ'
                     ]]
df_subset_time = pd.read_csv(subset_time_path, sep=',')
df_subset_time = df_subset_time[[
                                   'EVENT_LOG',
                                   'DISCOVERY_ALG',
                                   'TIME_50_FREQ_FIT ',
                                   'TIME_25_FREQ_FIT '
                                ]]
df_subset = df_subset.merge(df_subset_time,
                            how='inner',
                            on=['EVENT_LOG','DISCOVERY_ALG'])
df_subset = df_subset.rename(columns={'TIME_50_FREQ_FIT ':'TIME_FREQ_50',
                                      'FITNESS_50_FREQ':'FREQ_50',
                                      'TIME_25_FREQ_FIT ':'TIME_FREQ_25',
                                      'FITNESS_25_FREQ':'FREQ_25'})
df_subset = df_subset.round(4)


df_approaches = df_ref[['EVENT_LOG','DISCOVERY_ALG']]
df_approaches = df_approaches.merge(df_align,
                                    how='left',
                                    on=['EVENT_LOG','DISCOVERY_ALG'])
df_approaches = df_approaches[df_approaches['ALIGN'] != -1]

df_approaches = df_approaches.merge(df_prop2,
                                    how='left',
                                    on=['EVENT_LOG','DISCOVERY_ALG'])
df_approaches = df_approaches.merge(df_prop,
                                    how='left',
                                    on=['EVENT_LOG','DISCOVERY_ALG'])
df_approaches = df_approaches.merge(df_bl,
                                    how='left',
                                    on=['EVENT_LOG','DISCOVERY_ALG'])
df_approaches = df_approaches.merge(df_subset,
                                    how='left',
                                    on=['EVENT_LOG','DISCOVERY_ALG'])


df_approaches = df_approaches[[
    'EVENT_LOG',
    'DISCOVERY_ALG',
    'ALIGN',
    'PROP2',
    'PROP',
    'BL',
    'FREQ_50',
    'FREQ_25',
    'TIME_ALIGN',
    'TIME_PROP2',
    'TIME_PROP',
    'TIME_BL',
    'TIME_FREQ_50',
    'TIME_FREQ_25',
]]

df_approaches.to_csv('experiments/results/reports/' + \
                     'comp_approaches/df_approaches.csv', sep='\t', index=False)

print(df_approaches)
print('done!')

