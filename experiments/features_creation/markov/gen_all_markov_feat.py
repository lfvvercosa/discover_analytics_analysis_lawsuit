import pandas as pd


path1 = 'experiments/features_creation/feat_markov/feat_markov_k_1.csv'
path2 = 'experiments/features_creation/feat_markov/feat_markov_k_2.csv'
path3 = 'experiments/features_creation/feat_markov/feat_markov_k_3.csv'
align_path = 'experiments/results/metrics_mixed_dataset.csv'
out_path = 'experiments/features_creation/feat_markov/feat_markov_k_all.csv'

cols = ['EVENT_LOG','DISCOVERY_ALG','ABS_EDGES_ONLY_IN_LOG_W']

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
df_align = pd.read_csv(align_path)


df_final = df_align[['event_log','algorithm','fitness_alignment']]
df_final = df_final.rename(columns={'event_log':'EVENT_LOG',
                                    'algorithm':'DISCOVERY_ALG',
                                    'fitness_alignment':'FITNESS'})
df_final = df_final[df_final['FITNESS'] != -1]

df_merge = df3[cols]
df_merge = df_merge.rename(columns={'ABS_EDGES_ONLY_IN_LOG_W':'ABS_EDGES_ONLY_IN_LOG_W3'})

df_final = df_final.merge(df_merge, 
                          how='inner', 
                          on=['EVENT_LOG','DISCOVERY_ALG'])

df_merge = df2[cols]
df_merge = df_merge.rename(columns={'ABS_EDGES_ONLY_IN_LOG_W':'ABS_EDGES_ONLY_IN_LOG_W2'})

df_final = df_final.merge(df_merge, 
                          how='left', 
                          on=['EVENT_LOG','DISCOVERY_ALG'])

df_merge = df1[cols]
df_merge = df_merge.rename(columns={'ABS_EDGES_ONLY_IN_LOG_W':'ABS_EDGES_ONLY_IN_LOG_W1'})

df_final = df_final.merge(df_merge, 
                          how='left', 
                          on=['EVENT_LOG','DISCOVERY_ALG'])

df_final.to_csv(out_path, index=False, sep='\t')




print('done!')