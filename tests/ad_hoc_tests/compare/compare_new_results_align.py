import pandas as pd


my_path = 'experiments/features_creation/feat_markov/'+\
          'feat_markov_align_2_comp_k_1.csv'
k1_path = 'experiments/features_creation/feat_markov/'+\
          'feat_markov_k_1.csv'
metrics_path = 'experiments/results/metrics_mixed_dataset.csv'

df = pd.read_csv(my_path, sep='\t')
df_k1 = pd.read_csv(k1_path, sep=',')
df_me = pd.read_csv(metrics_path, sep=',')

df_k1 = df_k1[['EVENT_LOG','DISCOVERY_ALG','ALIGNMENTS_MARKOV']]
df_me = df_me.rename(columns={'event_log':'EVENT_LOG',
                              'algorithm':'DISCOVERY_ALG',
                              'fitness_alignment':'FITNESS'})

df_me = df_me[['EVENT_LOG','DISCOVERY_ALG','FITNESS']]

df = df.merge(df_k1, 'left', ['EVENT_LOG','DISCOVERY_ALG'])
df = df.merge(df_me, 'left', ['EVENT_LOG','DISCOVERY_ALG'])

print(df)

df.to_csv('temp/new_results_align.csv', sep='\t', index=None)