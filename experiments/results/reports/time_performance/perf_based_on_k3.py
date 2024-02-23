import pandas as pd


def remove_suffix_of_log(df, col='event_log'):
    df[col] = df[col].str.replace(r'\.xes\.gz$', '')
    df[col] = df[col].str.replace(r'\.xes$', '')

    return df


k = 2
path_model = 'experiments/results/markov/k_' + str(k) + '/model_perf_k_' + \
             str(k) + '.csv'
aux_path = 'experiments/results/markov/k_3/df_markov_k_3.csv'

df_model = pd.read_csv(path_model, sep='\t')
df_aux = pd.read_csv(aux_path, sep='\t')
df_aux = df_aux[['EVENT_LOG', 'DISCOVERY_ALG']]
df_aux = df_aux.rename(columns={
    'EVENT_LOG':'event_log', 
    'DISCOVERY_ALG':'algorithm',
})

df_model = remove_suffix_of_log(df_model)
df = df_model.merge(df_aux,
                    on=['event_log', 'algorithm'],
                    how='inner')

mean_time = df[['total_time']].mean()
std = df[['total_time']].std()

print('mean time: ' + str(mean_time))
print('std: ' + str(std))

