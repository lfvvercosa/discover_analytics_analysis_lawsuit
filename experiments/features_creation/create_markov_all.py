import pandas as pd


if __name__ == '__main__':
    k1_path = 'experiments/results/markov/k_1' + \
              '/df_markov_k_1.csv'
    k2_path = 'experiments/results/markov/k_2' + \
              '/df_markov_k_2.csv'
    k3_path = 'experiments/results/markov/k_3' + \
              '/df_markov_k_3.csv'
    out_path = 'experiments/results/markov/df_all.csv'

    df_k1 = pd.read_csv(k1_path, sep='\t')
    df_k2 = pd.read_csv(k2_path, sep='\t')
    df_k3 = pd.read_csv(k3_path, sep='\t')

    df_k1 = df_k1[[
        'EVENT_LOG',
        'DISCOVERY_ALG',
        'FITNESS',
        'ALIGNMENTS_MARKOV',
        'ABS_EDGES_ONLY_IN_LOG_W',
        'ALIGNMENTS_MARKOV_2_K1',
        'ALIGNMENTS_MARKOV_3_K1',
        'ALIGNMENTS_MARKOV_4_K1',
    ]]

    df_k1 = df_k1.rename(columns={'ALIGNMENTS_MARKOV':'ALIGN_K1',
                                  'ABS_EDGES_ONLY_IN_LOG_W':'EDGES_K1'})

    df_k2 = df_k2[[
        'EVENT_LOG',
        'DISCOVERY_ALG',
        'ALIGNMENTS_MARKOV',
        'ABS_EDGES_ONLY_IN_LOG_W',
    ]]

    df_k2 = df_k2.rename(columns={'ALIGNMENTS_MARKOV':'ALIGN_K2',
                                  'ABS_EDGES_ONLY_IN_LOG_W':'EDGES_K2'})

    df_k3 = df_k3[[
        'EVENT_LOG',
        'DISCOVERY_ALG',
        'ALIGNMENTS_MARKOV',
        'ABS_EDGES_ONLY_IN_LOG_W',
    ]]

    df_k3 = df_k3.rename(columns={'ALIGNMENTS_MARKOV':'ALIGN_K3',
                                  'ABS_EDGES_ONLY_IN_LOG_W':'EDGES_K3'})

    df = df_k1.merge(df_k2, 
                     how='inner',
                     on=['EVENT_LOG','DISCOVERY_ALG'])
    
    df = df.merge(df_k3, 
                  how='inner',
                  on=['EVENT_LOG','DISCOVERY_ALG'])

    df = df[df['FITNESS'] != -1]

    # df['ERR_ALIGN_K1'] = df['FITNESS'] - df['ALIGN_K1']
    # df['ERR_ALIGN_K1'] = df['ERR_ALIGN_K1'].abs()

    # df_small_err = df[]

    df.to_csv(out_path, sep='\t')

    print('done!')

