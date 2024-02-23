import pandas as pd
from sklearn.metrics import mean_absolute_error


if __name__ == '__main__':
    all_path = 'experiments/results/markov/df_all.csv'
    df = pd.read_csv(all_path, sep='\t')
    thrs = 0.2

    df_etm = df[df['DISCOVERY_ALG'] == 'ETM']
    df_imd = df[df['DISCOVERY_ALG'] == 'IMd']
    df_imf = df[df['DISCOVERY_ALG'] == 'IMf']


    print('align k3 error for etm: ')
    print(mean_absolute_error(df_etm['FITNESS'],
                              df_etm['ALIGN_K3']))

    print('align k3 error for imd: ')
    print(mean_absolute_error(df_imd['FITNESS'],
                              df_imd['ALIGN_K3']))

    print('align k3 error for imf: ')
    print(mean_absolute_error(df_imf['FITNESS'],
                              df_imf['ALIGN_K3']))
    