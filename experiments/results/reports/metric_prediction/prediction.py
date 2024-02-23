import pandas as pd
from sklearn.metrics import mean_absolute_error
import statistics

 
if __name__ == '__main__':
    markov_path = 'experiments/results/markov/k_2/df_markov_k_2.csv'
    df = pd.read_csv(markov_path, sep='\t')

    print('')
    print('MAE (All): ' + str(mean_absolute_error(df['FITNESS'],
                                              df['ABS_EDGES_ONLY_IN_LOG_W'])))
    print('STD (All): ' + str(statistics.stdev(df['FITNESS'] - \
                                           df['ABS_EDGES_ONLY_IN_LOG_W'])))
    
    df2 = df[df['DISCOVERY_ALG'] == 'ETM']

    print('')
    print('MAE (ETM): ' + str(mean_absolute_error(df2['FITNESS'],
                                              df2['ABS_EDGES_ONLY_IN_LOG_W'])))
    print('STD (ETM): ' + str(statistics.stdev(df2['FITNESS'] - \
                                           df2['ABS_EDGES_ONLY_IN_LOG_W'])))

    df2 = df[df['DISCOVERY_ALG'] == 'IMf']

    print('')
    print('MAE (IMf): ' + str(mean_absolute_error(df2['FITNESS'],
                                              df2['ABS_EDGES_ONLY_IN_LOG_W'])))
    print('STD (IMf): ' + str(statistics.stdev(df2['FITNESS'] - \
                                           df2['ABS_EDGES_ONLY_IN_LOG_W'])))

    df2 = df[df['DISCOVERY_ALG'] == 'IMd']

    print('')
    print('MAE (IMd): ' + str(mean_absolute_error(df2['FITNESS'],
                                              df2['ABS_EDGES_ONLY_IN_LOG_W'])))
    print('STD (IMd): ' + str(statistics.stdev(df2['FITNESS'] - \
                                           df2['ABS_EDGES_ONLY_IN_LOG_W'])))