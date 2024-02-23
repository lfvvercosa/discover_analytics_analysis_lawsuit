import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    my_path = 'experiments/features_creation/' + \
              'feat_markov/feat_markov_k_1_dfg.csv'
    df = pd.read_csv(my_path, sep='\t')
    cols = list(df.columns)
    cols.remove('EVENT_LOG')
    cols.remove('DISCOVERY_ALG')
    
    plt.figure(figsize=(15,75))

    for i in range(len(cols)):
        # plt.subplot(8,2,i+1)
        plt.hist(df[cols[i]])
        plt.title(cols[i])
        plt.show(block=True)
