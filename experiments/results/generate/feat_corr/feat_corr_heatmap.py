import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    k = 2
    dataset_path = 'experiments/results/markov/k_' + str(k) + \
                   '/df_markov_k_' +  str(k) + '.csv'
    
    df = pd.read_csv(dataset_path, sep='\t')

    plt.figure(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()