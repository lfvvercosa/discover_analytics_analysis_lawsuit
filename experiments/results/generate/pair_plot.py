import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    my_path = 'experiments/results/markov/k_1/df_markov_k_1.csv'
    gt = 'FITNESS'
    cols = [
        'ALIGNMENTS_MARKOV_4_K1',
        # 'ABS_LOG_MAX_DEGREE',
        # 'ABS_LOG_DEGREE_ENTROPY',
        # 'ABS_LOG_LINK_DENSITY',
        # 'ABS_LOG_NODE_LINK_RATIO',
        # 'ABS_LOG_MAX_CLUST_COEF',
        # 'ABS_LOG_MEAN_CLUST_COEF',
        # 'ABS_LOG_MAX_BTW',
        # 'ABS_LOG_MEAN_BTW',
        # 'ABS_LOG_BTW_ENTROPY',
        # 'ABS_LOG_BTW_ENTROPY_NORM',
        # 'ABS_LOG_DFT_ENTROPY',
        # 'ABS_LOG_CC_ENTROPY',
        # 'ABS_LOG_CC_ENTROPY_NORM',
        # 'ABS_LOG_DEGREE_ASSORT',
        # 'LOG_PERC_UNIQUE_SEQS',
        # 'ABS_MEAN_DEGREE',
        # 'ABS_MEAN_DEGREE_DIV',
        # 'ABS_NODE_LINK_RATIO',
        # 'ABS_NODE_LINK_RATIO_DIV',
        # 'ABS_LINK_DENSITY',
        # 'ABS_LINK_DENSITY_DIV',
        # 'ABS_MEAN_CLUST',
        # 'ABS_MEAN_CLUST_DIV',
        # 'ABS_MAX_BTW',
        # 'ABS_MAX_BTW_DIV',
        # 'ABS_MEAN_BTW',
        # 'ABS_MEAN_BTW_DIV',
        # 'ABS_BTW_ENTROPY',
        # 'ABS_BTW_ENTROPY_DIV',
        # 'ABS_BTW_ENTROPY_NORM',
        # 'ABS_BTW_ENTROPY_NORM_DIV',
        # 'ABS_DFT_ENTROPY',
        # 'ABS_DFT_ENTROPY_DIV',
        # 'ABS_CC_ENTROPY',
        # 'ABS_CC_ENTROPY_DIV',
        # 'ABS_CC_ENTROPY_NORM',
        # 'ABS_DEGREE_ASSORT',
        # 'ABS_DEGREE_ASSORT_DIV',
    ]
    df = pd.read_csv(my_path, sep='\t')
    df = df[['DISCOVERY_ALG'] + cols + [gt]]

    out_path = 'experiments/results/reports/pairplots/'

    df = df[(df[gt] >= 0)]
    # df = df[df['DISCOVERY_ALG'] == 'ETM']
   
    sns.set_context(rc={"axes.labelsize":15})
    
    for c in cols:
        df_pp = df[[gt, c]]
        sns.pairplot(df_pp)
        print(df_pp)
        plt.show(block=True)

