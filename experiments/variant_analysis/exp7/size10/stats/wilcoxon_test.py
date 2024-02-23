import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

base_path = 'experiments/variant_analysis/exp7/size10/results/'
out_path = 'experiments/variant_analysis/exp7/size10/boxplots/images/'


def gen_ranking(data, tech):

    rank = {}
    
    for idx in range(len(data)):
        pos = 1

        for idx2 in range(len(data)):
            if idx != idx2:
                res_wilc = wilcoxon(data[idx], 
                                    data[idx2],
                                    alternative='greater')
                if res_wilc.pvalue < 0.05:
                    pos += 1
        
        if pos not in rank:
            rank[pos] = []
        
        rank[pos].append(tech[idx])


    return rank


paths_metrics = [
    # base_path + 'ARI_1step_ngram_kms.csv',
    # base_path + 'Fitness_1step_ngram_kms.csv',
    base_path + 'Complexity_1step_ngram_kms.csv',
]

path_metrics_2step = [
    # base_path + 'ARI_2step_kms_agglom.csv',
    # base_path + 'Fitness_2step_kms_agglom.csv',
    base_path + 'Complexity_2step_kms_agglom.csv',
]

complexity = [
    'low_complexity',
    'medium_complexity',
    'high_complexity',
]

path_metrics_gd = [
    # base_path + 'ARI_2step_kms_agglom.csv',
    # base_path + 'Fitness_gd.csv',
    base_path + 'Complexity_gd.csv',
]

df_1step = pd.read_csv(paths_metrics[0], sep='\t')
df_2step = pd.read_csv(path_metrics_2step[0], sep='\t')
df_gd = pd.read_csv(path_metrics_gd[0], sep='\t')

for c in complexity:
    print()
    print('#### ' + c + ' ####')
    print()

    data = [df_1step[c], df_2step[c], df_gd[c]]
    tech = ['1step_kms', '2step_kms', 'gd']

    rank = gen_ranking(data, tech)

    for pos in range(1,len(tech)+1):
        if pos in rank:
            print('position: ' + str(pos))
            print(rank[pos])
            print()

# print(wilcoxon(df_1step['low_complexity'], 
#                df_2step['low_complexity'],
#                alternative='less'))

print()

# for i in range(len(paths_metrics)):
#     df_1step = pd.read_csv(paths_metrics[i], sep='\t')
#     df_2step = pd.read_csv(path_metrics_comp[i], sep='\t')
#     df_gd = pd.read_csv(path_metrics_gd[i], sep='\t')