import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


base_path = 'experiments/variant_analysis/exp7/size10/results/'
out_path = 'experiments/variant_analysis/exp7/size10/stats/boxplots/images/'
paths_metrics = [
    # base_path + 'ARI_1step_ngram_kms_5clus.csv',
    base_path + 'Fitness_1step_ngram_kms_5clus.csv',
    # base_path + 'Complexity_1step_ngram_kms.csv',
    # base_path + 'Fitness_gd.csv',
    # base_path + 'Complexity_gd.csv',
]

path_metrics_comp = [
    # base_path + 'ARI_2step_kms_agglom_5clus.csv',
    base_path + 'Fitness_2step_kms_agglom_5clus.csv',
    # base_path + 'Complexity_2step_kms_agglom.csv',
]

for i in range(len(paths_metrics)):
    df = pd.read_csv(paths_metrics[i], sep='\t')
    df_comp = pd.read_csv(path_metrics_comp[i], sep='\t')

    bpl = plt.boxplot(df, 
                      positions=np.array(range(len(df.columns)))*2.0-0.4, 
                      sym='', 
                      widths=0.6)
    
    bpr = plt.boxplot(df_comp, 
                      positions=np.array(range(len(df_comp.columns)))*2.0+0.4, 
                      sym='', 
                      widths=0.6)

    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    ticks = list(df.columns)
    
    ax = plt.gca()

    if 'Fitness_' in paths_metrics[i]:
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    if 'Complexity_' in paths_metrics[i]:
        ax.set_ylim([1, 10000])
        plt.yscale('log')

    if 'ARI_' in paths_metrics[i]:
        plt.yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1])

    plt.plot([], c='#D7191C', label='KMS_1Step')
    plt.plot([], c='#2C7BB6', label='KMS_2Step')
    plt.legend()

    title = paths_metrics[i][paths_metrics[i].rfind('/') + 1 :]
    title = title[:title.find('_')]
    ax.set_title(title)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()

    metric_name = paths_metrics[i][paths_metrics[i].rfind('/')+1:]
    metric_name = metric_name[:metric_name.find('_')]

    path_output = out_path + 'Comp_' + metric_name + '_5clus.png'
    
    plt.savefig(path_output)
    plt.figure().clear()
    
    # plt.show()


print('done!')