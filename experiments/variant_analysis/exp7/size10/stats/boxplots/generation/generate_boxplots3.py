import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


base_path = 'experiments/variant_analysis/exp7/size10/results/'
out_path = 'experiments/variant_analysis/exp7/size10/stats/boxplots/images/ics_fitness/'
paths_metrics = [
    # base_path + 'ARI_1step_ngram_kms.csv',
    # base_path + 'Fitness_1step_ngram_kms.csv',
    base_path + 'Complexity_1step_ngram_kms.csv',
    # base_path + 'Fitness_gd.csv',
    # base_path + 'Complexity_gd.csv',
]

path_metrics_comp = [
    # base_path + 'ARI_2step_kms_agglom.csv',
    # base_path + 'Fitness_2step_kms_agglom.csv',
    base_path + 'Complexity_2step_kms_agglom.csv',
]

path_metrics_gd = [
    # base_path + 'ARI_2step_kms_agglom.csv',
    # base_path + 'Fitness_gd.csv',
    base_path + 'Complexity_gd.csv',
]

for i in range(len(paths_metrics)):
    df_1step = pd.read_csv(paths_metrics[i], sep='\t')
    df_2step = pd.read_csv(path_metrics_comp[i], sep='\t')
    df_gd = pd.read_csv(path_metrics_gd[i], sep='\t')

    bp_gd = plt.boxplot(df_gd, 
                      positions=np.array(range(len(df_gd.columns)))*3.0-0.8, 
                      sym='', 
                      widths=0.6)
    
    bp_1step = plt.boxplot(df_1step, 
                      positions=np.array(range(len(df_1step.columns)))*3.0+0, 
                      sym='', 
                      widths=0.6)
    
    bp_2step = plt.boxplot(df_2step, 
                      positions=np.array(range(len(df_2step.columns)))*3.0+0.8, 
                      sym='', 
                      widths=0.6)

    set_box_color(bp_gd, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bp_1step, '#49BE25')
    set_box_color(bp_2step, '#2C7BB6')

    ticks = list(df_1step.columns)
    
    ax = plt.gca()

    if 'Fitness_' in paths_metrics[i]:
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.ylim(0, 0.6)

    if 'Complexity_' in paths_metrics[i]:
        ax.set_ylim([1, 2500])
        plt.yscale('log')

    if 'ARI_' in paths_metrics[i]:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.plot([], c='#D7191C', label='Ground_Truth')
    plt.plot([], c='#49BE25', label='KMS_1Step')
    plt.plot([], c='#2C7BB6', label='KMS_2Step')
    plt.legend()

    title = paths_metrics[i][paths_metrics[i].rfind('/') + 1 :]
    title = title[:title.find('_')]
    ax.set_title(title)

    plt.xticks(range(0, len(ticks) * 3, 3), ticks)
    plt.xlim(-3, len(ticks)*3)
    
    plt.tight_layout()

    metric_name = paths_metrics[i][paths_metrics[i].rfind('/')+1:]
    metric_name = metric_name[:metric_name.find('_')]

    path_output = out_path + 'Comp_' + metric_name + '.png'
    
    plt.savefig(path_output)
    plt.figure().clear()
    
    # plt.show()


print('done!')