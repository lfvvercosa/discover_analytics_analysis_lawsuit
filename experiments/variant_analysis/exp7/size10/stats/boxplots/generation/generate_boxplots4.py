import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


base_path = 'experiments/variant_analysis/exp7/size10/results/ics_fitness/'
out_path = 'experiments/variant_analysis/exp7/size10/stats/boxplots/images/'
paths_metrics = [
    # base_path + 'ARI_1step_ngram_kms.csv',
    # base_path + 'Fitness_1step_ngram_kms_ics.csv',
    base_path + 'Complexity_1step_ngram_kms_ics.csv',
]

path_metrics_comp = [
    # base_path + 'ARI_2step_kms_agglom.csv',
    # base_path + 'Fitness_LevWeight_ics.csv',
    base_path + 'Complexity_LevWeight_ics.csv',
]

path_metrics_gd = [
    # base_path + 'ARI_2step_kms_agglom.csv',
    # base_path + 'Fitness_gd.csv',
    base_path + 'Complexity_gd.csv',
]

path_metrics_act = [
    base_path + 'Complexity_ActiTraC_ics.csv',
    # base_path + 'Fitness_ActiTraC_ics.csv',
]

for i in range(len(paths_metrics)):
    df_1step = pd.read_csv(paths_metrics[i], sep='\t')
    df_levw = pd.read_csv(path_metrics_comp[i], sep=',')
    df_gd = pd.read_csv(path_metrics_gd[i], sep='\t')
    df_act = pd.read_csv(path_metrics_act[i], sep='\t')

    bp_gd = plt.boxplot(df_gd, 
                      positions=np.array(range(len(df_gd.columns)))*4.0 -1, 
                      sym='', 
                      widths=0.5)
    
    bp_1step = plt.boxplot(df_1step, 
                      positions=np.array(range(len(df_1step.columns)))*4.0-0.4, 
                      sym='', 
                      widths=0.5)
    
    bp_levw = plt.boxplot(df_levw, 
                      positions=np.array(range(len(df_levw.columns)))*4.0+0.4, 
                      sym='', 
                      widths=0.5)

    bp_act = plt.boxplot(df_act, 
                         positions=np.array(range(len(df_levw.columns)))*4.0+1, 
                         sym='', 
                         widths=0.5)

    set_box_color(bp_gd, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bp_1step, '#49BE25')
    set_box_color(bp_levw, '#2C7BB6')
    set_box_color(bp_act, '#2A7C06')

    ticks = list(df_1step.columns)
    
    ax = plt.gca()

    if 'Fitness_' in paths_metrics[i]:
        plt.yticks([-1,-0.8,-0.6,-0.4,-0.2,0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.ylim(-1, 1)

    if 'Complexity_' in paths_metrics[i]:
        ax.set_ylim([1, 1500])
        plt.yscale('log')

    if 'ARI_' in paths_metrics[i]:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.plot([], c='#D7191C', label='Ground_Truth')
    plt.plot([], c='#49BE25', label='KMS_1Step')
    plt.plot([], c='#2C7BB6', label='LevWeight')
    plt.plot([], c='#2A7C06', label='ActiTraC')
    plt.legend()

    title = paths_metrics[i][paths_metrics[i].rfind('/') + 1 :]
    title = title[:title.find('_')]
    ax.set_title(title)

    plt.xticks(range(0, len(ticks) * 4, 4), ticks)
    plt.xlim(-4, len(ticks)*4)
    
    plt.tight_layout()

    metric_name = paths_metrics[i][paths_metrics[i].rfind('/')+1:]
    metric_name = metric_name[:metric_name.find('_')]

    path_output = out_path + 'Comp3_lev_' + metric_name + '.png'
    
    plt.savefig(path_output)
    plt.figure().clear()
    
    # plt.show()


print('done!')