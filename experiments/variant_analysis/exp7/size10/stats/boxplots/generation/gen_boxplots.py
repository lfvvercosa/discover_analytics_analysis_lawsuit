import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def gen(path_gd, 
        path_tech_1step, 
        path_tech_2steps,
        name_gd,
        name_tech,
        name_tech_2steps,
        out_path,
        out_preffix):

    df_gd = pd.read_csv(path_gd, sep='\t')
    df_1step = pd.read_csv(path_tech_1step, sep='\t')
    df_2step = pd.read_csv(path_tech_2steps, sep='\t')

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

    if 'Fitness_' in path_gd:
        plt.yticks([-2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, 
                    -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.ylim(-2, 1)

    if 'Complexity_' in path_gd:
        ax.set_ylim([1, 1500])
        plt.yscale('log')

    if 'ARI_' in path_gd:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.plot([], c='#D7191C', label=name_gd)
    plt.plot([], c='#49BE25', label=name_tech)
    plt.plot([], c='#2C7BB6', label=name_tech_2steps)
    plt.legend()

    title = path_gd[path_gd.rfind('/') + 1 :]
    title = title[:title.find('_')]
    ax.set_title(title)

    plt.xticks(range(0, len(ticks) * 3, 3), ticks)
    plt.xlim(-3, len(ticks)*3)
    
    plt.tight_layout()

    metric_name = path_gd[path_gd.rfind('/')+1:]
    metric_name = metric_name[:metric_name.find('_')]

    path_output = out_path + out_preffix + metric_name + '.png'
    
    plt.savefig(path_output)
    plt.figure().clear()


def gen2(path_tech_1step, 
        path_tech_2steps,
        name_tech,
        name_tech_2steps,
        out_path,
        out_preffix):
    
    df_1step = pd.read_csv(path_tech_1step, sep='\t')
    df_steps = pd.read_csv(path_tech_2steps, sep='\t')

    bp_1step = plt.boxplot(df_1step, 
                      positions=np.array(range(len(df_1step.columns)))*2.0-0.4, 
                      sym='', 
                      widths=0.6)
    
    bp_2steps = plt.boxplot(df_steps, 
                      positions=np.array(range(len(df_steps.columns)))*2.0+0.4, 
                      sym='', 
                      widths=0.6)

    set_box_color(bp_1step, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bp_2steps, '#2C7BB6')

    ticks = list(df_1step.columns)
    
    ax = plt.gca()

    if 'Fitness_' in path_tech_1step:
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    if 'Complexity_' in path_tech_1step:
        ax.set_ylim([1, 10000])
        plt.yscale('log')

    if 'ARI_' in path_tech_1step:
        plt.yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1])

    plt.plot([], c='#D7191C', label=name_tech)
    plt.plot([], c='#2C7BB6', label=name_tech_2steps)
    plt.legend()

    title = path_tech_1step[path_tech_1step.rfind('/') + 1 :]
    title = title[:title.find('_')]
    ax.set_title(title)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()

    metric_name = path_tech_1step[path_tech_1step.rfind('/')+1:]
    metric_name = metric_name[:metric_name.find('_')]

    path_output = out_path + out_preffix + metric_name + '.png'
    
    plt.savefig(path_output)
    plt.figure().clear()


if __name__ == '__main__':
    base_path = 'experiments/variant_analysis/exp7/size10/results/ics_fitness/'
    out_path = 'experiments/variant_analysis/exp7/size10/stats/boxplots/images/ics_fitness/'
    out_preffix = 'Comp_KMS_'

    # path_gd =  base_path + 'Complexity_gd.csv'
    # path_tech_1step = base_path + 'Complexity_ActiTraC_ics.csv'
    # path_tech_2steps = base_path + 'Complexity_ActiTraC_Agglom_ics.csv'

    # name_gd = 'Ground Truth'
    # name_tech = '1Step ActiTraC'
    # name_tech_2steps = '2Step ActiTraC + Agglom'

    # path_gd =  base_path + 'Complexity_gd.csv'
    # path_tech_1step = base_path + 'Complexity_1step_ngram_kms_ics.csv'
    # path_tech_2steps = base_path + 'Complexity2step_kms_agglom_ics.csv'

    # name_gd = 'Ground Truth'
    # name_tech = '1Step KMS'
    # name_tech_2steps = '2Step KMS + Agglom'

    # path_tech_1step = base_path + 'ARI_ActiTraC_ics.csv'
    # path_tech_2steps = base_path + 'ARI_ActiTraC_Agglom_ics.csv'

    # name_tech = '1Step ActiTraC'
    # name_tech_2steps = '2Step ActiTraC + Agglom'

    path_tech_1step = base_path + 'ARI_1step_ngram_kms_ics.csv'
    path_tech_2steps = base_path + 'ARI2step_kms_agglom_ics.csv'

    name_tech = '1Step KMS'
    name_tech_2steps = '2Step KMS + Agglom'

    # gen(path_gd, 
    #     path_tech_1step, 
    #     path_tech_2steps,
    #     name_gd,
    #     name_tech,
    #     name_tech_2steps,
    #     out_path,
    #     out_preffix)

    gen2(path_tech_1step, 
         path_tech_2steps,
         name_tech,
         name_tech_2steps,
         out_path,
         out_preffix)
    
    print('done!')