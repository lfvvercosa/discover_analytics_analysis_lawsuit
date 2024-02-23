import pandas as pd
import matplotlib.pyplot as plt


base_path = 'experiments/variant_analysis/exp7/results/size10/'
paths_metrics = [
    base_path + 'ARI_2step_kms_agglom.csv',
    base_path + 'Fitness_2step_kms_agglom.csv',
    base_path + 'Complexity_2step_kms_agglom.csv',

    # base_path + 'ARI_1step_ngram_kms.csv',
    # base_path + 'Fitness_1step_ngram_kms.csv',
    # base_path + 'Complexity_1step_ngram_kms.csv',
    
    # base_path + 'Fitness_gd.csv',
    # base_path + 'Complexity_gd.csv',

]

for path in paths_metrics:
    df = pd.read_csv(path, sep='\t')

    ax = plt.gca()


    plt.boxplot(df, labels=['low_complexity',
                            'medium_complexity',
                            'high_complexity'])

    if 'Fitness_' in path:
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    if 'Complexity_' in path:
        ax.set_ylim([1, 3000])
        plt.yscale('log')

    if 'ARI_' in path:
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    title = path[path.rfind('/') + 1 :]
    title = title[:title.find('_')]
    ax.set_title(title)

    # plt.show()
    name = base_path + path[path.rfind('/') + 1 : path.rfind('.')] + '.png'
    plt.savefig(name)
    plt.figure().clear()

print('done!')