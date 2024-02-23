import pandas as pd
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/mini/stats_dataset_trt_model.csv'
    df_feat = pd.read_csv(input_path, sep='\t')
    vals = {}
    medians = {}
    map_keys = {}
    count = 0

    # col = 'MOV_RECEBIMENTO_132'
    col = 'CLUS_KME'
    
    occur = df_feat[col].drop_duplicates().to_list()
    occur.sort()

    for c in occur:
        vals[c] = df_feat[df_feat[col] == c]['TEMPO_PROCESSUAL_TOTAL_DIAS'].to_list()
        medians[c] = stat.median(vals[c])
    
    medians = {k: v for k, v in sorted(medians.items(), key=lambda item: item[1])}
    labels = []
    values = []

    for k in medians:
        map_keys[count] = vals[k]
        count += 1

    for k in map_keys:
        labels.append(k)
        values.append(map_keys[k])

    # fig, ax = plt.subplots()
    # ax.plot(occur,medians, color='red', linewidth=2)
   

    # Add a title
    plt.title("Time by Recebimento_132")

    # df = pd.DataFrame(vals_mapped)

    # Change colors and fill boxes
    bp = plt.boxplot(values, 
                     labels=labels,
                     patch_artist=True, 
                     boxprops=dict(facecolor='lightblue'), 
                     medianprops=dict(linewidth=2, color='red'),
                     notch=True,
                    )
    # plt.ylim(bottom=0, top=1)

    medians = [b.get_ydata()[1] for b in bp['medians']]
    positions = [(l.get_xdata()[0] + l.get_xdata()[1])/2 for l in bp['medians']]
    # positions = [b.get_x() + b.get_width() / 2 for b in bp['boxes']]

    
    # plt.plot(positions,medians, color='red', linewidth=2)

    # plt.scatter(positions, medians, color='red', s=50)  # Add markers for emphasis

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('dataset/tribunais_trabalho/mini/results/corr_clus.png')


    print()
