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
    map_keys = {}
    count = 0

    regexp = re.compile('^ASSU_')
    assu_cols = [c for c in df_feat.columns if regexp.search(c)]

    col = 'MOV_RECEBIMENTO_132'
    # col = 'MOV_LIQUIDACAO_INICIADA_11384'
    
    occur = df_feat[col].drop_duplicates().to_list()
    occur.sort()

    for c in occur:
        vals[c] = df_feat[df_feat[col] == c]['TEMPO_PROCESSUAL_TOTAL_DIAS'].to_list()

    
    labels = []
    values = []

    for k in vals:
        labels.append(k)
        values.append(vals[k])

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

    sns.boxplot(
        x = col,
        y = "TEMPO_PROCESSUAL_TOTAL_DIAS",
        showmeans=True,  # Show means as diamonds
        data=df_feat,
        order=order
    )



    medians = [b.get_ydata()[1] for b in bp['medians']]
    positions = [(l.get_xdata()[0] + l.get_xdata()[1])/2 for l in bp['medians']]
    # positions = [b.get_x() + b.get_width() / 2 for b in bp['boxes']]

    
    plt.plot(positions,medians, color='grey', linewidth=2, linestyle='dotted')

    # plt.scatter(positions, medians, color='red', s=50)  # Add markers for emphasis

    # Show the plot
    plt.tight_layout()
    plt.show()
    # plt.savefig('dataset/tribunais_trabalho/mini/results/corr_mov.png')
    

    print()
