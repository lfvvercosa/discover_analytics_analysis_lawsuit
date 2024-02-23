import pandas as pd
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

    for c in assu_cols:
        vals[c] = df_feat[df_feat[c] == 1]['TEMPO_PROCESSUAL_TOTAL_DIAS'].to_list()

    for k in vals:
        map_keys[k] = 'ASSU_' + str(count)
        count += 1
    
    vals_mapped = {map_keys[k]:v for k,v in vals.items()}
    

    my_range = {}

    for k in vals_mapped:
        my_min = min(vals_mapped[k])
        my_max = max(vals_mapped[k])
        my_range[k] = my_max - my_min

    print('keys mapping: ' + str(map_keys))
    sorted_keys = sorted(my_range, key=my_range.get)

    labels = []
    values = []

    for k in sorted_keys:
        labels.append(k)
        values.append(vals_mapped[k])

    # Add a title
    plt.title("Time by Assunto")

    # df = pd.DataFrame(vals_mapped)

    # Change colors and fill boxes
    plt.boxplot(values, 
                labels=labels,
                patch_artist=True, 
                boxprops=dict(facecolor='lightblue'), 
                medianprops=dict(linewidth=2, color='red'),
               )
    # plt.ylim(bottom=0, top=1)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('dataset/tribunais_trabalho/mini/results/corr_assu.png')
