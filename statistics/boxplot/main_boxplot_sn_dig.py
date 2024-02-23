import pandas as pd
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    out_path = 'dataset/tribunais_trabalho/statistics/corr_digital.png'
    df_feat = pd.read_csv(input_path, sep='\t')
    vals = {}
    map_keys = {}
    count = 0

    regexp = re.compile('^ASSU_')
    assu_cols = [c for c in df_feat.columns if regexp.search(c)]

    # col = 'MOV_RECEBIMENTO_132'
    # newName = 'PM$_{Submission}$'

    # col = 'MOV_LIQUIDACAO_INICIADA_11384'
    # newName = 'PM$_{Liquidation}$'
    
    # col = 'NUMERO_TRT'
    # newName = 'Justice'

    col = 'PROCESSO_DIGITAL'
    newName = 'Digital'

    gt = "TEMPO_PROCESSUAL_TOTAL_DIAS"
    gtNewName = 'Duration (days)'

    df_feat = df_feat[df_feat['PROCESSO_DIGITAL'] != 'DIG_0.0']
    df_feat['PROCESSO_DIGITAL'] = df_feat['PROCESSO_DIGITAL'].map({'DIG_1.0':'Yes',
                                                                   'DIG_2.0':'No'})
    
    df_feat = df_feat.rename(columns={col:newName, gt:gtNewName})

    sns.set(font_scale=1.9)
    # Create the boxplot
    sns.boxplot(
        x = newName,
        y = gtNewName,
        showmeans=True,  # Show means as diamonds
        data=df_feat,
        palette="Blues",
        order=['No','Yes']
    )
    plt.xlabel(newName, labelpad=12)
    plt.ylabel(gtNewName, labelpad=12)

    # sns.swarmplot(
    #     x = col,
    #     y = "TEMPO_PROCESSUAL_TOTAL_DIAS",
    #     data=df_feat,
    #     # hue="smoker",
    #     color="gray", 
    #     alpha=0.5
    # )


    # plt.ylim(bottom=0, top=1)

    # medians = [b.get_ydata()[1] for b in bp['medians']]
    # positions = [(l.get_xdata()[0] + l.get_xdata()[1])/2 for l in bp['medians']]
    # plt.plot(positions,medians, color='grey', linewidth=2, linestyle='dotted')

    # plt.scatter(positions, medians, color='red', s=50)  # Add markers for emphasis

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_path, dpi=400)
    

    print()
