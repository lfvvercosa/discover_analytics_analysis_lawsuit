import pandas as pd
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    df_feat = pd.read_csv(input_path, sep='\t')
    vals = {}
    map_keys = {}
    count = 0

    # col = 'MOV_RECEBIMENTO_132'
    # newName = 'PM$_{1}$'
    # savingName = 'corr_mov_recebimento_132.png'

    # col = 'MOV_LIQUIDACAO_INICIADA_11384'
    # newName = 'PM$_{2}$'
    # savingName = 'corr_mov_liquidacao_11384.png'

    
    # col = 'NUMERO_TRT'
    # newName = 'Justice'

    # col = 'PROCESSO_DIGITAL'
    # newName = 'Digital'

    col = 'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO'
    newName = 'Subject$_{1}$'
    savingName = 'corr_subject'

    # col = 'CLASSE_PROCESSUAL'
    # newName = 'Class'

    out_path = 'dataset/tribunais_trabalho/statistics/' + savingName
    gt = "TEMPO_PROCESSUAL_TOTAL_DIAS"
    gtNewName = 'Time (days)'
    
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
    # plt.title("Time by Recebimento_132")

    # df = pd.DataFrame(vals_mapped)

    # Change colors and fill boxes
    # bp = plt.boxplot(values, 
    #                  labels=labels,
    #                  patch_artist=True, 
    #                  boxprops=dict(facecolor='lightblue'), 
    #                  medianprops=dict(linewidth=2, color='red'),
    #                  notch=True,
    #                 )col
    
    df_feat = df_feat.rename(columns={col:newName, gt:gtNewName})

    sns.set(font_scale=1.9)
    # Create the boxplot
    sns.boxplot(
        x = newName,
        y = gtNewName,
        showmeans=True,  # Show means as diamonds
        data=df_feat,
        palette="Blues"
        # color="#3B658E",
        # 
        # order=order
    )
    plt.xlabel(newName, labelpad=12)
    plt.ylabel(gtNewName, labelpad=12)

    

  

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_path, dpi=400)
    

    print()
