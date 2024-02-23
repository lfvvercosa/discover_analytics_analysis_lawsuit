import pandas as pd
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    out_path = 'dataset/tribunais_trabalho/statistics/corr_class.png'
    df_feat = pd.read_csv(input_path, sep='\t')
    vals = {}
    map_keys = {}
    count = 0

    # col = 'MOV_RECEBIMENTO_132'
    # newName = 'PM$_{Submission}$'

    # col = 'MOV_LIQUIDACAO_INICIADA_11384'
    # newName = 'PM$_{Liquidation}$'
    
    # col = 'NUMERO_TRT'
    # newName = 'Justice'

    # col = 'PROCESSO_DIGITAL'
    # newName = 'Digital'

    # col = 'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO'
    # newName = 'Subject'

    col = 'CLASSE_PROCESSUAL'
    newName = 'Class'

    df_feat = df_feat[df_feat[col] != 'CLA_OUTRO_CLASSE_PROCESSUAL']

    df_feat[col] = df_feat[col].map({
        'CLA_EXECUCAO_DE_CERTIDAO_DE_CREDITO_JUDICIAL_993': 'CL$_{2}$',
        'CLA_EXECUCAO_DE_TERMO_DE_AJUSTE_DE_CONDUTA_991': 'CL$_{3}$',
        'CLA_EXECUCAO_DE_TITULO_EXTRAJUDICIAL_990': 'CL$_{4}$',
        'CLA_EXECUCAO_FISCAL_1116': 'CL$_{5}$',
        'CLA_EXECUCAO_PROVISORIA_EM_AUTOS_SUPLEMENTARES_994': 'CL$_{1}$',
    })


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
    order = ['CL$_{1}$','CL$_{2}$','CL$_{3}$','CL$_{4}$','CL$_{5}$']

    sns.set(font_scale=1.9)
    # Create the boxplot
    sns.boxplot(
        x = newName,
        y = gtNewName,
        showmeans=True,  # Show means as diamonds
        data=df_feat,
        palette="Blues",
        order=order,
    )
    plt.xlabel(newName, labelpad=12)
    plt.ylabel(gtNewName, labelpad=12)
    plt.xticks(rotation=45)

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
