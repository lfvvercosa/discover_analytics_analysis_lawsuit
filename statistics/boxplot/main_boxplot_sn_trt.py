import pandas as pd
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re


if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    out_path = 'dataset/tribunais_trabalho/statistics/corr_trt.png'
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
    
    col = 'NUMERO_TRT'
    newName = 'Justice'


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
    
    # Choose order
    df_temp = df_feat.groupby('NUMERO_TRT').agg(
                                                max=(gt,'max'),
                                                min=(gt,'min'),
                                                median=(gt,'median')
                                               )
    # df_temp['SORT_COL'] = df_temp['max'] - df_temp['min']
    df_temp = df_temp.rename(columns={'median':'SORT_COL'})
    df_temp = df_temp.sort_values('SORT_COL')
    order = df_temp.index.to_list()
    
    df_feat = df_feat.rename(columns={col:newName, gt:gtNewName})

    # sns.set(rc={"figure.figsize":(12, 8)})
    # sns.set(font_scale=1.3)
    # Create the boxplot
    ax = sns.boxplot(
        x = newName,
        y = gtNewName,
        showmeans=True,  # Show means as diamonds
        data=df_feat,
        palette="Blues",
        order=order
    )
    ax.set_ylabel(gtNewName, fontsize=14, labelpad=14)
    ax.set_xlabel(newName, fontsize=14, labelpad=14)
    # plt.xlabel(newName, labelpad=14)
    # plt.ylabel(gtNewName, labelpad=14)
    plt.xticks(rotation=90)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_path, dpi=400)
    

    print('done!')
