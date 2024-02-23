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

    col = 'MOV_RECEBIMENTO_132'
    newName = 'PM$_{1}$'
    savingName = 'corr_mov_recebimento_132.png'

    # col = 'MOV_LIQUIDACAO_INICIADA_11384'
    # newName = 'PM$_{2}$'
    # savingName = 'corr_mov_liquidacao_11384.png'
    
    # col = 'NUMERO_TRT'
    # newName = 'Justice'

    # col = 'PROCESSO_DIGITAL'
    # newName = 'Digital'

    # col = 'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO'
    # newName = 'Subject$_{1}$'

    # col = 'CLASSE_PROCESSUAL'
    # newName = 'Class'

    out_path = 'dataset/tribunais_trabalho/statistics/' + savingName
    gt = "TEMPO_PROCESSUAL_TOTAL_DIAS"
    gtNewName = 'Duration (days)'
    df_feat_sample = df_feat.sample(frac=0.005, random_state=42)

    sns.pairplot(data=df_feat_sample, x_vars=col, y_vars=gt)
    
    
    plt.xlabel(newName, labelpad=12)
    plt.ylabel(gtNewName, labelpad=12)

    plt.tight_layout()
    plt.show()
    # plt.savefig(out_path, dpi=400)
    

    print()
