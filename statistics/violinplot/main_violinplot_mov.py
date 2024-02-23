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

    mov = 'MOV_PROTOCOLO_DE_PETICAO_118'
    df_feat = df_feat[['TEMPO_PROCESSUAL_TOTAL_DIAS', mov]]
    df_feat[mov] = df_feat[mov].astype('category')
    # df_feat['TEMPO_PROCESSUAL_TOTAL_DIAS'] = df_feat['TEMPO_PROCESSUAL_TOTAL_DIAS'].astype(float)

    sns.violinplot(data=df_feat, 
                   y='TEMPO_PROCESSUAL_TOTAL_DIAS',
                   x=mov,
                   cut=0
                  )
   
    plt.tight_layout()
    # plt.show()
    plt.savefig('dataset/tribunais_trabalho/statistics/corr_mov_viol.png')

    print()
