import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/mini/stats_dataset_trt_model.csv'
    df_feat = pd.read_csv(input_path, sep='\t')
    df_feat = df_feat[['TEMPO_PROCESSUAL_TOTAL_DIAS','CLASSE_PROCESSUAL']]

    df_temp = df_feat[['CLASSE_PROCESSUAL']].drop_duplicates()
    df_temp['NUMBER'] = np.arange(len(df_temp.index))
    df_temp['NUMBER'] = 'CLA_' + df_temp['NUMBER'].astype(str)
    df_temp = df_temp.set_index('CLASSE_PROCESSUAL')
    df_temp.to_dict()

    df_feat['CLASSE_PROCESSUAL'] = df_feat['CLASSE_PROCESSUAL'].map(df_temp.to_dict()['NUMBER'])

    print('mapping: ' + str(df_temp.to_dict()['NUMBER']))

    # Choose order
    df_temp = df_feat.groupby('CLASSE_PROCESSUAL').agg(max=('TEMPO_PROCESSUAL_TOTAL_DIAS','max'),
                                                       min=('TEMPO_PROCESSUAL_TOTAL_DIAS','min'))
    df_temp['RANGE'] = df_temp['max'] - df_temp['min']
    df_temp = df_temp.sort_values('RANGE')
    order = df_temp.index.to_list()

    # Create the boxplot
    sns.boxplot(
        x = "CLASSE_PROCESSUAL",
        y = "TEMPO_PROCESSUAL_TOTAL_DIAS",
        showmeans=True,  # Show means as diamonds
        data=df_feat,
        order=order
    )


    # Customize the plot
    plt.title("Boxplot of Tempo Processual by Classe")
    plt.xlabel("Classe")
    plt.ylabel("Tempo Processual")
    plt.xticks(rotation=45)  # Rotate category labels for better readability
    plt.tight_layout()
    # plt.show()
    plt.savefig('dataset/tribunais_trabalho/mini/results/corr_cla.png')
    print()
