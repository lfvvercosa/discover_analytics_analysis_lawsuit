import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    out_path = 'dataset/tribunais_trabalho/statistics/corr_trt.png'
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    df_feat = pd.read_csv(input_path, sep='\t')
    df_feat = df_feat[[gt,'NUMERO_TRT']]

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

    # Create the boxplot
    sns.boxplot(
        x = "NUMERO_TRT",
        y = "TEMPO_PROCESSUAL_TOTAL_DIAS",
        showmeans=True,  # Show means as diamonds
        data=df_feat,
        order=order
    )

    # Customize the plot
    plt.title("Boxplot of Tempo Processual by TRT")
    plt.xlabel("TRT")
    plt.ylabel("Tempo Processual")
    plt.xticks(rotation=45)  # Rotate category labels for better readability
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_path)


    print()
