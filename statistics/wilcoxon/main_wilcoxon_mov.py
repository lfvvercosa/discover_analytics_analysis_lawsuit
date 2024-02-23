import pandas as pd
import scipy.stats as stats

if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    df_feat = pd.read_csv(input_path, sep='\t')

    col = 'CLASSE_PROCESSUAL'
    # col = 'MOV_RECEBIMENTO_132'
    # col = 'MOV_LIQUIDACAO_INICIADA_11384'
    # col = 'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO'
    df_temp = df_feat[['TEMPO_PROCESSUAL_TOTAL_DIAS',col]]

    groups = df_temp.groupby(col)
    num_groups = len(groups)

    for i, (group_name, group_data) in enumerate(groups):
        for j in range(0, num_groups):
            other_group_name = list(groups.groups.keys())[j]
            other_group_data = groups.get_group(other_group_name)

            _, p_value = stats.ks_2samp(group_data['TEMPO_PROCESSUAL_TOTAL_DIAS'], 
                                        other_group_data['TEMPO_PROCESSUAL_TOTAL_DIAS'],
                                        alternative='greater')
            print(f"Kolmogorov-Smirnov test between group {group_name} and group {other_group_name}: p-value = {p_value}")

    

    
    