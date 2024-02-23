import pandas as pd
import scipy.stats as stats

if __name__ == "__main__":
    input_path = 'dataset/tribunais_trabalho/stats_dataset_model.csv'
    out_path = 'dataset/tribunais_trabalho/statistics/corr_trt.png'
    df_feat = pd.read_csv(input_path, sep='\t')

    df_temp = df_feat[['TEMPO_PROCESSUAL_TOTAL_DIAS','NUMERO_TRT']]

    groups = df_temp.groupby('NUMERO_TRT')
    num_groups = len(groups)

    for i, (group_name, group_data) in enumerate(groups):
        for j in range(0, num_groups):
            other_group_name = list(groups.groups.keys())[j]
            other_group_data = groups.get_group(other_group_name)

            _, p_value = stats.ks_2samp(group_data['TEMPO_PROCESSUAL_TOTAL_DIAS'], 
                                        other_group_data['TEMPO_PROCESSUAL_TOTAL_DIAS'],
                                        alternative='less')
            print(f"Kolmogorov-Smirnov test between group {group_name} and group {other_group_name}: p-value = {p_value}")

    

    
    