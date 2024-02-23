import pandas as pd
import matplotlib.pyplot as plt


my_path = 'experiments/results/reports/size_check/size_check.csv'
df = pd.read_csv(my_path, sep='\t')

df_imf = df[df['DISCOVERY_ALG'] == 'IMf']
df_imd = df[df['DISCOVERY_ALG'] == 'IMd']
df_etm = df[df['DISCOVERY_ALG'] == 'ETM']

df_etm = df_etm[df_etm['PERCENT_ACT'] <= 1]

# df_imf['PERCENT_MODEL_REPEAT_ACT'] = df_imf['PERCENT_MODEL_REPEAT_ACT'] * 100
# df_imd['PERCENT_MODEL_REPEAT_ACT'] = df_imd['PERCENT_MODEL_REPEAT_ACT'] * 100
# df_etm['PERCENT_MODEL_REPEAT_ACT'] = df_etm['PERCENT_MODEL_REPEAT_ACT'] * 100

plt.hist(df_imf['PERCENT_ACT'])
plt.title('IMf - Atividades Presentes no Modelo (%)')
plt.savefig('experiments/results/reports/size_check/imf_perc_act.png')
plt.figure().clear()
# plt.show(block=True)

plt.hist(df_imd['PERCENT_ACT'])
plt.title('IMd - Atividades Presentes no Modelo (%)')
plt.savefig('experiments/results/reports/size_check/imd_perc_act.png')
plt.figure().clear()
# plt.show(block=True)

plt.hist(df_etm['PERCENT_ACT'])
plt.title('ETM - Atividades Presentes no Modelo (%)')
plt.savefig('experiments/results/reports/size_check/etm_perc_act.png')
plt.figure().clear()
# plt.show(block=True)

# plt.hist(df_imf['PERCENT_MODEL_REPEAT_ACT'])
# plt.title('IMf - Atividades Repetidas no Modelo (*)')
# plt.savefig('experiments/results/reports/size_check/imf_ativ_rep.png')
# plt.figure().clear()
# # plt.show(block=True)

# plt.hist(df_imd['PERCENT_MODEL_REPEAT_ACT'])
# plt.title('IMd - Atividades Repetidas no Modelo (*)')
# plt.savefig('experiments/results/reports/size_check/imd_ativ_rep.png')
# plt.figure().clear()
# # plt.show(block=True)

# plt.hist(df_etm['PERCENT_MODEL_REPEAT_ACT'])
# plt.title('ETM - Atividades Repetidas no Modelo (*)')
# plt.savefig('experiments/results/reports/size_check/etm_ativ_rep.png')
# # plt.show(block=True)

print()
# df_imf['PERCENT_ACT']