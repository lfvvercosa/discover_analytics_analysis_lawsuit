import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



approaches_path = 'experiments/results/reports/' + \
                  'comp_approaches/df_approaches.csv'

out_error_etm = 'experiments/results/reports/' + \
                  'comp_approaches/df_error_etm.csv'
out_error_imf = 'experiments/results/reports/' + \
                  'comp_approaches/df_error_imf.csv'
out_error_imd = 'experiments/results/reports/' + \
                  'comp_approaches/df_error_imd.csv'                  

df = pd.read_csv(approaches_path, sep='\t')
df = df[[
    'EVENT_LOG',
    'DISCOVERY_ALG',
    'ALIGN',
    'PROP2',
    'BL',
    'FREQ_50',
    'FREQ_25',
]]

df['Proposta'] = df['ALIGN'] - df['PROP2']
df['Baseline'] = df['ALIGN'] - df['BL']
df['Freq 50%'] = df['ALIGN'] - df['FREQ_50']
df['Freq 25%'] = df['ALIGN'] - df['FREQ_25']

df_box = df[[
    'Proposta',
    'Baseline',
    'Freq 50%',
    'Freq 25%',
]]
df_box = df_box.abs()

# figure(figsize=(4, 3), dpi=80)

# df_box.boxplot(grid=False)
# plt.xlabel("Abordagem", labelpad=15)
# plt.ylabel("Erro Absoluto (%)", labelpad=15)

# plt.show()

df_temp = df[[
    'EVENT_LOG',
    'DISCOVERY_ALG',
    'Proposta',
    'Baseline',
    'Freq 50%',
    'Freq 25%',
]]
df_temp['Proposta'] = df_temp['Proposta'].abs() + 0.0001
df_temp['Baseline'] = df_temp['Baseline'].abs() + 0.0001
df_temp['Freq 50%'] = df_temp['Freq 50%'].abs() + 0.0001
df_temp['Freq 25%'] = df_temp['Freq 25%'].abs() + 0.0001

df_etm = df_temp[df_temp['DISCOVERY_ALG'] == 'ETM']
df_imf = df_temp[df_temp['DISCOVERY_ALG'] == 'IMf']
df_imd = df_temp[df_temp['DISCOVERY_ALG'] == 'IMd']

df_etm.to_csv(out_error_etm, sep='\t', index=False)
df_imf.to_csv(out_error_imf, sep='\t', index=False)
df_imd.to_csv(out_error_imd, sep='\t', index=False)

df_etm = df_etm.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])
df_imf = df_imf.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])
df_imd = df_imd.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])

df_boxplot = df_etm
df_boxplot.boxplot(grid=False)
# max_val = df_boxplot.select_dtypes(include=[np.number]).max().max()

plt.title("ETM", fontdict={'fontsize':13, 'fontweight':'bold'}, pad=20)
plt.xlabel("Abordagem", labelpad=15)
plt.yscale("log")
plt.ylabel("Log do Erro Absoluto (%)", labelpad=15)

ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_yticks([0.0001,0.001,0.01,0.1,1])
ax.set_yticklabels(["0", 
                    f'10$^{{-3}}$', 
                    f'10$^{{-2}}$',
                    f'10$^{{-1}}$',
                    1])

plt.show()