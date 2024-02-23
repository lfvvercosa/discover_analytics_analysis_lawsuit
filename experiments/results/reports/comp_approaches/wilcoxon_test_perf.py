import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.stats import wilcoxon
from scipy.stats import ks_2samp


approaches_path = 'experiments/results/reports/' + \
                  'comp_approaches/df_approaches.csv'

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
df_temp['Proposta'] = df_temp['Proposta'].abs()
df_temp['Baseline'] = df_temp['Baseline'].abs()
df_temp['Freq 50%'] = df_temp['Freq 50%'].abs()
df_temp['Freq 25%'] = df_temp['Freq 25%'].abs()

df_etm = df_temp[df_temp['DISCOVERY_ALG'] == 'ETM']
df_imf = df_temp[df_temp['DISCOVERY_ALG'] == 'IMf']
df_imd = df_temp[df_temp['DISCOVERY_ALG'] == 'IMd']

print('### ETM ###')
print()

prop = df_etm['Proposta']
freq_50 = df_etm['Freq 50%']
freq_25 = df_etm['Freq 25%']
bl = df_etm['Baseline']

print('### Prop, Freq 50% ###')
print(wilcoxon(prop, freq_50, alternative='less'))
print()

print('### Freq 50%, Freq 25% ###')
print(wilcoxon(freq_50, freq_25, alternative='less'))
print()

print('### Freq 25%, BL ###')
print(wilcoxon(freq_25, bl, alternative='less'))
print()

print('### IMf ###')
print()

prop = df_imf['Proposta']
freq_50 = df_imf['Freq 50%']
freq_25 = df_imf['Freq 25%']
bl = df_imf['Baseline']

print('### Prop, Freq 50% ###')
print(wilcoxon(prop, freq_50, alternative='less'))
print()

print('### Freq 50%, Freq 25% ###')
print(wilcoxon(freq_50, freq_25, alternative='less'))
print()

print('### BL, Freq 50% ###')
print(wilcoxon(bl, freq_50, alternative='less'))
print()

print('### Freq 25%, BL ###')
print(wilcoxon(freq_25, bl, alternative='less'))
print()

print('### IMd ###')
print()

prop = df_imd['Proposta']
freq_50 = df_imd['Freq 50%']
freq_25 = df_imd['Freq 25%']
bl = df_imd['Baseline']

print('### Prop, Freq 50% ###')
print(wilcoxon(prop, freq_50, alternative='less'))
print()

print('### Freq 50%, Freq 25% ###')
print(wilcoxon(freq_50, freq_25, alternative='less'))
print()

print('### Freq 25%, BL ###')
print(wilcoxon(freq_25, bl, alternative='less'))
print()

print()