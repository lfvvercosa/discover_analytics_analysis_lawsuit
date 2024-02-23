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
    'TIME_ALIGN',
    'TIME_FREQ_50',
    'TIME_FREQ_25',
    'TIME_PROP',
    'TIME_BL',
]]

df = df.rename(columns={
    'TIME_PROP':'Proposta',
    'TIME_BL':'Baseline',
    'TIME_FREQ_50':'Freq 50%',
    'TIME_FREQ_25':'Freq 25%',
    'TIME_ALIGN':'Alinhamento',
})

df = df[[
    'EVENT_LOG',
    'DISCOVERY_ALG',
    'Proposta',
    'Baseline',
    'Freq 50%',
    'Freq 25%',
    'Alinhamento']]

# print(df.groupby('DISCOVERY_ALG').agg(
#     {'TIME_FREQ_50':'mean',
#      'TIME_PROP':'mean',
#      'TIME_BL':'mean'}
# ))

df_etm = df[df['DISCOVERY_ALG'] == 'ETM']
df_imf = df[df['DISCOVERY_ALG'] == 'IMf']
df_imd = df[df['DISCOVERY_ALG'] == 'IMd']

df_etm = df_etm.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])
df_imf = df_imf.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])
df_imd = df_imd.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])

print('### ETM ###')
print()

prop = df_etm['Proposta']
freq_50 = df_etm['Freq 50%']
freq_25 = df_etm['Freq 25%']
bl = df_etm['Baseline']
align = df_etm['Alinhamento']

print('### BL, Prop ###')
print(wilcoxon(bl, prop, alternative='less'))
print()

print('### Prop, Freq 25% ###')
print(wilcoxon(prop, freq_25, alternative='less'))
print()

print('### Freq 25%, Freq 50% ###')
print(wilcoxon(freq_25, freq_50, alternative='less'))
print()

print('### Freq 50%, Alinhamento ###')
print(wilcoxon(freq_50, align, alternative='less'))
print()


print('### IMf ###')
print()

prop = df_imf['Proposta']
freq_50 = df_imf['Freq 50%']
freq_25 = df_imf['Freq 25%']
bl = df_imf['Baseline']
align = df_imf['Alinhamento']

print('### BL, Prop ###')
print(wilcoxon(bl, prop, alternative='less'))
print()

print('### Prop, Freq 25% ###')
print(wilcoxon(prop, freq_25, alternative='less'))
print()

print('### Freq 25%, Freq 50% ###')
print(wilcoxon(freq_25, freq_50, alternative='less'))
print()

print('### Freq 50%, Alinhamento ###')
print(wilcoxon(freq_50, align, alternative='less'))
print()

print('### IMd ###')
print()

prop = df_imd['Proposta']
freq_50 = df_imd['Freq 50%']
freq_25 = df_imd['Freq 25%']
bl = df_imd['Baseline']
align = df_imd['Alinhamento']

print('### BL, Prop ###')
print(wilcoxon(bl, prop, alternative='less'))
print()

print('### Prop, Freq 25% ###')
print(wilcoxon(prop, freq_25, alternative='less'))
print()

print('### Freq 25%, Freq 50% ###')
print(wilcoxon(freq_25, freq_50, alternative='less'))
print()

print('### Freq 50%, Alinhamento ###')
print(wilcoxon(freq_50, align, alternative='less'))
print()

print()