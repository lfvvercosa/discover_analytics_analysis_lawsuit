import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



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

df_etm.boxplot(grid=False)

plt.title("ETM", fontdict={'fontsize':13, 'fontweight':'bold'}, pad=20)
plt.xlabel("Abordagem", labelpad=15)
plt.ylabel("Tempo (s)", labelpad=15)
plt.yscale("log")
plt.show()