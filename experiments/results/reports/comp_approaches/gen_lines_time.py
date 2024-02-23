import matplotlib.pyplot as plt
import pandas as pd


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
    'TIME_ALIGN':'Align',
    'TIME_FREQ_50':'Freq 50%',
    'TIME_FREQ_25':'Freq 25%',
    'TIME_PROP':'Proposta',
    'TIME_BL':'Baseline',
})

df_etm = df[df['DISCOVERY_ALG'] == 'ETM']
df_imf = df[df['DISCOVERY_ALG'] == 'IMf']
df_imd = df[df['DISCOVERY_ALG'] == 'IMd']

df_etm = df_etm.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])
df_imf = df_imf.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])
df_imd = df_imd.drop(columns=['EVENT_LOG','DISCOVERY_ALG'])

df_imf = df_imf.sort_values(by=['Align'], ascending=False)
df_imf = df_imf.reset_index(drop=True)

df_imd = df_imd.sort_values(by=['Align'], ascending=False)
df_imd = df_imd.reset_index(drop=True)

df_etm = df_etm.sort_values(by=['Align'], ascending=False)
df_etm = df_etm.reset_index(drop=True)

df_imf.to_csv('temp/temp.csv', sep='\t')

lines = df_etm.plot.line(color=
    {
        'Align':'#CC0000',
        'Freq 50%':'#FF8000',
        'Freq 25%':'#FFFF00',
        'Proposta':'#00CC00',
        'Baseline':'#0080FF',
    })
# plt.yscale('log')
plt.title("Custo Computacional - ETM", fontdict={'fontsize':13, 'fontweight':'bold'}, pad=20)
plt.xlabel("#Processo", labelpad=15)
plt.ylabel("Tempo (s)", labelpad=15)
# plt.yscale("log")

plt.show()
