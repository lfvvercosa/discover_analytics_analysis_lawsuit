# importing package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


approaches_path = 'experiments/results/reports/' + \
                  'comp_approaches/df_approaches.csv'

df = pd.read_csv(approaches_path, sep='\t')
df = df[[
    'EVENT_LOG',
    'DISCOVERY_ALG',
    'TIME_ALIGN',
    'TIME_PROP',
    'TIME_BL',
    'TIME_FREQ_50',
    'TIME_FREQ_25',
]]

print('total records: ' + str(len(df.index)))
print('IMf: ' + str(len(df[df['DISCOVERY_ALG'] == 'IMf'].index)))
print('IMd: ' + str(len(df[df['DISCOVERY_ALG'] == 'IMd'].index)))
print('ETM: ' + str(len(df[df['DISCOVERY_ALG'] == 'ETM'].index)))


df_time = df.groupby('DISCOVERY_ALG').agg(
            TIME_ALIGN=('TIME_ALIGN','mean'),
            TIME_PROP=('TIME_PROP','mean'),
            TIME_BL=('TIME_BL','mean'),
            TIME_FREQ_50=('TIME_FREQ_50','mean'),
            TIME_FREQ_25=('TIME_FREQ_25','mean'),
          )

df_time = df_time.round(2)

x = np.arange(3)    
y_align = df_time['TIME_ALIGN'].tolist()
y_prop = df_time['TIME_PROP'].tolist()
y_bl = df_time['TIME_BL'].tolist()
y_freq_50 = df_time['TIME_FREQ_50'].tolist()
y_freq_25 = df_time['TIME_FREQ_25'].tolist()

width = 0.15

# plot data in grouped manner of bar type
plt.bar(x-0.15, y_align, width, color='#CC0000')
plt.bar(x, y_freq_50, width, color='#FF8000')
plt.bar(x+0.15, y_freq_25, width, color='#FFFF00')
plt.bar(x+0.3, y_prop, width, color='#00CC00')
plt.bar(x+0.45, y_bl, width, color='#0080FF')

plt.title("Custo Computacional", fontdict={'fontsize':13, 'fontweight':'bold'}, pad=20)
plt.xticks(x, ['ETM', 'IMd', 'IMf'])
plt.xlabel("Grupo Petri-nets", labelpad=15)
plt.ylabel("Tempo (s)", labelpad=15)
plt.legend(["Align", "Freq 50%", "Freq 25%", "Proposta", "Baseline"])
plt.show()





