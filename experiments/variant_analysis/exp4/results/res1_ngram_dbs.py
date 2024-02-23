import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('experiments/variant_analysis/exp4/results/' + \
                 'results_1step_ngram_dbs_complex.csv', sep='\t')

df = df.sort_values('eps')
df = df[df['n'] == 1]
df.corr().round(4).to_csv('temp/df_corr.csv', sep='\t')
x_axis = list(range(len(df)))
y_axis = df['Fit'].to_list()
y2_axis = df['eps'].to_list()

plt.plot(x_axis, y_axis)
plt.plot(x_axis, y2_axis, '-.')
plt.show(block=True)

print()

