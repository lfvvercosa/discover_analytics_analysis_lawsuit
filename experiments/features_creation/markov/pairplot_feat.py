import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


path = 'experiments/features_creation/feat_markov/feat_markov_k_all_min.csv'


df = pd.read_csv(path, sep='\t')

df['ERROR_W3'] = df['FITNESS'] - df['ABS_EDGES_ONLY_IN_LOG_W3']

df_pp = df[['ERROR_W3','MIN_COST']]

sns.set_context(rc={"axes.labelsize":15})
sns.pairplot(df_pp)

plt.show(block=True)
