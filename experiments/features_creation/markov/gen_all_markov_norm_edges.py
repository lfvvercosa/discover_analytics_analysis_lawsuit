import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_rows', None)

path = 'experiments/features_creation/feat_markov/feat_markov_k_all_min.csv'

df = pd.read_csv(path, sep='\t')

df['MIN_COST_NORM'] = (df['MIN_COST'] // 10000) + 1

df['FIT_W1'] = 1 - (1 - df['ABS_EDGES_ONLY_IN_LOG_W1'])/(1 + df['MIN_COST_NORM']/10)
df['FIT_W2'] = 1 - (1 - df['ABS_EDGES_ONLY_IN_LOG_W2'])/(1 + df['MIN_COST_NORM']/10)
df['FIT_W3'] = 1 - (1 - df['ABS_EDGES_ONLY_IN_LOG_W3'])/(1 + df['MIN_COST_NORM']/10)

df_pp = df[['FITNESS','ABS_EDGES_ONLY_IN_LOG_W3']]

x = df[['ABS_EDGES_ONLY_IN_LOG_W3']]
y = df['FITNESS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
model = LinearRegression()

model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(r2_score(predictions, y_test))



# print(df[['FITNESS','ABS_EDGES_ONLY_IN_LOG_W1','ABS_EDGES_ONLY_IN_LOG_W2']])

# print(r2_score(df['FITNESS'],df['ABS_EDGES_ONLY_IN_LOG_W1']))

# sns.set_context(rc={"axes.labelsize":15})
# sns.pairplot(df_pp)

# plt.show(block=True)

# df_pp = df[['FITNESS','FIT_W3']]

# sns.set_context(rc={"axes.labelsize":15})
# sns.pairplot(df_pp)

# plt.show(block=True)

