import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Load the feature importance file
feature_import = "dataset/tribunais_trabalho/statistics/svr_feat_import.csv"
out_path = 'dataset/tribunais_trabalho/statistics/feat_import_svr.png'

df_feat_import = pd.read_csv(feature_import, sep='\t')



color1 = (0.16696655132641292, 0.48069204152249134, 0.7291503267973857)

# Sort the dataframe by import_gain in descending order
df_feat_import_sorted = df_feat_import.sort_values(by='import_mean',
                                                         ascending=False)

# Get the top ten features
top_ten_features = df_feat_import_sorted.head(10)

# Data
labels = top_ten_features['Feature'].tolist()
importance = top_ten_features['import_mean'].tolist()

# gain = [g/1e11 for g in gain]
# split = [s/1e2 for s in split]

# Map values of labels list to custom names
custom_names = {
                'CLUS_AGG':'CL$_{Ag}$',
                'CLASSE_PROCESSUAL':'Class',
                'CLUS_KME':'CL$_{Km}$',
                'NUMERO_TRT':'Justice',
                'ASSU_LIQUIDACAO___CUMPRIMENTO___EXECUCAO':'Subject$_{2}$',
                'PROCESSO_DIGITAL':'Digital',
                'MOV_REMESSA_982':'PM$_{5}$',
                'MOV_REMESSA_123':'PM$_{6}$',
                'MOV_LIQUIDACAO_INICIADA_11384':'PM$_{2}$',
                'MOV_DESARQUIVAMENTO_893':'PM$_{7}$'
               }

labels = [custom_names.get(label, label) for label in labels]

# Create bar plot
fig, ax = plt.subplots()
ax.bar(labels, importance, color=color1, width=0.5)

# Set x-axis tick labels, 
ax.set_xticklabels(labels, rotation=90)

# Set y-axis label
ax.set_ylabel('Importance', fontsize=12, labelpad=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylim(0, 0.18)
# ax.set_yticks([0, 0.05, 0.1, 0.15])
# ax.set_yscale('log')

# Set y-axis ticks and labels
# yticks = [0.001, 0.01, 0.1, 1]
# yticklabels = ['0.001', '0.01', '0.1', '1']
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticklabels)

# Add value labels to the bars
# for i, v in enumerate(importance):
# ax.bar(labels, importance, color=color1, width=0.8)  # Decrease the width to 0.8 or any desired value

# Save the plot
plt.tight_layout()
plt.savefig(out_path, dpi=400)
print('done!')