import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Load the feature importance file
feature_import_split = "dataset/tribunais_trabalho/statistics/lgbm_feat_import_split.csv"
feature_import_gain = "dataset/tribunais_trabalho/statistics/lgbm_feat_import_gain.csv"
out_path = 'dataset/tribunais_trabalho/statistics/feat_import.png'

df_feat_import_gain = pd.read_csv(feature_import_gain, sep='\t')
df_feat_import_split = pd.read_csv(feature_import_split, sep='\t')

df_feat_import_gain = df_feat_import_gain[['Feature', 'import_gain']]
df_feat_import_split = df_feat_import_split[['Feature', 'import_split']]

color1 = (0.16696655132641292, 0.48069204152249134, 0.7291503267973857)
color2 = (0.5356862745098039, 0.746082276047674, 0.8642522106881968)

# Sort the dataframe by import_gain in descending order
df_feat_import_gain_sorted = df_feat_import_gain.sort_values(by='import_gain', ascending=False)

# Get the top ten features
top_ten_features = df_feat_import_gain_sorted.head(10)
top_ten_features = top_ten_features.merge(df_feat_import_split, on='Feature', how='left')

# Data
labels = top_ten_features['Feature'].tolist()
gain = top_ten_features['import_gain'].tolist()
split = top_ten_features['import_split'].tolist()

# gain = [g/1e11 for g in gain]
# split = [s/1e2 for s in split]

# Map values of labels list to custom names
custom_names = {
                'CLUS_AGG':'CL$_{Ag}$',
                'CLASSE_PROCESSUAL':'RE$_{Cl}$',
                'CLUS_KME':'CL$_{Km}$',
                'NUMERO_TRT':'RE$_{Ju}$',
                'ASSU_RESCISAO_DO_CONTRATO_DE_TRABALHO':'RE$_{Su1}$',
                'PROCESSO_DIGITAL':'RE$_{Di}$',
                'MOV_DESARQUIVAMENTO_893':'PM$_{4}$',
                'TOTAL_ASSUNTOS_DISTINTOS':'RE$_{Sus}$',
                'MOV_ENTREGA_EM_CARGA_VISTA_493':'PM$_{3}$',
                'TOTAL_MAGISTRATE':'PM$_{Mag}$'
               }

labels = [custom_names.get(label, label) for label in labels]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, gain, width, label='Gain    ', color=color1)

ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, split, width, label='#Splits', color=color2)

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Log(Gain) (10$^{9}$)', fontsize=12)
ax.set_yscale('log')
ax.yaxis.labelpad = 14

# ax.set_title('Scores by group and values')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)

ax.set_yticks([1e+9, 10e+9, 100e+9, 390e+9])
ax.set_yticklabels(['1','10', '100', '390'])

ax.legend(loc='upper right', bbox_to_anchor=(1, 1),frameon=False)
ax2.set_ylabel('#Splits', fontsize=12)
ax2.yaxis.labelpad = 14

ax2.legend(bbox_to_anchor=(1, 0.93),frameon=False)

ax.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

ax2.set_yticks([100, 1000, 2000, 3000, 4000])
ax2.set_yticklabels(['100','1000', '2000', '3000', '4000'])

plt.tight_layout()
plt.savefig(out_path, dpi=400)
# plt.show()

