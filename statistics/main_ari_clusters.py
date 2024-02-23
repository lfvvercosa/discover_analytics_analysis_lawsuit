import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score

# Load the cluster_feat_all.csv file
df_feat_all = pd.read_csv('dataset/tribunais_trabalho/cluster_feat_all.csv', 
                          sep='\t')

ari = adjusted_rand_score(df_feat_all['CLUS_KME'], 
                          df_feat_all['CLUS_AGG'])

print()
