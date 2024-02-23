from leven import levenshtein       
import numpy as np
from sklearn.cluster import dbscan



data = ["ABD", "ACEF", "ACE", # p1v1
        "ABDCEF", "ACBEDF", "ACEBD", # p1v2
        "ABED", "ACBEDFD", "ACBEF", # p1v3
        "GADADADE","GCF", "GADCF", #p2v1
        "BGADCFE","BGCAFDE", "BGCFADE", #p2v2
       ]
def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return levenshtein(data[i], data[j])

X = np.arange(len(data)).reshape(-1, 1)
clustering = dbscan(X, metric=lev_metric, eps=1, min_samples=2)

for i in clustering[0]:
    print(data[i] + ': ' + str(clustering[1][i]))