from sklearn.cluster import AgglomerativeClustering
import numpy as np


X = np.array([[1, 1], [1, 1], [1, 1],
              [4, 4], [4, 4], [4, 4]])

clustering = AgglomerativeClustering(n_clusters=2).fit(X)

print(clustering.labels_)