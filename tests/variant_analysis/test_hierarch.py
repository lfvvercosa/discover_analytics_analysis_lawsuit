import numpy as np
from scipy.cluster.hierarchy import fclusterdata
 
 
# a custom function that just computes Euclidean distance
def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5
 
X = np.random.randn(100, 2)
 
fclust1 = fclusterdata(X, 1.0, metric=mydist)
fclust2 = fclusterdata(X, 1.0, metric='euclidean')
 
print(np.allclose(fclust1, fclust2))
# True