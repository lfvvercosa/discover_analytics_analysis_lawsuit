from leven import levenshtein       
import numpy as np
from scipy.cluster import hierarchy


class AglomClust:
    variants = None
    clustering = None
    metric = None

    def __init__(self, variants, metric=None):
        self.variants = variants

        if not metric:
            self.metric = self.lev_metric
        else:
            self.metric = metric


    def lev_metric(self, x, y):
        i, j = int(x[0]), int(y[0])     # extract indices

        return levenshtein(self.variants[i], self.variants[j])
    

    def cluster(self, method='average'):
        X = np.arange(len(self.variants)).reshape(-1, 1)
        Z = hierarchy.linkage(X, method=method, metric=self.metric)
        

        return Z

