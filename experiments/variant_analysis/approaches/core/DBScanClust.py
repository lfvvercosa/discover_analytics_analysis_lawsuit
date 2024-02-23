from leven import levenshtein       
import numpy as np
from sklearn.cluster import dbscan


class DBScanClust:
    variants = None
    clustering = None


    def __init__(self, variants):
        self.variants = variants


    def lev_metric(self, x, y):
        i, j = int(x[0]), int(y[0])     # extract indices


        return levenshtein(self.variants[i], self.variants[j])
    

    def cluster(self, eps, min_samples):
        X = np.arange(len(self.variants)).reshape(-1, 1)
        self.clustering = dbscan(X, 
                                 metric=self.lev_metric, 
                                 eps=eps, 
                                 min_samples=min_samples)
        

        return list(self.clustering[1])
        