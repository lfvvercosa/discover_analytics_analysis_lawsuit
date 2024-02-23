from leven import levenshtein       
import numpy as np
from scipy.cluster import hierarchy


class AglomClustNgram:
        

    def cluster(self, df, method='average', metric='euclidean'):
        X = df.to_numpy()
        Z = hierarchy.linkage(X, method=method, metric=metric)
        

        return Z

