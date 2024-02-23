import numpy as np
from scipy.cluster import hierarchy

from experiments.variant_analysis.approaches.core.context_aware_clust.\
     ContextAwareClust import ContextAwareClustering
from experiments.variant_analysis.approaches.core.Agglom import Aglom

class ContextAgglomClustering(Aglom):
    variants = None
    clustering = None
    metric = None
    context_aware_clust = None


    def __init__(self, log, variants):
        self.variants = variants
        self.context_aware_clust = ContextAwareClustering(log)
        self.metric = self.lev_metric
        

    def lev_metric(self, x, y):
        i, j = int(x[0]), int(y[0])     # extract indices

        print(self.variants[i], self.variants[j])

        return self.context_aware_clust.levenshtein_context(self.variants[i],
                                                            self.variants[j])


    def cluster(self, method='average'):
        X = np.arange(len(self.variants)).reshape(-1, 1)
        Z = hierarchy.linkage(X, method=method, metric=self.metric)
        

        return Z

