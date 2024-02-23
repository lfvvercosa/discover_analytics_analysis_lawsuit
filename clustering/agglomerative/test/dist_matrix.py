import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist 
from scipy.cluster import hierarchy
import Levenshtein 
import numpy as np


def leven(a, b):
    lev = Levenshtein.distance(a, b)
        
    return lev


def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])  # extract indices

    return leven(variants[i], variants[j])


def get_position(m, i, j):
    return m * i + j - ((i + 2) * (i + 1)) // 2


if __name__ == '__main__':
    variants = ['ABC','ABCD','ACB', 'ADCBB']
    X = np.arange(len(variants)).reshape(-1, 1)
    m = len(variants)

    dist_matrix = pdist(X, metric=lev_metric)

    print(dist_matrix)
    print('distance ' + variants[0] + ' and ' + variants[1] + ': ' + str(get_position(m, 0, 1)))
    print('distance ' + variants[1] + ' and ' + variants[0] + ': ' + str(get_position(m, 1, 0)))
    print('distance ' + variants[0] + ' and ' + variants[2] + ': ' + str(get_position(m, 0, 2)))
    print('distance ' + variants[0] + ' and ' + variants[3] + ': ' + str(get_position(m, 0, 3)))

    Z = hierarchy.linkage(dist_matrix, method='single', metric=lev_metric)
    a = hierarchy.dendrogram(Z,
                             labels=variants,
                             leaf_rotation=90)
    plt.tight_layout()
    plt.show()
