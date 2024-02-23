from sklearn import metrics


labels_true = [0, 0, 0, 0, 1, 1, 1]
labels_pred = [3, 3, 3, 3, 3, 3, 3]


print('ARI: ' + str(metrics.adjusted_rand_score(labels_true, labels_pred)))

print('Homogeneity: ' + str(metrics.homogeneity_score(labels_true, labels_pred)))

print('Completeness: ' + str(metrics.completeness_score(labels_true, labels_pred)))

print('V-measure: ' + str(metrics.v_measure_score(labels_true, labels_pred)))
