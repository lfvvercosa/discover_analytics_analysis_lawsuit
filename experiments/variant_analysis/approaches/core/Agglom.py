from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score

from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.utils.Utils import Utils


class Aglom:

    def __init__(self):
        return

    def get_best_cutting_point(self, Z, y_true, traces):
        t = max(Z[:,2])
        stats = StatsClust()
        utils = Utils()

        min_perc = 0.3
        max_perc = 0.9
        step_perc = 0.025
        perc = min_perc

        best_ARI = -1
        best_perc = -1
        

        while perc <= max_perc:
            labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')
            labels = [l-1 for l in labels]

            # get variants by cluster
            dict_var = stats.get_variants_by_cluster(traces, labels)

            # get performance by adjusted rand-score metric
            
            # y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

            ARI = adjusted_rand_score(y_true, labels)

            if ARI > best_ARI:
                best_ARI = ARI
                best_perc = perc

            perc += step_perc
        

        return best_perc


