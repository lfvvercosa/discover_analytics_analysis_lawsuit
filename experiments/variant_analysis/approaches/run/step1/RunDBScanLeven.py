from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.DBScanClust import DBScanClust
from experiments.variant_analysis.utils.Utils import Utils


class RunDBScanLeven:
    def run_best_ARI(self, log, eps_list):

        # convert to df
        df = convert_to_dataframe(log)

        # extract variants from log
        split_join = SplitJoinDF(df)
        traces = split_join.split_df()
        ids = split_join.ids
        ids_clus = [l[0] for l in ids]
        # df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
        best_ARI = float('-inf')
        best_eps = 0
        min_samples = 1
        utils = Utils()

        for eps in eps_list:
            dbscan = DBScanClust(traces)
            labels = dbscan.cluster(eps, min_samples)

            y_pred = labels
            y_true = utils.get_ground_truth(ids_clus)

            ARI = adjusted_rand_score(y_true, y_pred)
            
            if ARI > best_ARI:
                best_ARI = ARI
                best_eps = eps
            
        
        return (best_ARI, best_eps)