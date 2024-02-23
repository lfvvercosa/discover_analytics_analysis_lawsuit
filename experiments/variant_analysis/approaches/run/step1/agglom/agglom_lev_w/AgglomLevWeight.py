from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.LevWeight import LevWeight
from experiments.variant_analysis.approaches.core.AgglomClust import AglomClust
from experiments.variant_analysis.approaches.core.metrics.\
     ComplexFitnessConnector import ComplexFitnessConnector
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
     import FindFitnessComplexity
from experiments.variant_analysis.utils.Utils import Utils


from sklearn.metrics import adjusted_rand_score
from pm4py import convert_to_dataframe
from scipy.cluster import hierarchy 
import pandas as pd
import pm4py
import matplotlib.pyplot as plt

class AgglomLevWeight():

    def run(self, 
            log, 
            n_clusters, 
            w_paral, 
            w_new_act, 
            w_subst,
            agglom_method):

        
        utils = Utils()
        fit_complex = ComplexFitnessConnector()
        cluster_finder = FindFitnessComplexity()

        best_fit = float('-inf')
        best_ARI = float('-inf')
        best_complex = float('inf')


        for wp in w_paral:
            for wn in w_new_act:
                for ws in w_subst:
                    for method in agglom_method:

                        df = convert_to_dataframe(log)
                        split_join = SplitJoinDF(df)
                        traces = split_join.split_df()

                        ids = split_join.ids
                        ids_clus = [l[0] for l in ids]
                        df_ids = pd.DataFrame.\
                                    from_dict({'case:concept:name':ids_clus})
                    
                        levWeight = LevWeight(traces,
                                              log,
                                              weight_parallel = wp,
                                              weight_new_act = wn,
                                              weight_substitute = ws,
                                             )

                        agglomClust = AglomClust(traces, 
                                                 levWeight.lev_metric_weight)
                        Z = agglomClust.cluster(method)

                        min_size = int(len(traces)/n_clusters)
                        logs = cluster_finder.find_best_match_clusters_hier(
                                    Z, 
                                    n_clusters, 
                                    min_size, 
                                    log, 
                                    traces, 
                                    None
                               )
                        
                        if logs is not None:

                            df_clus = self.create_unique_df(logs)
                            labels = self.get_labels(df_clus, ids_clus)

                            y_pred = labels
                            y_true = utils.get_ground_truth2(ids_clus)

                            ARI = adjusted_rand_score(y_true, y_pred)

                            # get df-log with cluster labels
                            split_join.join_df(labels)
                            df = split_join.df

                            fit, complx = fit_complex.\
                                        run_fitness_and_complexity(df)

                            if fit > best_fit:
                                print('best fitness: ' + str(fit))

                                best_ARI = ARI
                                best_fit = fit
                                best_complex = complx


                        # labels = hierarchy.fcluster(Z=Z, 
                        #                             t=n_clusters, 
                        #                             criterion='maxclust')
                        # labels = [l-1 for l in labels]

                        # y_pred = labels
                        # y_true = utils.get_ground_truth2(ids_clus)

                        # ARI = adjusted_rand_score(y_true, y_pred)

                        # get df-log with cluster labels
                        # split_join.join_df(labels)
                        # df = split_join.df


        return best_ARI, best_fit, best_complex
    

    def create_unique_df(self, logs):
        count = 0
        dfs = []

        for log in logs:
            df = pm4py.convert_to_dataframe(log)
            df['cluster_label'] = count
            dfs.append(df)

            count += 1


        return pd.concat(dfs)
    

    def get_labels(self, df_clus, ids_clus):
        df_work = df_clus.drop_duplicates('case:concept:name').\
                          set_index('case:concept:name')
        trace_cluster = df_work.to_dict()['cluster_label']
        labels = []

        for id in ids_clus:
            labels.append(trace_cluster[id])


        return labels