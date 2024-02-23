from experiments.variant_analysis.approaches.core.CustomAgglomClust \
    import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.metrics.\
     ComplexFitnessConnector import ComplexFitnessConnector

from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score


class Agglom:

    def cut_n_clusters(self, Z, num_clusters):
        min_perc_dendro = 0
        max_perc_dendro = 1
        step_perc = 0.01
        perc = min_perc_dendro
        t = max(Z[:,2])

        while perc <= max_perc_dendro:
            labels = hierarchy.fcluster(Z=Z, 
                                        t=perc*t, 
                                        criterion='distance')
            
            if len(set(labels)) == num_clusters:
                break

        
        return labels
    

    def run(self, split_join, params_agglom, num_clusters, dict_var, k_markov):
        
        agglomClust = CustomAgglomClust()
        # fit_complex = FindFitnessComplexity()
        fit_complex = ComplexFitnessConnector()


        utils = Utils()
        stats_clus = StatsClust()

        dend = agglomClust.agglom_fit(split_join.df, params_agglom)

        if dend:
            Z = agglomClust.gen_Z(dend)

            if len(dend) < num_clusters:
                labels = list(range(num_clusters))
            else:
                # labels = self.cut_n_clusters(Z, num_clusters)
                labels = hierarchy.fcluster(Z=Z, 
                                            t=num_clusters, 
                                            criterion='maxclust')
                labels = [l-1 for l in labels]

            # get performance by adjusted rand-score metric
            
            ids = split_join.ids
            ids_clus = [l[0] for l in ids]

            y_true = utils.get_ground_truth2(ids_clus)
            y_pred = utils.get_agglom_labels_by_trace(split_join.traces, dict_var, labels)

            ARI = adjusted_rand_score(y_true, y_pred)
            df_distrib = stats_clus.get_distrib2(ids_clus, y_pred)

            df_variants = split_join.join_df_uniques(y_pred)
            df_full = split_join.join_df(y_pred)

            # fit, complex = fit_complex.\
            #                     get_metrics_from_simulation(df_variants, 
            #                                                 k_markov)
            
            fit, complex = fit_complex.run_fitness_and_complexity(df_full)
                                

            return ARI, fit, complex, df_distrib, df_full
        

        return None
