from experiments.variant_analysis.approaches.run.\
     literature.actitrac.ActiTraCConnector import ActiTracConnector
from experiments.variant_analysis.approaches.run.\
     step2.actitrac.AgglomComp import Agglom
from pm4py.algo.filtering.log.variants import variants_filter
import pm4py

        

class ActiTracAgglom:

    actitrac = None
    agglom = None
    CLUSTERS_MAX = 50

    def __init__(self):
        self.actitrac = ActiTracConnector()
        self.agglom = Agglom()


    def find_perc(self, total_variants):
        perc = 0.5

        while int(perc*total_variants) > self.CLUSTERS_MAX:
            perc -= 0.05

        
        return perc
    

    def get_clusters_size(self, 
                          number_of_clusters_1step, 
                          number_of_clusters_final,
                          total_variants):
        number_of_clusters_final
        n_clusters = []
        perc = self.find_perc(total_variants)

        for i in range(number_of_clusters_1step):
            clust = int(total_variants * perc)

            if clust < number_of_clusters_final:
                break

            n_clusters.append(clust)
            perc /= 2
        

        return n_clusters
    

    def run(self, 
            n_clusters, 
            is_greedy_params,
            dist_greed_params,
            target_fit_params,
            log_path, 
            saving_path, 
            number_of_trials_1step,
            number_of_clusters_final):
        
        best_fit = float('-inf')
        best_complex = float('inf')
        best_ARI = float('-inf')
        best_df = None

        log = pm4py.read_xes(log_path)
        variants = variants_filter.get_variants(log)
        total_variants = len(variants)
        number_of_clusters_1step = self.get_clusters_size(number_of_trials_1step,
                                                          number_of_clusters_final,
                                                          total_variants)
        
        params_agglom = {}
        params_agglom['custom_distance'] = 'alignment_between_logs'
        

        for target_fit in target_fit_params:
                for is_greedy in is_greedy_params:
                    for dist_greed in dist_greed_params:
                        for n_clus_1step in number_of_clusters_1step:
                            min_clus_size = 1/(n_clus_1step * 4)

                            ARI, fit, complx = self.actitrac.\
                            run_actitrac(n_clus_1step,
                                        is_greedy,
                                        dist_greed,
                                        target_fit,
                                        min_clus_size,
                                        log_path,
                                        saving_path)

                            if fit > best_fit:
                                best_fit = fit
                                best_complex = complx
                                best_ARI = ARI
                                best_df = self.actitrac.df.copy()
                                
        ARI2, fit2, complx2 = self.agglom.run(best_df, 
                                              params_agglom, 
                                              n_clusters)
        

        return ARI2, fit2, complx2