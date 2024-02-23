from experiments.variant_analysis.approaches.run.\
     step1.kmeans.RunKmsNgram import RunKmeansNgram
from experiments.variant_analysis.approaches.run.\
     step2.crossfitness.ngram.Agglom import Agglom

from pm4py.algo.filtering.log.variants import variants_filter

from utils.global_var import DEBUG

import os


class KmsNgramAgglom:

    SAVE_OUTPUT = True
    CLUSTERS_MAX = 50

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
            log,
            clusters,
            ngram,
            min_percents,
            max_percents,
            backlog_path,
            reps,
            k_markov,
            number_of_clusters_1step,
            number_of_clusters_final,
            params_agglom):
        
        best_fit = float('-inf')
        best_ARI = -1
        best_complex = float('inf')
        best_n_clusters = -1
        best_df_distrib_1step = None
        best_df_distrib_2step = None
        best_df_full = None

        variants = variants_filter.get_variants(log)
        total_variants = len(variants)

        number_of_clusters_1step = self.get_clusters_size(number_of_clusters_1step,
                                                          number_of_clusters_final,
                                                          total_variants)
        
        number_of_clusters_1step.append(3)

        number_of_clusters_1step = list(set(number_of_clusters_1step))
        # number_of_clusters_1step = [3]
        
        if not number_of_clusters_1step:
            number_of_clusters_1step = [number_of_clusters_final]
            # raise ValueError('It was not possible to find clusters')
        
        for number_of_clusters in number_of_clusters_1step:
            run_kms = RunKmeansNgram()
            ARI, fit, complex_, split_join, dict_var, df_distrib, \
            best_ngram, best_min_perc, best_max_perc = \
                run_kms.run(log,
                            [number_of_clusters],
                            ngram,
                            min_percents,
                            max_percents,
                            '',
                            reps,
                            k_markov)
            
            if fit == float('-inf'):
                continue

            run_agglom = Agglom()
            ARI2, fit2, complex_2, df_distrib2, df_full = \
                run_agglom.run(split_join,
                            params_agglom,
                            number_of_clusters_final,
                            dict_var,
                            k_markov
                            )
            
            if fit2 > best_fit:
                best_fit = fit2
                best_ARI = ARI2
                best_complex = complex_2
                best_n_clusters = number_of_clusters
                best_df_distrib_1step = df_distrib.copy()
                best_df_distrib_2step = df_distrib2.copy()
                best_df_full = df_full.copy()


        if best_fit == float('-inf'):
            raise ValueError('It was not possible to find clusters')

        last_pos_dir = backlog_path.rfind('/')
        my_path = backlog_path[:last_pos_dir]

        if self.SAVE_OUTPUT:
            if not os.path.exists(my_path):
                os.makedirs(my_path)

            with open(backlog_path, 'w') as f:
                f.write('best_fit: ' + str(best_fit) + '\n\n')
                f.write('best_ARI: ' + str(best_ARI) + '\n\n')
                f.write('best_complex: ' + str(best_complex) + '\n\n')
                f.write('best_n_clusters: ' + str(best_n_clusters) + '\n\n')
                f.write('best_df_distrib_1step: ' + str(best_df_distrib_1step) + '\n\n')
                f.write('best_ngram_1step: ' + str(best_ngram) + '\n\n')
                f.write('best_min_percent_1step: ' + str(best_min_perc) + '\n\n')
                f.write('best_max_percent_1step: ' + str(best_max_perc) + '\n\n')
                f.write('best_df_distrib_2step: ' + str(best_df_distrib_2step) + '\n\n')


        # log_clusters = split_join.split_df_clusters(best_df_full)

        return best_ARI, best_fit, best_complex, None