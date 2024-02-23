from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import numpy as np

import pm4py    
from pm4py.algo.discovery.inductive.variants.im_f import algorithm as IMf
from pm4py.algo.discovery.inductive.variants.im_clean import algorithm as IM
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.replay_fitness.variants import alignment_based 
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.algo.conformance.alignments.petri_net import variants
from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
from pm4py.statistics.variants.log import get as variants_module


import utils.read_and_write.s3_handle as s3_handle

from pm4py.visualization.petri_net import visualizer as pn_visualizer


class CustomAgglomClust:


    def gen_dendrogram(self, df, dist='average'):
        X = df.to_numpy()
        dist_matrix = distance_matrix(X,X)
        Z = hierarchy.linkage(dist_matrix, dist)

        hierarchy.dendrogram(Z)

        plt.show(block=True)
        plt.close()


    def fit_petri_net_cross_align(self, df_log1, df_log2):
        log1 = pm4py.convert_to_event_log(df_log1)
        log2 = pm4py.convert_to_event_log(df_log2)

        thresh = 0.8

        params_ind = {IMf.Parameters.NOISE_THRESHOLD:thresh}
        net1, im1, fm1 = IMf.apply(log1, params_ind)
        net2, im2, fm2 = IMf.apply(log2, params_ind)
        
        align_var = variants.dijkstra_less_memory
        params_ali = {replay_fitness.Parameters.ALIGN_VARIANT:align_var}

        v1 = replay_fitness.apply(log2, net1, im1, fm1, 
                                  variant=replay_fitness.Variants.ALIGNMENT_BASED,
                                  parameters=params_ali)
        
        v2 = replay_fitness.apply(log1, net2, im2, fm2, 
                                  variant=replay_fitness.Variants.ALIGNMENT_BASED,
                                  parameters=params_ali)


        return (v1['log_fitness']*len(log2) + v2['log_fitness']*len(log1))\
               /(len(log1) + len(log2))
    

    def fit_earth_move(self, df_log1, df_log2):
        log1 = pm4py.convert_to_event_log(df_log1)
        log2 = pm4py.convert_to_event_log(df_log2)

        language1 = variants_module.get_language(log1)    
        language2 = variants_module.get_language(log2)


        return 1 - emd_evaluator.apply(language1, language2)


    def fit_log_alignment(self, df_log1, df_log2):
        # print('fit_log_alignment')

        log1 = pm4py.convert_to_event_log(df_log1)
        log2 = pm4py.convert_to_event_log(df_log2)

        align_log1_2 = logs_alignments.apply(log1, log2)
        align_log2_1 = logs_alignments.apply(log2, log1)

        fit_log1_2 = [e['fitness'] for e in align_log1_2]
        fit_log2_1 = [e['fitness'] for e in align_log2_1]

        fitness = fit_log1_2 + fit_log2_1


        return sum(fitness)/len(fitness)


    def get_fitness(self, type, df_log1, df_log2):
        if type == 'earth_move_distance':
            return self.fit_earth_move(df_log1, df_log2)
        elif type == 'petri_net_cross_alignment':
            return self.fit_petri_net_cross_align(df_log1, df_log2) 
        elif type == 'alignment_between_logs':
            return self.fit_log_alignment(df_log1, df_log2)   
        
        raise Exception('Unknown distance measure!')


    def aggregate(self, clusters, i, j):
        for e in clusters[j]:
            clusters[i].append(e)

        del clusters[j]


    def agglom_fit(self, df, params=None):
        clusters = list(df['cluster_label'].drop_duplicates())
        clusters.sort()
        clusters = [[c] for c in clusters]
        result = []
        count = 0

        if params is not None:
            if 'AWS_filename' in params or 'DEBUG' in params:
                total = len(clusters)
    

        while len(clusters) > 1:
            max_fit = (-1, (-1,-1))
            count += 1

            for i in range(len(clusters) - 1):
                for j in range(i + 1, len(clusters)):

                    df_log1 = df[df['cluster_label'].isin(clusters[i])]
                    df_log2 = df[df['cluster_label'].isin(clusters[j])]

                    if 'custom_distance' in params:
                        fit = self.get_fitness(params['custom_distance'],
                                                df_log1,
                                                df_log2)
                    else:
                        fit = self.fit_petri_net_cross_align(df_log1, df_log2)

                    if fit > max_fit[0]:
                        new_max = (fit, (i, j))
                        max_fit = new_max
            
            c1 = clusters[max_fit[1][0]].copy()
            c2 = clusters[max_fit[1][1]].copy()
            fit = round(max_fit[0],4)

            # print(c1,c2,fit)

            result.append((c1,c2,fit))
            self.aggregate(clusters, max_fit[1][0], max_fit[1][1])

            if params is not None:
                if 'AWS_filename' in params:
                    progress = 1 - len(clusters)/total
                    progress = str(round(progress * 100,3)) + '%'
                    print('progress: ', progress)

                    s3_handle.write_to_s3(params['AWS_bucket'],
                                          params['AWS_filename'],
                                          progress
                                         )
                elif 'DEBUG' in params:
                    if params['DEBUG']:
                        progress = 1 - len(clusters)/total
                        progress = str(round(progress, 3) * 100) + '%'
                        
                        print('progress: ' + progress)


        return result
    

    def gen_Z(self, dend):
        all = dend[-1][0] + dend[-1][1]
        count = max(all) + 1
        Z  =[]
        labels = {}
        
        def label(l):
            if len(l) == 1:
                return l[0]
            else:
                return labels[str(l)]


        for i,t in enumerate(dend):

            e = [label(t[0]), 
                 label(t[1]), 
                 round(1 - t[2],4)*100, 
                 len(t[0] + t[1])]
            Z.append(e)

            labels[str(t[0] + t[1])] = count + i


        Z = np.array(Z)

        # print(Z)


        return Z
    

       