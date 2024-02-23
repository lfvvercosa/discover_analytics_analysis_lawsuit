from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from pathlib import Path
import os
from experiments.variant_analysis.approaches.core.metrics.\
     ics_fitness.ICSFitnessConnector import ICSFitnessConnector
from experiments.variant_analysis.approaches.core.metrics.\
     complexity_heu_net.ComplexityHNConnector import ComplexityHNConnector
from experiments.variant_analysis.approaches.core.metrics.\
     ComplexFitnessConnector import ComplexFitnessConnector
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity
from xes_files.creation.LogCreator import LogCreator


class RunKmeansNgram():

    SAVE_OUTPUT = False

    def rename_columns(self, df_gram):
        map_name = {n:str(n) for n in df_gram.columns}
        df_gram = df_gram.rename(columns=map_name)


        return df_gram
        

    def cashe_df_grams(self, log, ngram, min_percent, max_percent):
        cashed_dfs = {}
        print('cashing df-grams...')

        for n in ngram:
            cashed_dfs[n] = {}
            for min_perc in min_percent:
                cashed_dfs[n][min_perc] = {}
                for max_perc in max_percent:
                    cashed_dfs[n][min_perc][max_perc] = {}
                    # convert to df
                    df = convert_to_dataframe(log)

                    # create n-gram
                    split_join = SplitJoinDF(df)
                    traces = split_join.split_df()
                    ids = split_join.ids
                    ids_clus = [l[0] for l in ids]
                    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                    # get ground-truth
                    utils = Utils()
                    y_true = utils.get_ground_truth2(ids_clus)

                    df_gram = create_n_gram(df_clus, 'case:concept:name', 'concept:name', n)
                    df_gram = self.rename_columns(df_gram)
                    df_gram.index = pd.Categorical(df_gram.index, ids_clus)
                    df_gram = df_gram.sort_index()

                    preProcess = PreProcessClust()
                    rem_cols = preProcess.selectColsByFreq(min_perc, max_perc, df_gram)
                    df_gram = df_gram.drop(columns=rem_cols)

                    # df_gram.to_csv('temp/df_gram.csv', sep='\t')

                    # normalize n-gram

                    for c in df_gram.columns:
                        if len(df_gram[c].drop_duplicates().to_list()) == 1:
                            df_gram = df_gram.drop(columns=c)

                    df_gram_norm = df_gram.copy()

                    for c in df_gram.columns:
                        df_gram_norm[c] = (df_gram_norm[c] - df_gram_norm[c].min())/ \
                                          (df_gram_norm[c].max() - df_gram_norm[c].min())

                    # df_gram_norm = (df_gram - df_gram.min())/\
                    #             (df_gram.max() - df_gram.min())
                    
                    df_gram_norm = df_gram_norm.round(4)

                    cashed_dfs[n][min_perc][max_perc] = df_gram_norm.copy()


        return cashed_dfs


    def run(self, 
            log,
            clusters,
            ngram,
            min_percents,
            max_percents,
            exp_backlog,
            reps,
            k_markov):
        
        if exp_backlog == '':
            self.SAVE_OUTPUT = False

        log_creator = LogCreator()
        log = log_creator.remove_single_activity_traces(log)

        best_ARI = -1
        best_ARI_list = None
        best_VM = -1
        best_clusters = -1
        best_ngram = -1
        best_min_percents = -1
        best_max_percents = -1
        best_df_distrib = None
        best_dict_var = None
        best_dict_gd = None
        best_fit = float('-inf')
        best_complex = float('inf')
        best_split_join = None

        total = len(clusters) * len(ngram) * len(min_percents) * \
                len(max_percents) * reps
        count = 0

        cashed_dfs = self.cashe_df_grams(log, ngram, min_percents, max_percents)

        ## Calculate fitness and complexity
        # fit_complex = FindFitnessComplexity()
        fit_complex = ComplexFitnessConnector()
        fit_ics = ICSFitnessConnector()
        complex_hn = ComplexityHNConnector()


        for n_clusters in clusters:
            for n in ngram:
                for min_perc in min_percents:
                    for max_perc in max_percents:
                        ari_avg = 0
                        vm_avg = 0
                        fit_avg = 0
                        complex_avg = 0
                        valid_simulation = True

                        # if count % 10 == 0:
                        print('progress: ' + str(round((count/total)*100,2)) + '%\n')

                        count += 1
                        # ARI_list = []

                        for i in range(reps):

                            df = convert_to_dataframe(log)
                            split_join = SplitJoinDF(df)
                            traces = split_join.split_df()
                            ids = split_join.ids
                            ids_clus = [l[0] for l in ids]
                            df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                            df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                            df_gram_norm = cashed_dfs[n][min_perc][max_perc]

                            if df_gram_norm.empty:
                                valid_simulation = False

                                print('#### Not valid simulation ####')

                                break

                            # print()

                            # apply first-step clustering (KMeans)
                            model = KMeans(n_clusters=n_clusters, n_init=5)
                            model.fit(df_gram_norm)
                            cluster_labels = list(model.labels_)

                            centroids = model.cluster_centers_

                            if len(set(cluster_labels)) < n_clusters:
                                valid_simulation = False

                                print('#### Not valid simulation ####')

                                break

                            # centroids = np.around(centroids, decimals=4)
                            # df_centroids = pd.DataFrame(centroids, 
                            #                             index=list(range(n_clusters)),
                            #                             columns=df_gram_norm.columns)

                            # get performance by adjusted rand-score metric
                            utils = Utils()
                            y_pred = cluster_labels
                            y_true = utils.get_ground_truth2(ids_clus)

                            # get df-log with cluster labels
                            split_join.join_df(cluster_labels)
                            df = split_join.df
                            df_variants = split_join.join_df_uniques(cluster_labels)
                            dict_labels = {'index':ids_clus, 
                                            'cluster_label':cluster_labels}
                            
                            df_labels = pd.DataFrame.from_dict(dict_labels)
                            df_labels = df_labels.set_index('index')
                            df_gram_norm = df_gram_norm.join(df_labels, how='left')

                            # get distribution of traces cluster per variant
                            stats = StatsClust()
                            df_distrib = stats.get_distrib(df, df_ids)

                            # get variants by cluster
                            dict_var = stats.get_variants_by_cluster(traces, 
                                                                        cluster_labels)
                            # print()

                            # get variants ground truth by cluster
                            dict_gd = stats.get_ground_truth_by_cluster(dict_var, 
                                                                        traces, 
                                                                        y_true)
                            
                            ARI = adjusted_rand_score(y_true, y_pred)
                            ari_avg += ARI
                            # ARI_list.append(ARI)
                        
                            Vm = v_measure_score(y_true, y_pred)
                            vm_avg += Vm

                            # if n_clusters == number_of_clusters:

                            # fit, complex = fit_complex.\
                            #                 get_metrics_from_simulation(df_variants, 
                            #                                             k_markov)

                            fit, complex = fit_complex.run_fitness_and_complexity(df)
                            # fit = fit_ics.run_ics_fitness_metric(df)
                            # complex = complex_hn.run_complexity_metric(df)

                            fit_avg += fit
                            complex_avg += complex

                        if valid_simulation:

                            ari_avg /= reps
                            vm_avg /= reps

                            # if n_clusters == number_of_clusters:
                            fit_avg /= reps
                            complex_avg /= reps

                            if fit_avg > best_fit:
                                best_fit = fit_avg
                                best_ARI = ari_avg
                                best_VM = vm_avg
                                best_complex = complex_avg

                                best_clusters = n_clusters
                                best_ngram = n
                                best_min_percents = min_perc
                                best_max_percents = max_perc
                                best_df_distrib = df_distrib.copy()
                                best_dict_var = dict_var.copy()
                                best_dict_gd = dict_gd.copy()
                                best_split_join = split_join
                        else:
                            continue


        # print('best params')
        # print('best_ARI: ' + str(best_ARI))
        # print('best_VM: ' + str(best_VM))
        # print('best_clusters: ' + str(best_clusters))
        # print('best_ngram: ' + str(best_ngram))
        # print('best_min_percents: ' + str(best_min_percents))
        # print('best_max_percents: ' + str(best_max_percents))
        # print('best_fit: ' + str(best_fit))
        # print('best_complex: ' + str(best_complex))

        last_pos_dir = exp_backlog.rfind('/')
        dir = exp_backlog[:last_pos_dir]

        if self.SAVE_OUTPUT:
            if not os.path.exists(dir):
                os.makedirs(dir)

            with open(exp_backlog, 'w') as f:
                f.write('Adjusted Rand Score (ARI): ' + str(best_ARI) + '\n\n')
                f.write('df_distrib: \n\n')
                f.write('V-measure (Vm): ' + str(best_VM) + '\n\n')
                f.write('best_clusters: ' + str(best_clusters) + '\n\n')
                f.write('best_ngram: ' + str(best_ngram) + '\n\n')
                f.write('best_min_percents: ' + str(best_min_percents) + '\n\n')
                f.write('best_max_percents: ' + str(best_max_percents) + '\n\n')
                f.write('best df_distrib: \n\n')
                f.write(best_df_distrib.to_string(header=True, index=True) + '\n\n')
                f.write('dict_var: \n\n')
                f.write(str(best_dict_var) + '\n\n')
                f.write('dict_gd: \n\n')
                f.write(str(best_dict_gd) + '\n\n')

            
        
        return best_ARI, \
               best_fit, \
               best_complex, \
               best_split_join, \
               best_dict_var, \
               best_df_distrib, \
               best_ngram, \
               best_min_percents, \
               best_max_percents