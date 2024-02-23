from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.CustomAgglomClust \
    import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity
from xes_files.creation.LogCreator import LogCreator

def rename_columns(df_gram):
    map_name = {n:str(n) for n in df_gram.columns}
    df_gram = df_gram.rename(columns=map_name)


    return df_gram
    

def cashe_df_grams(log, ngram, min_percent, max_percent):
    cashed_dfs = {}
    print('cashing df-grams...')

    for n in ngram:
        cashed_dfs[n] = {}
        for min_perc in min_percent:
            cashed_dfs[n][min_perc] = {}
            for max_perc in max_percent:

                # if n == 2 and min_perc == 0 and max_perc == 1:
                #     print()

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
                df_gram = rename_columns(df_gram)
                df_gram.index = pd.Categorical(df_gram.index, ids_clus)
                df_gram = df_gram.sort_index()

                preProcess = PreProcessClust()
                rem_cols = preProcess.selectColsByFreq(min_perc, max_perc, df_gram)
                df_gram = df_gram.drop(columns=rem_cols)

                # df_gram.to_csv('temp/df_gram.csv', sep='\t')

                # normalize n-gram
                df_gram_norm = (df_gram - df_gram.min())/\
                            (df_gram.max() - df_gram.min())
                df_gram_norm = df_gram_norm.round(4)

                cashed_dfs[n][min_perc][max_perc] = df_gram_norm.copy()


    return cashed_dfs


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/variant_analysis/exp7/size10/low_complexity/0/log.xes'
    log = xes_importer.apply(log_path)

    log_creator = LogCreator()
    log = log_creator.remove_single_activity_traces(log)

    exp_backlog = 'experiments/variant_analysis/exp7/results/' + \
                  'size10/low_comp/0/1step_ngram_kms.txt'

    clusters = [2,3,4,5,6,7]
    ngram = [1,2]
    min_percents = [0, 0.1, 0.2, 0.3]
    max_percents = [1, 0.9, 0.8, 0.7]
    reps = 1

    # clusters = [5]
    # ngram = [2]
    # min_percents = [0.15]
    # max_percents = [1]
    # reps = 1

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
    best_fit = -1
    best_complex = float('inf')

    # best_ARI = 0.7841302081466887
    # best_clusters = 6
    # best_ngram = 1
    # best_min_percents = 0
    # best_max_percents = 1

    total = len(clusters) * len(ngram) * len(min_percents) * \
            len(max_percents) * reps
    count = 0

    cashed_dfs = cashe_df_grams(log, ngram, min_percents, max_percents)

    ## Calculate fitness and complexity
    fit_complex = FindFitnessComplexity()
    k_markov = 2


    for n_clusters in clusters:
        print('total clusters: ', n_clusters)
        for n in ngram:
            print('ngram: ',n)
            for min_perc in min_percents:
                for max_perc in max_percents:
                    ari_avg = 0
                    vm_avg = 0
                    fit_avg = 0

                    if count % 10 == 0:
                        print('progress: ' + str(round((count/total)*100,2)) + '%\n')

                    count += 1
                    ARI_list = []

                    for i in range(reps):

                        df = convert_to_dataframe(log)
                        split_join = SplitJoinDF(df)
                        traces = split_join.split_df()
                        ids = split_join.ids
                        ids_clus = [l[0] for l in ids]
                        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                        df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                        df_gram_norm = cashed_dfs[n][min_perc][max_perc]

                        # apply first-step clustering (KMeans)
                        model = KMeans(n_clusters=n_clusters)
                        model.fit(df_gram_norm)
                        cluster_labels = list(model.labels_)

                        centroids = model.cluster_centers_
                        centroids = np.around(centroids, decimals=4)
                        df_centroids = pd.DataFrame(centroids, 
                                                    index=list(range(n_clusters)),
                                                    columns=df_gram_norm.columns)

                        # get performance by adjusted rand-score metric
                        utils = Utils()
                        y_pred = cluster_labels
                        y_true = utils.get_ground_truth2(ids_clus)

                        # if i == reps - 1:
                            
                        # get df-log with cluster labels
                        split_join.join_df(cluster_labels)
                        df = split_join.df
                        df_variants = split_join.join_df_uniques(cluster_labels)
                        dict_labels = {'index':ids_clus, 
                                        'cluster_label':cluster_labels}
                        
                        try:
                            df_labels = pd.DataFrame.from_dict(dict_labels)
                        except Exception as e:
                            print(e)
                            raise Exception(e)
                        
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
                        ARI_list.append(ARI)
                    
                        Vm = v_measure_score(y_true, y_pred)
                        vm_avg += Vm

                        fit, complex = fit_complex.\
                                        get_metrics_from_simulation(df_variants, 
                                                                    k_markov)
                    
                        fit_avg += fit

                    ari_avg /= reps
                    vm_avg /= reps
                    fit_avg /= reps

                    if ari_avg > best_ARI:
                        print('best_ARI: ', ari_avg)
                        # best_fit = fit_avg
                        best_ARI = ari_avg
                        best_ARI_list = ARI_list.copy()
                        best_VM = vm_avg
                        best_clusters = n_clusters
                        best_ngram = n
                        best_min_percents = min_perc
                        best_max_percents = max_perc
                        best_df_distrib = df_distrib.copy()
                        best_dict_var = dict_var.copy()
                        best_dict_gd = dict_gd.copy()
                        # best_complex = complex

                    if fit > best_fit:
                        best_fit = fit

                        print()
                        print('best fit: ' + str(best_fit))
                        print('params best fit:')
                        print('clusters: ' + str(n_clusters))
                        print('ngram: ' + str(n))
                        print('min_percents: ' + str(min_perc))
                        print('max_percents: ' + str(max_perc))
                        print()
                        
                    if complex < best_complex:
                        best_complex = complex

                        print()
                        print('best complex: ' + str(best_complex))
                        print('params best complex:')
                        print('clusters: ' + str(n_clusters))
                        print('ngram: ' + str(n))
                        print('min_percents: ' + str(min_perc))
                        print('max_percents: ' + str(max_perc))
                        print()



    print('best params')
    print('best_ARI: ' + str(best_ARI))
    print('best_VM: ' + str(best_VM))
    print('best_clusters: ' + str(best_clusters))
    print('best_ngram: ' + str(best_ngram))
    print('best_min_percents: ' + str(best_min_percents))
    print('best_max_percents: ' + str(best_max_percents))
    print('best_fit: ' + str(best_fit))
    print('best_complex: ' + str(best_complex))


    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(ARI) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write('V-measure (Vm): ' + str(Vm) + '\n\n')
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

    print('done!')

