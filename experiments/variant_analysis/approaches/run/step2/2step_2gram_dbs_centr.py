from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy 
import matplotlib.pyplot as plt

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.clustering.StatsClust import StatsClust

import pandas as pd


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp4/exp4.xes'
    log = xes_importer.apply(log_path)
    exp_backlog = 'experiments/variant_analysis/exp4/results/1step_1gram_dbs_centroid.txt'

    

    # ngrams = [1,2]
    # min_percentage = [0, 0.05, 0.1]
    # max_percentage = [1, 0.95, 0.9]
    # epsilon = [0.2, 0.3, 0.7, 1.2, 1.7]
    # percentage = [0.2, 0.4, 0.6, 0.8]
    # reps = 3

    ngrams = [1]
    min_percentage = [0]
    max_percentage = [1]
    epsilon = [0.5]
    percentage = list(range(0, 900, 25))
    percentage = [x/1000 for x in percentage]
    reps = 1

    best_ARI = -1
    best_VM = -1
    best_ngram = -1
    best_epsilon = -1
    best_percentage = -1
    best_min_perc = -1
    best_max_perc = -1

    total = len(ngrams) * len(min_percentage) * len(max_percentage) * \
            len(epsilon) * len(percentage)
    count = 0
    stats = StatsClust()

    for n in ngrams:
        for min_perc in min_percentage:
            for max_perc in max_percentage:
                for eps in epsilon:
                    for perc in percentage:
                        ari_avg = 0
                        vm_avg = 0

                        if count % 10 == 0:
                            print('progress: ' + \
                                  str(round((count/total)*100,2)) + '%\n')

                        count += 1

                        for i in range(reps):
                            # convert to df
                            df = convert_to_dataframe(log)
                            # create n-gram
                            split_join = SplitJoinDF(df)
                            traces = split_join.split_df()
                            ids = split_join.ids
                            ids_clus = [l[0] for l in ids]
                            df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
                            df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

                            df_gram = create_n_gram(df_clus, 'case:concept:name', 'concept:name', n)
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

                            # apply first-step clustering (DBScan) 
                            min_samples = 1
                            
                            model = DBSCAN(eps = eps, 
                                           min_samples = min_samples, 
                                           metric='euclidean')
                            cluster_labels = model.fit_predict(df_gram_norm)

                            print(list(set(cluster_labels)))
                            print(cluster_labels)


                            # get df-log with cluster labels
                            split_join.join_df(cluster_labels)
                            df = split_join.df
                            dict_labels = {'index':ids_clus, 'cluster_label':cluster_labels}
                            df_labels = pd.DataFrame.from_dict(dict_labels)
                            df_labels = df_labels.set_index('index')
                            df_gram_norm = df_gram_norm.join(df_labels, how='left')

                            if i == reps - 1:

                                # get distribution of traces cluster per variant
                                df_distrib = stats.get_distrib(df, df_ids)

                                # print()

                            dict_var = stats.get_variants_by_cluster(traces, cluster_labels)
                            # get centroids of each cluster
                            df_centroids = df_gram_norm.groupby('cluster_label').mean()
                            df_centroids.to_csv('temp/df_centroids_dbs.csv', sep='\t')

                            if len(df_centroids) == 1:
                                break

                            Z = hierarchy.linkage(df_centroids, 
                                                  method='centroid', 
                                                  metric='euclidean')
                            t = max(Z[:,2])

                            hierarchy.dendrogram(Z, color_threshold=perc*t)
                            plt.show(block=True)
                            plt.close()

                            # get best number of clusters
                            
                            labels = hierarchy.fcluster(Z=Z, 
                                                        t=perc*t, 
                                                        criterion='distance')

                            # get performance by adjusted rand-score metric
                            utils = Utils()
                            y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)
                            y_true = utils.get_ground_truth(ids_clus)
                            # y_true = [0 if x < 20 else 1 for x in y_true]

                            ARI = adjusted_rand_score(y_true, y_pred)
                            ari_avg += ARI
                    
                            Vm = v_measure_score(y_true, y_pred)
                            vm_avg += Vm
                        
                        ari_avg /= reps
                        vm_avg /= reps

                        if ari_avg > best_ARI:
                            best_ARI = ari_avg
                            best_VM = vm_avg
                            best_ngram = n
                            best_epsilon = eps
                            best_percentage = perc
                            best_min_perc = min_perc
                            best_max_perc = max_perc


    print('best params:')
    print('best_ARI: ' + str(best_ARI))
    print('best_VM: ' + str(best_VM))
    print('best_ngram: ' + str(best_ngram))
    print('best_epsilon: ' + str(best_epsilon))
    print('best_percentage: ' + str(best_percentage))
    print('best_min_perc: ' + str(best_min_perc))
    print('best_max_perc: ' + str(best_max_perc))



    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(ARI) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write(df_distrib.to_string(header=True, index=True) + '\n\n')
        f.write('dict_var: \n\n')
        f.write(str(dict_var) + '\n\n')

    print('done!')

    # apply first-step clustering (DBScan) 
    min_samples = 1

    df = convert_to_dataframe(log)
    # create n-gram
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    df_gram = create_n_gram(df_clus, 'case:concept:name', 'concept:name', best_ngram)
    df_gram.index = pd.Categorical(df_gram.index, ids_clus)
    df_gram = df_gram.sort_index()

    preProcess = PreProcessClust()
    rem_cols = preProcess.selectColsByFreq(best_min_perc, best_max_perc, df_gram)
    df_gram = df_gram.drop(columns=rem_cols)

    # df_gram.to_csv('temp/df_gram.csv', sep='\t')

    # normalize n-gram
    df_gram_norm = (df_gram - df_gram.min())/\
                (df_gram.max() - df_gram.min())
    df_gram_norm = df_gram_norm.round(4)

    
    model = DBSCAN(eps = best_epsilon, 
                   min_samples = min_samples, 
                   metric='euclidean')
    cluster_labels = model.fit_predict(df_gram_norm)

    print(list(set(cluster_labels)))
    print(cluster_labels)


    # get df-log with cluster labels
    split_join.join_df(cluster_labels)
    df = split_join.df
    dict_labels = {'index':ids_clus, 'cluster_label':cluster_labels}
    df_labels = pd.DataFrame.from_dict(dict_labels)
    df_labels = df_labels.set_index('index')
    df_gram_norm = df_gram_norm.join(df_labels, how='left')

    # get distribution of traces cluster per variant
    df_distrib = stats.get_distrib(df, df_ids)


    dict_var = stats.get_variants_by_cluster(traces, cluster_labels)
    # get centroids of each cluster
    df_centroids = df_gram_norm.groupby('cluster_label').mean()
    df_centroids.to_csv('temp/df_centroids_dbs.csv', sep='\t')

    Z = hierarchy.linkage(df_centroids, 
                            method='centroid', 
                            metric='euclidean')
    t = max(Z[:,2])

    hierarchy.dendrogram(Z, color_threshold=best_percentage*t)
    plt.show(block=True)
    plt.close()

