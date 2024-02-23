from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.AgglomClustNgram \
    import AglomClustNgram
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity


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
                y_true = utils.get_ground_truth(ids_clus)

                df_gram = create_n_gram(df_clus, 'case:concept:name', 'concept:name', n)
                df_gram = rename_columns(df_gram)
                df_gram.index = pd.Categorical(df_gram.index, ids_clus)
                df_gram = df_gram.sort_index()

                preProcess = PreProcessClust()
                rem_cols = preProcess.selectColsByFreq(min_perc, max_perc, df_gram)
                df_gram = df_gram.drop(columns=rem_cols)

                # df_gram.to_csv('temp/df_gram.csv', sep='\t')

                # normalize n-gram
                df_gram_norm = df_gram.copy()

                for c in df_gram_norm.columns:
                    if df_gram_norm[c].max() != df_gram_norm[c].min():
                        df_gram_norm[c] = (df_gram_norm[c] - df_gram_norm[c].min())/\
                            (df_gram_norm[c].max() - df_gram_norm[c].min())

                df_gram_norm = df_gram_norm.round(4)

                cashed_dfs[n][min_perc][max_perc] = df_gram_norm.copy()


    return cashed_dfs


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp5/exp5.xes'
    exp_backlog = 'experiments/variant_analysis/exp5/results/1step_ngram_agg.txt'
    exp_dendrogram = 'experiments/variant_analysis/exp5/results/dendrograms/'+\
                     '1step_ngram_agg.png'
    exp_df_gram = 'experiments/variant_analysis/exp5/results/dendrograms/'+\
                  'df_gram.csv'
    log = xes_importer.apply(log_path)

    # n_grams = [1,2]
    # minimum_percents = [0, 0.1, 0.2]
    # maximum_percents = [1, 0.9, 0.8]
    # method = [
    #     'single',
    #     'complete',
    #     'average',
    #     'weighted',
    # ]
    # metric = [
    #     'euclidean',
    #     'hamming',
    #     'jaccard',
    #     # 'cosine',
    # ]

    n_grams = [2]
    minimum_percents = [0]
    maximum_percents = [1]
    method = [
        # 'single',
        # 'complete',
        # 'average',
        'weighted',
    ]
    metric = [
        # 'euclidean',
        'hamming',
        # 'jaccard',
        # 'cosine',
    ]

    total = len(minimum_percents) * \
            len(maximum_percents) * \
            len(method) * \
            len(n_grams) * \
            len(metric)
    count_run = 0

    ## Calculate fitness and complexity
    fit_complex = FindFitnessComplexity()
    k_markov = 2

    best_ARI = -1
    best_Vm = -1
    best_perc = -1
    best_n = -1
    best_min_perc = -1
    best_max_perc = -1
    best_method = None
    best_metric = None
    best_Z = None
    best_y_pred = []
    best_df_gram = None
    best_df_distrib = None
    best_dict_gd = None

    cashed_dfs = cashe_df_grams(log, 
                                n_grams, 
                                minimum_percents, 
                                maximum_percents)

    for n in n_grams:
        for min_perc in minimum_percents:
            for max_perc in maximum_percents:
                for md in method:
                    for mc in metric:
                        count_run += 1

                        if count_run % 10 == 0:
                            print('progress: ' + \
                                  str(round((count_run/total)*100,2)) + '%\n')
                            
                        # convert to df
                        df = convert_to_dataframe(log)

                        # extract variants from log
                        split_join = SplitJoinDF(df)
                        traces = split_join.split_df()
                        ids = split_join.ids
                        ids_clus = [l[0] for l in ids]
                        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

                        # cluster using only agglommerative with levenshtein distance
                        agglomClust = AglomClustNgram()
                        Z = agglomClust.cluster(cashed_dfs[n][min_perc][max_perc],
                                                method=md,
                                                metric=mc)

                        utils = Utils()
                        stats = StatsClust()
                        y_true = utils.get_ground_truth(ids_clus)

                        t = max(Z[:,2])

                        min_perc_dendro = 0.3
                        max_perc_dendro = 0.9
                        step_perc = 0.025
                        perc = min_perc_dendro

                        clusters_number = 5
                        variants_number = len(traces)
                        min_size = int(variants_number/clusters_number)
                        k_markov = 2

                        logs = fit_complex.find_best_match_clusters_hier(Z, 
                                                                clusters_number, 
                                                                min_size, 
                                                                log, 
                                                                traces,
                                                                None
                                                                )
                        
                        if logs:
                            fit, complex = fit_complex.get_metrics(logs, k_markov)

                        while perc <= max_perc_dendro:
                            labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')
                            labels = [l-1 for l in labels]

                            # get variants by cluster
                            dict_var = stats.get_variants_by_cluster(traces, labels)

                            # get performance by adjusted rand-score metric
                            
                            # y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

                            ARI = adjusted_rand_score(y_true, labels)
                            Vm = v_measure_score(y_true, labels)

                            if ARI > best_ARI:
                                print('best ARI: ', ARI)

                                best_ARI = ARI
                                best_Vm = Vm
                                best_perc = perc
                                best_y_pred = labels.copy()
                                best_method = md
                                best_metric = mc
                                best_Z = Z.copy()
                                best_n = n
                                best_min_perc = min_perc
                                best_max_perc = max_perc
                                best_df_gram = cashed_dfs[n][min_perc][max_perc].copy()

                                # convert to df
                                df = convert_to_dataframe(log)

                                # extract variants from log
                                split_join = SplitJoinDF(df)
                                traces = split_join.split_df()
                                ids = split_join.ids
                                ids_clus = [l[0] for l in ids]
                                df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

                                # get df-log with cluster labels
                                split_join.join_df(labels)
                                df = split_join.df

                                # get distribution of traces cluster per variant
                                stats = StatsClust()
                                df_distrib = stats.get_distrib(df, 
                                                               df_ids)

                                # get variants by cluster
                                dict_var = stats.get_variants_by_cluster(traces, 
                                                                         labels)

                                # get variants ground truth by cluster
                                dict_gd = stats.get_ground_truth_by_cluster(dict_var, 
                                                                            traces, 
                                                                            y_true)
                                
                                best_df_distrib = df_distrib.copy()
                                best_dict_gd = dict_gd.copy()

                            perc += step_perc
    
    labels = hierarchy.fcluster(Z=best_Z, t=best_perc*t, criterion='distance')
    labels = [l-1 for l in labels]

    best_ARI = round(best_ARI, 4)
    best_Vm = round(best_Vm, 4)

    # write results to s3

    # content = 'time: ' + str(round(end - start,4)) + '\n\n'
    content = 'Percent dendrogram cut: ' + str(best_perc) + '\n\n'
    content += 'ARI: ' + str(best_ARI) + '\n\n'
    content += 'V-measure: ' + str(best_Vm) + '\n\n'
    content += 'Method: ' + best_method + '\n\n'
    content += 'Metric: ' + best_metric + '\n\n'
    content += 'Ngram: ' + str(best_n) + '\n\n'
    content += 'Min-perc: ' + str(best_min_perc) + '\n\n'
    content += 'Max-perc: ' + str(best_max_perc) + '\n\n'
    content += 'traces: ' + str(traces) + '\n\n'
    content += 'y_true: ' + str(y_true) + '\n\n'
    content += 'y_pred: ' + str(best_y_pred) + '\n\n'
    content += 'df_distrib: ' + best_df_distrib.to_string() + '\n\n'
    content += 'dict_gd: ' + str(best_dict_gd) + '\n\n'

    with open(exp_backlog, 'w') as f:
        f.write(content)

    t = max(best_Z[:,2])

    best_df_gram.to_csv(exp_df_gram, sep='\t', index=False)

    hierarchy.dendrogram(best_Z, color_threshold=best_perc*t)
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    plt.savefig(exp_dendrogram, dpi=400)
    # plt.show(block=True)
    plt.close()

    # traces ids
    # trace_id = {}

    # for i in range(len(traces)):
    #     trace_id[i] = traces[i]

    # with open(exp_backlog, 'w') as f:
    #     f.write('Adjusted Rand Score (ARI): ' + str(ARI) + '\n\n')
    #     f.write('df_distrib: \n\n')
    #     f.write(df_distrib.to_string(header=True, index=True) + '\n\n')
    #     f.write('dict_var: \n\n')
    #     f.write(str(dict_var) + '\n\n')
    #     f.write('dend: \n\n')
    #     f.write(str(Z) + '\n\n')
    #     f.write('traces: \n\n')
    #     f.write(str(trace_id) + '\n\n')

    print('done!')
