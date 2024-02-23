import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from scipy.cluster import hierarchy 
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import boto3
import time
import requests
import sys

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.DBScanClust import DBScanClust
from experiments.variant_analysis.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.clustering.create_n_gram import create_n_gram
from experiments.clustering.PreProcessClust import PreProcessClust
from experiments.variant_analysis.MarkovMeasures import MarkovMeasures
import utils.read_and_write.s3_handle as s3_handle



def cash_df_grams(df_clus, n_grams, minimum_percents, maximum_percents):
    cashed_df_grams = {}

    for n in n_grams:
        cashed_df_grams[n] = {}

        for min_perc in minimum_percents:
            cashed_df_grams[n][min_perc] = {}

            for max_perc in maximum_percents:
                df_gram = create_n_gram(df_clus, 
                                        'case:concept:name', 
                                        'concept:name', 
                                        n)
                df_gram.index = pd.Categorical(df_gram.index, ids_clus)
                df_gram = df_gram.sort_index()

                preProcess = PreProcessClust()
                rem_cols = preProcess.selectColsByFreq(min_perc, max_perc, df_gram)
                df_gram = df_gram.drop(columns=rem_cols)

                # normalize n-gram
                df_gram_norm = (df_gram - df_gram.min())/\
                            (df_gram.max() - df_gram.min())
                df_gram_norm = df_gram_norm.round(4)

                cashed_df_grams[n][min_perc][max_perc] = df_gram_norm.copy()
    
    
    return cashed_df_grams


def calc_overall_fitness(df_labels,
                             cluster_labels, 
                             min_pts, 
                             weight_clus, 
                             my_percent,
                             k,
                             fit_func):

    count = Counter(cluster_labels)
    bigger_clusters = [c for c in count.most_common() if c[1] >= min_pts]
    total = sum([c[1] for c in bigger_clusters])
    weight_fit = 0

    for c in bigger_clusters:
        df_log = df_labels[df_labels['cluster_label'] == c[0]]
        log = pm4py.convert_to_event_log(df_log)
        markov_measure = MarkovMeasures(log, k)

        if fit_func == 'std':
            fit = markov_measure.get_fitness_gaussian(n=my_percent)
        elif fit_func == 'mean':
            fit = markov_measure.get_fitness_mean(p=my_percent)
        else:
            fit = 100/markov_measure.get_network_complexity()

        weight_fit += fit * c[1]

    if total == 0:
        return -1

    weight_fit /= total
    weight_fit -= weight_clus * len(bigger_clusters)


    return weight_fit


def run_dbscan_with_params(df_gram_norm,
                           eps, 
                           df_clus, 
                           ids_clus,
                           min_pts,
                           weight_clus,
                           std,
                           k,
                           fit_func
                          ):

    # apply clustering (DBScan) 
    min_samples = 1
    
    model = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean')
    cluster_labels = model.fit_predict(df_gram_norm)

    # get df with cluster labels
    df_labels = pd.DataFrame.from_dict({'case:concept:name':ids_clus,
                                        'cluster_label':cluster_labels})
    df_labels = df_clus.merge(df_labels, on='case:concept:name', how='inner')

    stats = StatsClust()
    # dict_var = stats.get_variants_by_cluster(traces, cluster_labels)

    return calc_overall_fitness(df_labels,
                                    cluster_labels,
                                    min_pts,
                                    weight_clus,
                                    std,
                                    k,
                                    fit_func), \
           cluster_labels
           


if __name__ == '__main__':

    if len(sys.argv) > 1:
        fit_func = sys.argv[1]
    else:
        fit_func = 'std'

    # load event-log
    log_path = 'xes_files/test_variants/exp4/exp4.xes'
    exp_backlog = {}
    bucket = 'luiz-doutorado-projetos2'
    local_file = 'experiments/variant_analysis/exp4/results/1step_ngram_dbs_' + \
        fit_func + '.txt'
    filename = 'variant_analysis/exp4/results/1step_ngram_dbs_' + \
        fit_func + '.txt'
    results_file = 'variant_analysis/exp4/results/results_1step_ngram_dbs_' + \
        fit_func + '.txt'
    content = ""

    params_agglom = {}
    params_agglom['AWS_bucket'] = bucket
    params_agglom['AWS_filename'] = \
        'variant_analysis/exp4/results/1step_ngram_dbs' + fit_func + '.txt'


    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)

    # convert to df
    df = convert_to_dataframe(log)
    
    split_join = SplitJoinDF(df)
    traces = split_join.split_df()
    ids = split_join.ids
    ids_clus = [l[0] for l in ids]
    df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})
    df_clus = df.merge(df_ids, on='case:concept:name',how='inner')

    # get ground-truth
    utils = Utils()
    y_true = utils.get_ground_truth(ids_clus)

    # optimize params

    minimum_pts = [1, 4, 7]
    weight_cluster_number = [0, 0.01, 0.001]
    measurements = [0.2, 0.3, 0.4]
    k_markov = [2, 3]

    # minimum_pts = [3]
    # weight_cluster_number = [0.01]
    # standard_deviations = [0.2]
    # k_markov = [2]

    # n_grams = [1, 2]
    # minimum_percents = [0, 0.1, 0.2]
    # maximum_percents = [0.8, 0.9, 1]
    # epsilons = [0.3, 0.6, 0.9, 1.2, 1.5]

    n_grams = [1, 2]
    minimum_percents = [0, 0.1]
    maximum_percents = [0.9, 1]
    epsilons = [0.3, 0.6, 0.9]

    total = len(minimum_pts) * len(weight_cluster_number) * \
            len(measurements) * len(k_markov) * len(n_grams) * \
            len(minimum_percents) * len(maximum_percents) * len(epsilons)
    

    best_min_pts = -1
    best_weight_cluster_number = -1
    best_standard_deviations = -1
    best_k_markov = -1
    ALL_best_Fit = -1

    count = 0
    firstTime = True

    cashed_df_grams = cash_df_grams(df_clus, n_grams, minimum_percents, maximum_percents)
    results = {
        'min_pts':[],
        'weight_clus':[],
        'meas_perc':[],
        'k':[],
        'n':[],
        'min_perc':[],
        'max_perc':[],
        'eps':[],
        'Fit':[],
        'ARI':[],
        'Vm':[],
    }

    for min_pts in minimum_pts:
        for weight_clus in weight_cluster_number:
            for my_meas in measurements:
                for k in k_markov:
                   
                    best_n = -1
                    best_min_perc = -1
                    best_max_perc = -1
                    best_eps = -1
                    best_Fit = -1
                    best_ARI = -1
                    best_Vm = -1

                    for n in n_grams:
                        for min_perc in minimum_percents:
                            for max_perc in maximum_percents:
                                for eps in epsilons:

                                    if count % 10 == 0:
                                        progress = str(round((count/total)*100,2)) + '%\n'
                                        
                                        s3_handle.write_to_s3(params_agglom['AWS_bucket'],
                                          params_agglom['AWS_filename'],
                                          progress
                                         )
                                        print('progress: ' + progress)

                                    df_gram_norm = cashed_df_grams[n][min_perc][max_perc]

                                    Fit, labels = run_dbscan_with_params(
                                                        df_gram_norm,
                                                        eps,
                                                        df_clus,
                                                        ids_clus,
                                                        min_pts,
                                                        weight_clus,
                                                        my_meas,
                                                        k,
                                                        fit_func
                                                   )

                                    Fit = round(Fit, 4)
                                    ARI = round(adjusted_rand_score(y_true, labels),4)
                                    Vm = round(v_measure_score(y_true, labels),4)

                                    results['min_pts'].append(min_pts)
                                    results['weight_clus'].append(weight_clus)
                                    results['meas_perc'].append(my_meas)
                                    results['k'].append(k)
                                    results['n'].append(n)
                                    results['min_perc'].append(min_perc)
                                    results['max_perc'].append(max_perc)
                                    results['eps'].append(eps)
                                    results['Fit'].append(Fit)
                                    results['ARI'].append(ARI)
                                    results['Vm'].append(Vm)

                                    if Fit > best_Fit:
                                        # print('best Fit: ' + str(Fit))
                                        best_Fit = Fit
                                        best_n = n
                                        best_min_perc = min_perc
                                        best_max_perc = max_perc
                                        best_eps = eps

                                        best_ARI = adjusted_rand_score(y_true, labels)
                                        best_Vm = v_measure_score(y_true, labels)

                                    count += 1

                    if firstTime:
                        openMode = 'w'
                        firstTime = False
                    else:
                        openMode = 'a+'

                    with open(local_file, openMode) as f:
                        f.write('min_pts: ' + str(min_pts) + '\n')
                        f.write('weight_clus: ' + str(weight_clus) + '\n')
                        f.write('meas: ' + str(my_meas) + '\n')
                        f.write('k: ' + str(k) + '\n')
                        f.write('best_n: ' + str(best_n) + '\n')
                        f.write('best_min_perc: ' + str(best_min_perc) + '\n')
                        f.write('best_max_perc: ' + str(best_max_perc) + '\n')
                        f.write('best_eps: ' + str(best_eps) + '\n')

                        f.write('Fit Markov: ' + str(best_Fit) + '\n')
                        f.write('ARI: ' + str(best_ARI) + '\n')
                        f.write('Vm: ' + str(best_Vm) + '\n\n')

                    if best_Fit > ALL_best_Fit:
                        ALL_best_Fit = best_Fit
                        best_min_pts = min_pts
                        best_weight_cluster_number = weight_clus
                        best_standard_deviations = my_meas
                        best_k_markov = k

                        print('all_best_fit: ' + str(round(ALL_best_Fit,2)))


    with open(local_file, 'a+') as f:
        f.write('#### Best Config #### \n')
        f.write('min_pts: ' + str(best_min_pts) + '\n')
        f.write('weight_clus: ' + str(best_weight_cluster_number) + '\n')
        f.write('std: ' + str(best_standard_deviations) + '\n')
        f.write('k: ' + str(best_k_markov) + '\n')
        f.write('Fit Markov: ' + str(ALL_best_Fit) + '\n')


    with open(local_file, 'r') as file:
        content = file.read()

    s3_handle.write_to_s3(bucket = bucket, 
                          filename = filename, 
                          file_content = content)

    print('done!')

    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv('df_results.csv', sep='\t', index=False)

    with open('df_results.csv', 'r') as file:
        content = file.read()

    s3_handle.write_to_s3(bucket = bucket, 
                          filename = results_file, 
                          file_content = content)

    # shutdown ec2 instance

    response = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
    instance_id = response.text

    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)

    print('id: ' + str(instance))
    print('shutdown: ' + str(instance.terminate()))




    # # run with best params
    # best_Fit, df_gram_norm, cluster_labels = \
    #     run_dbscan_with_params(best_n, 
    #                            best_min_perc, 
    #                            best_max_perc, 
    #                            best_eps,
    #                            df_clus,
    #                            ids_clus)

    # # get df-log with cluster labels
    # split_join.join_df(cluster_labels)
    # df = split_join.df
    # dict_labels = {'index':ids_clus, 'cluster_label':cluster_labels}
    # df_labels = pd.DataFrame.from_dict(dict_labels)
    # df_labels = df_labels.set_index('index')
    # df_gram_norm = df_gram_norm.join(df_labels, how='left')

    # # get distribution of traces cluster per variant
    # stats = StatsClust()
    # df_distrib = stats.get_distrib(df, df_ids)

    # # get variants by cluster
    # dict_var = stats.get_variants_by_cluster(traces, cluster_labels)
    # print()

    # # get variants ground truth by cluster
    # dict_gd = stats.get_ground_truth_by_cluster(dict_var, traces, y_true)

    # # get time needed
    # start = time.time()    

    # # get performance by adjusted rand-score metric
    # utils = Utils()
    # y_pred = cluster_labels
    # y_true = utils.get_ground_truth(ids_clus)

    # ARI = adjusted_rand_score(y_true, y_pred)
    # Vm = v_measure_score(y_true, y_pred)


    # with open(exp_backlog, 'w') as f:
    #     f.write('Adjusted Rand Score (ARI): ' + str(ARI) + '\n\n')
    #     f.write('df_distrib: \n\n')
    #     f.write(df_distrib.to_string(header=True, index=True) + '\n\n')
    #     f.write('dict_var: \n\n')
    #     f.write(str(dict_var) + '\n\n')

