from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy 
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.AgglomClust import AglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/variant_analysis/exp5/exp5.xes'
    exp_backlog = 'experiments/variant_analysis/exp5/results/1step_lev_agg.txt'
    exp_dendrogram = 'experiments/variant_analysis/exp5/results/dendrograms/'+\
                     '1step_lev_agg.png'
    log = xes_importer.apply(log_path)

    method = [
        # 'single',
        # 'complete',
        # 'average',
        'weighted',
        # 'centroid',
    ]

    ## Calculate fitness and complexity
    fit_complex = FindFitnessComplexity()
    k_markov = 2

    best_ARI = -1
    best_Vm = -1
    best_perc = -1
    best_method = None
    best_y_pred = []


    for m in method:

        print('method: ', m)

        # convert to df
        df = convert_to_dataframe(log)

        # extract variants from log
        split_join = SplitJoinDF(df)
        traces = split_join.split_df()
        ids = split_join.ids
        ids_clus = [l[0] for l in ids]
        df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

        # cluster using only agglommerative with levenshtein distance
        agglomClust = AglomClust(traces)
        Z = agglomClust.cluster(method=m)

        utils = Utils()
        stats = StatsClust()
        y_true = utils.get_ground_truth(ids_clus)

        t = max(Z[:,2])

        min_perc = 0.3
        max_perc = 0.9
        step_perc = 0.025
        perc = min_perc

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

        while perc <= max_perc:
            labels = hierarchy.fcluster(Z=Z, t=perc*t, criterion='distance')

            # get variants by cluster
            dict_var = stats.get_variants_by_cluster(traces, labels)

            # get performance by adjusted rand-score metric
            
            y_pred = utils.get_agglom_labels_by_trace(traces, dict_var, labels)

            ARI = adjusted_rand_score(y_true, y_pred)
            Vm = v_measure_score(y_true, y_pred)

            if ARI > best_ARI:
                best_ARI = ARI
                best_Vm = Vm
                best_perc = perc
                best_y_pred = y_pred.copy()
                best_method = m

            perc += step_perc
    

    labels = hierarchy.fcluster(Z=Z, t=best_perc*t, criterion='distance')
    labels = [l-1 for l in labels]

    # get df-log with cluster labels
    split_join.join_df(labels)
    df = split_join.df

    # get distribution of traces cluster per variant
    stats = StatsClust()
    df_distrib = stats.get_distrib(df, df_ids)

    # get variants by cluster
    dict_var = stats.get_variants_by_cluster(traces, labels)

    # get variants ground truth by cluster
    dict_gd = stats.get_ground_truth_by_cluster(dict_var, traces, y_true)

    best_ARI = round(best_ARI, 4)
    best_Vm = round(best_Vm, 4)

    # write results to s3

    # content = 'time: ' + str(round(end - start,4)) + '\n\n'
    content = 'Percent dendrogram cut: ' + str(best_perc) + '\n\n'
    content += 'ARI: ' + str(best_ARI) + '\n\n'
    content += 'V-measure: ' + str(best_Vm) + '\n\n'
    content += 'Method: ' + best_method + '\n\n'
    content += 'traces: ' + str(traces) + '\n\n'
    content += 'y_true: ' + str(y_true) + '\n\n'
    content += 'y_pred: ' + str(best_y_pred) + '\n\n'
    content += 'df_distrib: ' + df_distrib.to_string() + '\n\n'
    content += 'dict_gd: ' + str(dict_gd) + '\n\n'

    with open(exp_backlog, 'w') as f:
        f.write(content)

    t = max(Z[:,2])

    hierarchy.dendrogram(Z, color_threshold=best_perc*t)
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
