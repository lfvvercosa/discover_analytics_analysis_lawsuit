from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt
import pandas as pd

from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from experiments.variant_analysis.approaches.core.DBScanClust import DBScanClust
from experiments.variant_analysis.approaches.core.CustomAgglomClust import CustomAgglomClust
from experiments.variant_analysis.utils.Utils import Utils
from experiments.clustering.StatsClust import StatsClust
from experiments.variant_analysis.approaches.core.metrics.FindFitnessComplexity \
    import FindFitnessComplexity


if __name__ == '__main__':
    # load event-log
    log_path = 'xes_files/test_variants/exp5/exp5.xes'
    exp_backlog = 'experiments/variant_analysis/exp5/results/1step_lev_dbs.txt'
    log = xes_importer.apply(log_path)

    list_ARI = []
    simus = 3
    count_simu = 0

    ## Calculate fitness and complexity
    fit_complex = FindFitnessComplexity()
    k_markov = 2


    while count_simu < simus:
        count_simu += 1

        eps_list = [1,2,3,4,5,6,7,8]
        best_ARI = float('-inf')
        best_Vm = float('-inf')
        best_eps = -1
        best_distrib = None
        best_dict_var = None
    
        for eps in eps_list:
            # convert to df
            df = convert_to_dataframe(log)

            # extract variants from log
            split_join = SplitJoinDF(df)
            traces = split_join.split_df()
            ids = split_join.ids
            ids_clus = [l[0] for l in ids]
            df_ids = pd.DataFrame.from_dict({'case:concept:name':ids_clus})

            # apply clustering (DBScan)
            min_samples = 1
            dbscan = DBScanClust(traces)
            labels = dbscan.cluster(eps, min_samples)

            print(list(set(labels)))

            # get df-log with cluster labels
            split_join.join_df(labels)
            df = split_join.df
            df_variants = split_join.join_df_uniques(labels)

            # get distribution of traces cluster per variant
            stats = StatsClust()
            df_distrib = stats.get_distrib(df, df_ids)

            # get variants by cluster
            dict_var = stats.get_variants_by_cluster(traces, labels)

            # get performance by adjusted rand-score metric
            utils = Utils()
            y_pred = labels
            y_true = utils.get_ground_truth(ids_clus)

            ARI = adjusted_rand_score(y_true, y_pred)
            Vm = v_measure_score(y_true, y_pred)
        
            if ARI > best_ARI:
                print('best ARI: ' + str(ARI))
                print('best eps: ', eps)

                best_ARI = ARI
                best_Vm = Vm
                best_eps = eps
                best_distrib = df_distrib
                best_dict_var = dict_var

            
            clusters_number = 5
            variants_number = len(df_variants['case:concept:name'].drop_duplicates())
            min_size = int(variants_number/clusters_number)
            k_markov = 2

            logs = fit_complex.find_best_match_clusters(clusters_number,
                                                        min_size,
                                                        log,
                                                        traces,
                                                        labels)
            
            if logs:
                fit, complex = fit_complex.get_metrics(logs, k_markov)

            

        list_ARI.append(best_ARI)


    with open(exp_backlog, 'w') as f:
        f.write('Adjusted Rand Score (ARI): ' + str(best_ARI) + '\n\n')
        f.write('V-measure (ARI): ' + str(best_Vm) + '\n\n')
        f.write('eps: ' + str(best_eps) + '\n\n')
        f.write('df_distrib: \n\n')
        f.write(best_distrib.to_string(header=True, index=True) + '\n\n')
        f.write('dict_var: \n\n')
        f.write(str(best_dict_var) + '\n\n')

    print('done!')
