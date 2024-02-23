from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pm4py
import pandas as pd
from pathlib import Path

import core.my_loader as my_loader
from clustering.actitrac.ActiTraCConnector import ActiTracConnector


def create_path_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__": 
    base_path = 'dataset/'
    log_path = 'dataset/tribunais_eleitorais/tre-ne.xes'
    DEBUG = True

    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    
    # Update log
    # log._attributes['concept:name'] = 'TRE-NE'
    # pm4py.write_xes(log, log_path)

    # df_log = convert_to_dataframe(log)

    number_of_clusters_params = [5, 10, 15]
    target_fit_params = [0.5, 0.75, 1]
    min_clus_size_params = [0.05]
    is_greedy_params = [True]
    dist_greed_params = [0.1]

    total_runs = len(target_fit_params) * \
                 len(is_greedy_params) * \
                 len(dist_greed_params) * \
                 len(min_clus_size_params) * \
                 len(number_of_clusters_params)
    count_runs = 0

    best_fit = float('-inf')
    best_complx = float('inf')
    best_params = {}

    jar_path = 'temp/actitrac/actitrac.jar'
    actitrac = ActiTracConnector(jar_path)

    for number_of_clusters in number_of_clusters_params:
        for target_fit in target_fit_params:
            for is_greedy in is_greedy_params:
                for dist_greed in dist_greed_params:
                    for min_clus_size in min_clus_size_params:
                        print('progress: ' + str(round(count_runs/total_runs,2)))

                        saving_path = 'temp/actitrac/tre_ne/'
                        create_path_dir(saving_path)

                        fit, complx = actitrac.run(number_of_clusters,
                                                            is_greedy,
                                                            dist_greed,
                                                            target_fit,
                                                            min_clus_size,
                                                            log_path,
                                                            saving_path)
                        
                        if fit > best_fit:
                            best_fit = fit
                            best_complx = complx

                            best_params['n_clusters'] = number_of_clusters
                            best_params['target_fit'] = target_fit

                            print('Current best fitness: ' + str(best_fit))

                        count_runs += 1

    print('### Best fitness: ' + str(best_fit))
    print('### Best complexity: ' + str(best_complx))
    print('### Best Params: ' + str(best_params))


