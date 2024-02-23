from pathlib import Path
from experiments.variant_analysis.approaches.run.\
     literature.actitrac.ActiTraCConnector import ActiTracConnector
from experiments.variant_analysis.approaches.run.\
     step2.actitrac.ActiTracAgglom import ActiTracAgglom

from pm4py.objects.log.importer.xes import importer as xes_importer

import pandas as pd
import os
import glob


def create_path_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def remove_previous_files(path):
    files = glob.glob(path + '*')
    for f in files:
        os.remove(f)


if __name__ == '__main__':

    # Gerar estrutura de diretório para salvar os logs
    # create_path_dir(path, log_size, log_complexity, log_total)
    # Run ActiTrac for each log

    log_size = 'size10/'
    log_complexity = [
        'low_complexity/',
        'medium_complexity/',
        'high_complexity/', 
    ]
    log_total = 10
    metrics = {
        'ARI':{},
        'Fitness':{},
        'Complexity':{},
    }
    actitrac_agglom = ActiTracAgglom()
    name_approach = 'ActiTraC_Agglom_ics'
    
    number_of_clusters = 3
    number_of_trials_1step = 3
    target_fit_params = [0.5, 0.75, 1]
    is_greedy_params = [True]
    dist_greed_params = [0.1]

    total_runs = len(target_fit_params) * \
                 len(is_greedy_params) * \
                 len(log_complexity) * \
                 log_total
    count_runs = 0

    for log_complex in log_complexity:
        metrics['ARI'][log_complex] = []
        metrics['Fitness'][log_complex] = []
        metrics['Complexity'][log_complex] = []

        for i in range(log_total):

            log_path = 'xes_files/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'

            saving_path = 'temp/actitrac/'
            saving_path += log_size + log_complex + str(i) + '/'
            create_path_dir(saving_path)
            remove_previous_files(saving_path)

            ARI, fit, complx = actitrac_agglom.run(number_of_clusters,
                                                   is_greedy_params,
                                                   dist_greed_params,
                                                   target_fit_params,
                                                   log_path,
                                                   saving_path,
                                                   number_of_trials_1step,
                                                   number_of_clusters
                                                   )

            metrics['ARI'][log_complex].append(ARI)
            metrics['Fitness'][log_complex].append(fit)
            metrics['Complexity'][log_complex].append(complx)

    
    for m in metrics:
        results_path = 'experiments/variant_analysis/exp7/'
        results_path += log_size + 'results/ics_fitness/' + m + \
                       '_' + name_approach + '.csv'
        
        df_res = pd.DataFrame.from_dict(metrics[m])
        df_res.to_csv(results_path, sep='\t', index=False)

    


