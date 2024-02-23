from pathlib import Path
from experiments.variant_analysis.approaches.run.\
     literature.actitrac.ActiTraCConnector import ActiTracConnector

from pm4py.objects.log.importer.xes import importer as xes_importer

import pandas as pd


def create_path_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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
    jar_path = 'temp/actitrac/actitrac.jar'
    actitrac = ActiTracConnector(jar_path)
    name_approach = 'ActiTraC_ics'
    
    k_markov = 2

    number_of_clusters = 3
    # is_greedy = True
    # dist_greed = 0.25
    # target_fit = 1
    # min_clus_size = 0.25

    target_fit_params = [0.5, 0.75, 1]
    is_greedy_params = [True, False]
    dist_greed_params = [0.1, 0.25]
    min_clus_size_params = [0.1, 0.25, 0.4]

    total_runs = len(target_fit_params) * \
                 len(is_greedy_params) * \
                 len(dist_greed_params) * \
                 len(min_clus_size_params) * \
                 len(log_complexity) * \
                 log_total
    count_runs = 0

    for log_complex in log_complexity:
        metrics['ARI'][log_complex] = []
        metrics['Fitness'][log_complex] = []
        metrics['Complexity'][log_complex] = []

        for i in range(1,log_total):

            best_fit = 0
            best_ARI = 0
            best_complx = float('inf')

            for target_fit in target_fit_params:
                for is_greedy in is_greedy_params:
                    for dist_greed in dist_greed_params:
                        for min_clus_size in min_clus_size_params:


                            print(log_complex + ', ' + str(i))
                            print('progress: ' + str(round(count_runs/total_runs,2)))

                            log_path = 'xes_files/variant_analysis/exp7/'
                            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'

                            backlog_path = 'experiments/variant_analysis/exp7/'
                            backlog_path += log_size + log_complex + '/' + str(i) + \
                                            '/' + name_approach + '.txt'

                            saving_path = 'temp/actitrac/'
                            saving_path += log_size + log_complex + str(i) + '/'
                            create_path_dir(saving_path)

                            ARI, fit, complx = actitrac.run_actitrac(number_of_clusters,
                                                                    is_greedy,
                                                                    dist_greed,
                                                                    target_fit,
                                                                    min_clus_size,
                                                                    log_path,
                                                                    saving_path)
                            
                            if fit > best_fit:
                                best_fit = fit
                                best_ARI = ARI
                                best_complx = complx

                            count_runs += 1

            metrics['ARI'][log_complex].append(best_ARI)
            metrics['Fitness'][log_complex].append(best_fit)
            metrics['Complexity'][log_complex].append(best_complx)

    
    for m in metrics:
        results_path = 'experiments/variant_analysis/exp7/'
        results_path += log_size + 'results/ics_fitness/' + m + \
                       '_' + name_approach + '.csv'
        
        df_res = pd.DataFrame.from_dict(metrics[m])
        df_res.to_csv(results_path, sep='\t', index=False)

    


