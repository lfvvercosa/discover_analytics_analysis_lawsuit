from experiments.variant_analysis.approaches.run.\
     step2.crossfitness.ngram.KmsNgramAgglom import KmsNgramAgglom

from pm4py.objects.log.importer.xes import importer as xes_importer

import pm4py
import pandas as pd

from pathlib import Path


if __name__ == '__main__':
    log_size = 'size10/'
    log_complexity = [
       'low_complexity',
       'medium_complexity',
       'high_complexity', 
    ]
    log_total = 10
    metrics = {
        'ARI':{},
        'Fitness':{},
        'Complexity':{},
    }
    DEBUG = True

    number_of_trials_1step = 3
    number_of_clusters = 3
    clusters = [number_of_clusters]
    
    # ngram = [1,2]
    ngram = [2]
    # min_percents = [0, 0.1, 0.2]
    # max_percents = [1, 0.9, 0.8]
    min_percents = [0.2]
    max_percents = [0.8]
    reps = 1
    k_markov = 2
    params_agglom = {}
    params_agglom['custom_distance'] = 'alignment_between_logs'
    tec_name = '2step_kms_agglom_ics'

    run_kms_agglom  = KmsNgramAgglom()

    for log_complex in log_complexity:
        metrics['ARI'][log_complex] = []
        metrics['Fitness'][log_complex] = []
        metrics['Complexity'][log_complex] = []

        for i in range(log_total):

            print(log_complex + ', ' + str(i))

            log_path = 'xes_files/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'

            # if DEBUG:
            #     log_path = 'xes_files/variant_analysis/exp3/exp3_p1_v2.xes'

            log = xes_importer.apply(log_path)

            backlog_path = 'experiments/variant_analysis/exp7/'
            backlog_path += log_size + log_complex + '/' + str(i) + \
                            '/' 
            
            Path(backlog_path).mkdir(parents=True, exist_ok=True)

            backlog_path += tec_name 
            
            ARI, fit, complx, log_clusters = run_kms_agglom.run(log,
                                                                clusters,
                                                                ngram,
                                                                min_percents,
                                                                max_percents,
                                                                backlog_path,
                                                                reps,
                                                                k_markov,
                                                                number_of_trials_1step,
                                                                number_of_clusters,
                                                                params_agglom
                                                                )
            
            metrics['ARI'][log_complex].append(ARI)
            metrics['Fitness'][log_complex].append(fit)
            metrics['Complexity'][log_complex].append(complx)

            log_path = 'experiments/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + \
                            '/' + tec_name + '/'
            count = 0

            # for l in log_clusters:
            #     pm4py.write_xes(l, log_path + 'log_' + str(count) + '.xes')
            #     count += 1


        for m in metrics:
            results_path = 'experiments/variant_analysis/exp7/'
            results_path += log_size + 'results/ics_fitness/' + m + \
                            tec_name + '.csv'
            
            df_res = pd.DataFrame.from_dict(metrics[m])
            df_res.to_csv(results_path, sep='\t', index=False)

    
    print('done!')