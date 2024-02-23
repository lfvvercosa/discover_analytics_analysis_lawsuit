from experiments.variant_analysis.approaches.run.\
     step1.kmeans.RunKmsNgram import RunKmeansNgram

from pm4py.objects.log.importer.xes import importer as xes_importer

import pandas as pd


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

    number_of_clusters = 3
    clusters = [number_of_clusters]
    ngram = [1,2]
    min_percents = [0, 0.1, 0.2, 0.3]
    max_percents = [1, 0.9, 0.8, 0.7]
    reps = 1
    k_markov = 2
    name_file = '1step_ngram_kms_ics'

    run_kms = RunKmeansNgram()


    for log_complex in log_complexity:
        metrics['ARI'][log_complex] = []
        metrics['Fitness'][log_complex] = []
        metrics['Complexity'][log_complex] = []

        for i in range(log_total):

            print(log_complex + ', ' + str(i))

            log_path = 'xes_files/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'
            log = xes_importer.apply(log_path)

            backlog_path = 'experiments/variant_analysis/exp7/'
            backlog_path += log_size + log_complex + '/' + str(i) + \
                            '/' + name_file + '.txt'

            ARI, fit, complx, _, _, _, \
            _, _, _ = run_kms.run(log,
                                  clusters,
                                  ngram,
                                  min_percents,
                                  max_percents,
                                  backlog_path,
                                  reps,
                                  k_markov)

            metrics['ARI'][log_complex].append(ARI)
            metrics['Fitness'][log_complex].append(fit)
            metrics['Complexity'][log_complex].append(complx)

    
    for m in metrics:
        results_path = 'experiments/variant_analysis/exp7/'
        results_path += log_size + 'results/ics_fitness/' + m + \
                       '_' + name_file + '.csv'
        
        df_res = pd.DataFrame.from_dict(metrics[m])
        df_res.to_csv(results_path, sep='\t', index=False)

    
    print('done!')