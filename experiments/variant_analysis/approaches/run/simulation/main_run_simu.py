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

    clusters = [2,3,4,5,6,7]
    ngram = [1,2]
    min_percents = [0, 0.1, 0.2, 0.3]
    max_percents = [1, 0.9, 0.8, 0.7]
    reps = 1
    k_markov = 2
    number_of_clusters = 3

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

            backlog_path = 'experiments/variant_analysis/exp7/results/'
            backlog_path += log_size + log_complex + '/' + str(i) + \
                            '/1step_ngram_kms.txt'

            ARI, fit, complx = run_kms.run(log,
                                        clusters,
                                        ngram,
                                        min_percents,
                                        max_percents,
                                        backlog_path,
                                        reps,
                                        k_markov,
                                        number_of_clusters)

            metrics['ARI'][log_complex].append(ARI)
            metrics['Fitness'][log_complex].append(fit)
            metrics['Complexity'][log_complex].append(complx)

    
    for m in metrics:
        results_path = 'experiments/variant_analysis/exp7/results/'
        results_path += log_size + log_complex + '/' + m + \
                       '_1step_ngram_kms.csv'
        
        df_res = pd.DataFrame.from_dict(metrics[m])
        df_res.to_csv(results_path, sep='\t')

    
    print('done!')