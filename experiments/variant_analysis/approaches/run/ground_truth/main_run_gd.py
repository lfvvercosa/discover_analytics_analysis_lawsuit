from RunGroundTruth import RunGroundTruth
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
        'Fitness':{},
        'Complexity':{},
    }

    run_gd = RunGroundTruth()

    for log_complex in log_complexity:
        metrics['Fitness'][log_complex] = []
        metrics['Complexity'][log_complex] = []

        for i in range(log_total):
            print(log_complex + ', ' + str(i))

            log_path = 'xes_files/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'
            log = xes_importer.apply(log_path)
            
            fit, complx = run_gd.run(log)

            metrics['Fitness'][log_complex].append(fit)
            metrics['Complexity'][log_complex].append(complx)

    for m in metrics:
        results_path = 'experiments/variant_analysis/exp7/' + log_size + 'results/ics_fitness/'
        results_path += m + '_gd.csv'
        
        df_res = pd.DataFrame.from_dict(metrics[m])
        df_res.to_csv(results_path, sep='\t', index=False)

    
    print('done!')