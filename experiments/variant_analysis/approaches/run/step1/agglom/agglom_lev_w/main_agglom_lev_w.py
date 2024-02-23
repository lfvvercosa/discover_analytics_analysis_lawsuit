from experiments.variant_analysis.approaches.run.\
     step1.agglom.agglom_lev_w.AgglomLevWeight import AgglomLevWeight
from utils.read_and_write import s3_handle

from pm4py.objects.log.importer.xes import importer as xes_importer

import pandas as pd
import sys
import boto3
from ec2_metadata import ec2_metadata


if __name__ == '__main__':
    log_size = 'size10/'
    log_complexity = [
       'low_complexity',
       'medium_complexity',
       'high_complexity', 
    ]
    # log_complexity = [sys.argv[1]]
    log_total = 10
    metrics = {
        'ARI':{},
        'Fitness':{},
        'Complexity':{},
    }

    send_to_s3 = False
    bucket = 'luiz-doutorado-projetos2'
    filename = 'results_' + log_complexity[0] + '.txt'
    fullname = 'variant_analysis/exp7/agglom_lev_w/' + filename

    weight_parallel = [0, 1, 2, 3]
    weight_new_act = [0, 1, 2, 3]
    weight_subst = [0, 1, 2, 3]
    agglom_method = ['single','complete','average']

    number_of_clusters = 3
    agglom_lev_w = AgglomLevWeight()
    name_file = '1step_agglom_lev_w_ics'

    for log_complex in log_complexity:
        metrics['ARI'][log_complex] = []
        metrics['Fitness'][log_complex] = []
        metrics['Complexity'][log_complex] = []

        for i in range(1,log_total):

            print(log_complex + ', ' + str(i))

            log_path = 'xes_files/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'
            log = xes_importer.apply(log_path)

            backlog_path = 'experiments/variant_analysis/exp7/'
            backlog_path += log_size + log_complex + '/' + str(i) + \
                            '/' + name_file + '.txt'
            
            ARI, fit, complx = agglom_lev_w.run(log, 
                                                number_of_clusters, 
                                                weight_parallel,
                                                weight_new_act,
                                                weight_subst,
                                                agglom_method)
            
            metrics['ARI'][log_complex].append(ARI)
            metrics['Fitness'][log_complex].append(fit)
            metrics['Complexity'][log_complex].append(complx)

    if send_to_s3:
        content = 'ARI:\n\n'
        content += str(metrics['ARI']) + '\n\n'

        content += 'Fitness:\n\n'
        content += str(metrics['Fitness']) + '\n\n'

        content += 'Complexity:\n\n'
        content += str(metrics['Complexity']) + '\n\n'

        s3_handle.write_to_s3(bucket = bucket, 
                              filename = fullname, 
                              file_content = content)


    for m in metrics:
        results_path = 'experiments/variant_analysis/exp7/'
        results_path += log_size + 'results/ics_fitness/' + m + \
                       '_' + name_file + '.csv'
        
        df_res = pd.DataFrame.from_dict(metrics[m])
        df_res.to_csv(results_path, sep='\t', index=False)

    
    print('done!')

    instance_id = ec2_metadata.instance_id
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)

    print('id: ' + str(instance))
    print('shutdown: ' + str(instance.terminate()))

