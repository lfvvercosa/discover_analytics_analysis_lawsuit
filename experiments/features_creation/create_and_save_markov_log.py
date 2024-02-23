import time
import json
import pickle
import boto3
from os import listdir
from os.path import isfile, join, exists
from pm4py.objects.log.importer.xes import importer as xes_importer
from utils.converter.markov.create_markov_log_2 import create_mk_abstraction_log_2


def write_to_s3(bucket, filename, file_content):
    s3 = boto3.resource('s3')
    object = s3.Object(bucket, 
                       filename)
    object.put(Body=file_content)


if __name__ == '__main__':
    base_path = 'xes_files/'
    k_markov = 3
    folders = ['1/', '2/', '3/', '4/', '5/']
    s3_bucket = 'luiz-doutorado-projetos2'
    s3_dir_res = 'testes_markov/resultados_markov_log/k_'
    saving_path = 'experiments/features_creation/markov_log/k_' + \
                  str(k_markov) + '/'
    results = []

    for fol in folders:
        my_path = base_path + fol
        my_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

        for fil in my_files:

            print('### Current folder: ' + str(fol))
            print('### Current log: ' + str(fil))

            log_path = my_path + fil
            log = xes_importer.apply(log_path)

            start = time.time()
            G_log = create_mk_abstraction_log_2(log=log, k=k_markov)
            end = time.time()
            markov_log_time = round(end - start, 4)

            results.append(
                            {
                                'event_log':fil,
                                'total_time':markov_log_time,
                            }
                        )
            
            sav_path = saving_path + fil + '.txt'
            pickle.dump(G_log, open(sav_path, 'wb'))

            filename = s3_dir_res + str(k_markov)  + '/' + \
                           'res' + fol[:-1] + '.json'
            file_content = json.dumps(results)
            write_to_s3(s3_bucket, filename, file_content)




