import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from experiments.log.my_filter_variants import most_frequent_and_random_filter
from experiments.log.discover_dfg import discover_dfg
from pm4py.objects.log.importer.xes import importer as xes_importer


df_ref = pd.read_csv(
            'experiments/results/metrics_mixed_dataset.csv',
            sep='\t'
        )

etm_path = 'experiments/aux/filt_log_etm/'
filt_logs_etm = [f for f in listdir(etm_path) if isfile(join(etm_path, f))]

def get_filtered_log(log_name, log, alg):
    df_params = df_ref[(df_ref['EVENT_LOG'] == log_name) & \
                       (df_ref['DISCOVERY_ALG'] == alg)]
    
    print('filter_freq: ' + str(float(df_params['FILTER_FREQ'])))
    print('filter_rand: ' + str(float(df_params['FILTER_RAND'])))

    return most_frequent_and_random_filter(log, 
                float(df_params['FILTER_FREQ']), 
                float(df_params['FILTER_RAND']))


def get_filtered_weighted_graph(log_name, log, alg):
    if alg != 'ETM':
        filt_log = get_filtered_log(log_name, log, alg)
        G_log = discover_dfg(filt_log)
    else:
        if log_name in filt_logs_etm:
            filt_log = xes_importer.apply(etm_path + log_name)
            G_log = discover_dfg(filt_log)
        else:
            filt_log = log
            G_log = discover_dfg(log)

    # print('size of log after: ' + str(len(filt_log)))

    return (G_log, filt_log)


if __name__ == '__main__':
    # base_path = 'xes_files/real_processes/set_for_simulations/2/'
    # log_name = 'BPI_Challenge_2014_Inspection.xes.gz'

    base_path = 'xes_files/real_processes/set_for_simulations/1/'
    log_name = '3a_VARA_DE_FAMILIA_E_REGISTRO_CIVIL_DA_COMARCA_DE_OLINDA_-_TJPE.xes'


    log = xes_importer.apply(base_path + log_name)
    alg = 'ETM'

    print('size of log before: ' + str(len(log)))
    (G_log, filt_log) = get_filtered_weighted_graph(log_name, log, alg)
    


