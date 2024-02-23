import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

import pandas as pd


if __name__ == '__main__':
    log_size = 'size10/'
    log_complexity = [
    #    'low_complexity',
       'medium_complexity',
       'high_complexity', 
    ]
    log_total = 10

    for log_complex in log_complexity:
        for i in range(1,log_total):
            print(log_complex + ', ' + str(i))

            log_path = 'xes_files/variant_analysis/exp7/'
            log_path += log_size + log_complex + '/' + str(i) + '/log.xes'
            log = xes_importer.apply(log_path)
            
            df_log = pm4py.convert_to_dataframe(log)

            if df_log.isnull().values.any():
                print(df_log.columns[df_log.isna().any()].tolist())
                raise ValueError('df_log contains null values!')

    
    print('Not any null values!')
