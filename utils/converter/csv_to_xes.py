import pandas as pd
import os

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from parameters import DEBUG

def convert_csv_to_xes(path_csv,
                       path_xes,
                       case_id_col,
                       act_col,
                       timestamp_col,
                       csv_sep=','):
    
    df = pd.read_csv(path_csv, sep=csv_sep)

    if DEBUG:
        print('df sample:')
        print(df)

    df = df[[case_id_col, act_col, timestamp_col]]

    if DEBUG:
        print('converting timestamp columns...')

    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    df = df.sort_values(timestamp_col)

    parameters = \
        {log_converter.Variants.TO_EVENT_LOG.\
            value.Parameters.CASE_ID_KEY: case_id_col}

    if DEBUG:
        print('converting CSV to log...')

    log = log_converter.apply(df, 
                        parameters=parameters, 
                        variant=log_converter.Variants.TO_EVENT_LOG)

    if DEBUG:
        print('exporting log to XES...')

    xes_exporter.apply(log, path_xes)   

    if DEBUG:
        print('compressing XES file with gzip...')

    os.system('gzip ' + path_xes)

    if DEBUG:
        print('done!')

    
    