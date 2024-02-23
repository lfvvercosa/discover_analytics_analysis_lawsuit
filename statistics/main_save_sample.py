import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

import pandas as pd
import re
from pathlib import Path

import core.my_loader as my_loader
from core import my_create_features
from core import my_stats


if __name__ == "__main__": 
    log_path = 'dataset/tribunais_trabalho/TRT.xes'
    out_path = 'dataset/tribunais_trabalho/TRT_mini.xes'
    frac = 0.1
    # frac = 1
    name = 'TRT - Mini'

    print('Processing...')

    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    df_log = convert_to_dataframe(log)
    df_log["lifecycle:transition"] =  "complete"

    df_ids = df_log[['case:concept:name']].drop_duplicates()
    df_ids_sample = df_ids.sample(frac=frac)

    df_log_sample = df_ids_sample.merge(df_log, on='case:concept:name', how='left')
    log_sample = pm4py.convert_to_event_log(df_log_sample)
    log_sample._attributes['concept:name'] = name


    pm4py.write_xes(log_sample, out_path)

    print('done!')


