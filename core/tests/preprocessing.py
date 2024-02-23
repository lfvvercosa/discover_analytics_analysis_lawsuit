import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe


if __name__ == "__main__": 
    # log_path = 'dataset/tribunais_eleitorais/tre-ne.xes'
    loop_log_path = 'dataset/tribunais_eleitorais/LOOP_tre-ne.xes'

    # log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    # df_log = convert_to_dataframe(log)

    log_loop = xes_importer.apply(loop_log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    df_log_loop = convert_to_dataframe(log_loop)

    # df_log = df_log.groupby('case:concept:name').agg(trace=('concept:name',list))
    df_log_loop = df_log_loop.groupby('case:concept:name').agg(trace=('concept:name',list))

    # df_log.to_csv('temp/df_log.csv', sep='\t')
    df_log_loop.to_csv('temp/df_log_loop.csv', sep='\t')

    print('done!')

