import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe


CASE_ID = 'case:concept:name'
ACT_LABEL = 'concept:name'

path_p1v1 = 'xes_files/test_variants/exp4/exp4_p2_v2.xes'
log_p1v1 = xes_importer.apply(path_p1v1)

df = convert_to_dataframe(log_p1v1)
df = df.groupby(CASE_ID).agg({ACT_LABEL:list})
df['concept:name'] = df['concept:name'].str.join('')
print(df)
print('size: ' + str(len(df)))

