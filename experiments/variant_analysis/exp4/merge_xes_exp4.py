import pandas as pd

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe


path_p1_v1 = 'xes_files/test_variants/exp4/exp4_p1_v1.xes'
path_p1_v2 = 'xes_files/test_variants/exp4/exp4_p1_v2.xes'
path_p1_v3 = 'xes_files/test_variants/exp4/exp4_p1_v3.xes'
path_p2_v1 = 'xes_files/test_variants/exp4/exp4_p2_v1.xes'
path_p2_v2 = 'xes_files/test_variants/exp4/exp4_p2_v2.xes'

out_path = 'xes_files/test_variants/exp4/exp4_p1.xes'

log_p1v1 = xes_importer.apply(path_p1_v1)
log_p1v2 = xes_importer.apply(path_p1_v2)
log_p1v3 = xes_importer.apply(path_p1_v3)

log_p2v1 = xes_importer.apply(path_p2_v1)
log_p2v2 = xes_importer.apply(path_p2_v2)

df_p1v1 = convert_to_dataframe(log_p1v1)
df_p1v2 = convert_to_dataframe(log_p1v2)
df_p1v3 = convert_to_dataframe(log_p1v3)

df_p2v1 = convert_to_dataframe(log_p2v1)
df_p2v2 = convert_to_dataframe(log_p2v2)


df_p1v1['case:concept:name'] += '_p1v1'
df_p1v1['case:cluster'] = 'P1V1'

df_p1v2['case:concept:name'] += '_p1v2'
df_p1v2['case:cluster'] = 'P1V2'
df_p1v2['time:timestamp'] += pd.Timedelta(days=365)

df_p1v3['case:concept:name'] += '_p1v3'
df_p1v3['case:cluster'] = 'P1V3'
df_p1v3['time:timestamp'] += pd.Timedelta(days=365*2)

df_p2v1['case:concept:name'] += '_p2v1'
df_p2v1['case:cluster'] = 'P2V1'
df_p2v1['time:timestamp'] += pd.Timedelta(days=365*3)

df_p2v2['case:concept:name'] += '_p2v2'
df_p2v2['case:cluster'] = 'P2V2'
df_p2v2['time:timestamp'] += pd.Timedelta(days=365*4)

df_all = pd.concat([
    df_p1v1,
    df_p1v2,
    df_p1v3,
    # df_p2v1,
    # df_p2v2,
])

log_all = pm4py.convert_to_event_log(df_all)
pm4py.write_xes(log_all, out_path)

print('done!')