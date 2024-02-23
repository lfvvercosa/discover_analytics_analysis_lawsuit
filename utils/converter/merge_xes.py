import pandas as pd

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe


path_p1v1 = 'xes_files/test_variants/exp2/p1_v1.xes'
path_p1v2 = 'xes_files/test_variants/exp2/p1_v2.xes'
path_p1v3 = 'xes_files/test_variants/exp2/p1_v3.xes'
path_p1v4 = 'xes_files/test_variants/exp2/p1_v4.xes'
path_p1v5 = 'xes_files/test_variants/exp2/p1_v5.xes'
path_p1v6 = 'xes_files/test_variants/exp2/p1_v6.xes'
path_p1v7 = 'xes_files/test_variants/exp2/p1_v7.xes'
path_p1v8 = 'xes_files/test_variants/exp2/p1_v8.xes'
path_p2v1 = 'xes_files/test_variants/exp2/p2_v1.xes'
path_p2v2 = 'xes_files/test_variants/exp2/p2_v2.xes'

out_path = 'xes_files/test_variants/exp2/p1_v2v5.xes'

log_p1v1 = xes_importer.apply(path_p1v1)
log_p1v2 = xes_importer.apply(path_p1v2)
log_p1v3 = xes_importer.apply(path_p1v3)
log_p1v4 = xes_importer.apply(path_p1v4)
log_p1v5 = xes_importer.apply(path_p1v5)
log_p1v6 = xes_importer.apply(path_p1v6)
log_p1v7 = xes_importer.apply(path_p1v7)
log_p1v8 = xes_importer.apply(path_p1v8)

log_p2v1 = xes_importer.apply(path_p2v1)
log_p2v2 = xes_importer.apply(path_p2v2)

df_p1v1 = convert_to_dataframe(log_p1v1)
df_p1v2 = convert_to_dataframe(log_p1v2)
df_p1v3 = convert_to_dataframe(log_p1v3)
df_p1v4 = convert_to_dataframe(log_p1v4)
df_p1v5 = convert_to_dataframe(log_p1v5)
df_p1v6 = convert_to_dataframe(log_p1v6)
df_p1v7 = convert_to_dataframe(log_p1v7)
df_p1v8 = convert_to_dataframe(log_p1v8)

df_p2v1 = convert_to_dataframe(log_p2v1)
df_p2v2 = convert_to_dataframe(log_p2v2)

# df_p1v1['case:concept:name'] += '_p1v1'
# df_p1v1['case:cluster'] = 'P1V1'
# df_p1v1['time:timestamp'] += pd.Timedelta(days=0)


df_p1v2['case:concept:name'] += '_p1v2'
df_p1v2['case:cluster'] = 'P1V2'
df_p1v2['time:timestamp'] += pd.Timedelta(days=365)

# df_p1v3['case:concept:name'] += '_p1v3'
# df_p1v3['case:cluster'] = 'P1V3'
# df_p1v3['time:timestamp'] += pd.Timedelta(days=365*2)

df_p1v4['case:concept:name'] += '_p1v4'
df_p1v4['case:cluster'] = 'P1V4'
df_p1v4['time:timestamp'] += pd.Timedelta(days=365*5)

df_p1v5['case:concept:name'] += '_p1v5'
df_p1v5['case:cluster'] = 'P1V5'
df_p1v5['time:timestamp'] += pd.Timedelta(days=365*6)

df_p1v6['case:concept:name'] += '_p1v6'
df_p1v6['case:cluster'] = 'P1V6'
df_p1v6['time:timestamp'] += pd.Timedelta(days=365*7)

df_p1v7['case:concept:name'] += '_p1v7'
df_p1v7['case:cluster'] = 'P1V7'
df_p1v7['time:timestamp'] += pd.Timedelta(days=365*8)

df_p1v8['case:concept:name'] += '_p1v8'
df_p1v8['case:cluster'] = 'P1V8'
df_p1v8['time:timestamp'] += pd.Timedelta(days=365*9)


# df_p2v1['case:concept:name'] += '_p2v1'
# df_p2v1['case:cluster'] = 'P2V1'
# df_p2v1['time:timestamp'] += pd.Timedelta(days=365*3)

# df_p2v2['case:concept:name'] += '_p2v2'
# df_p2v2['case:cluster'] = 'P2V2'
# df_p2v2['time:timestamp'] += pd.Timedelta(days=365*4)

df_all = pd.concat([
    # df_p1v1,
    df_p1v2,
    # df_p1v3,
    # df_p1v4,
    df_p1v5,
    # df_p1v6,
    # df_p1v7,
    # df_p1v8,
    # df_p2v1,
    # df_p2v2,
])

log_all = pm4py.convert_to_event_log(df_all)
pm4py.write_xes(log_all, out_path)

print('done!')