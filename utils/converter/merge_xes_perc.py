import pandas as pd
import random

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe


CASE_ID = 'case:concept:name'


def random_filter_log(df, perc):
    df_work = df.drop_duplicates(CASE_ID)
    case_ids = df_work[CASE_ID].to_list()
    sampled_ids = random.sample(range(0, len(case_ids)), int(perc*len(case_ids)))
    sampled = []

    for id in sampled_ids:
        sampled.append(case_ids[id])

    df_filter = pd.DataFrame.from_dict({CASE_ID:sampled})
    df_sampled = df.merge(df_filter, how='inner', on=CASE_ID)


    return df_sampled


path_p2v1 = 'xes_files/test_variants/exp4/exp4_p2_v1.xes'
path_p2v2 = 'xes_files/test_variants/exp4/exp4_p2_v2.xes'

perc_p2v1 = 0.3

out_path = 'xes_files/test_variants/exp4/exp4_p2v1_p2v2_perc.xes'

log_p2v1 = xes_importer.apply(path_p2v1)
log_p2v2 = xes_importer.apply(path_p2v2)

df_p2v1 = convert_to_dataframe(log_p2v1)
df_p2v2 = convert_to_dataframe(log_p2v2)

df_p2v1['case:concept:name'] += '_p2v1'
df_p2v1['case:cluster'] = 'P2V1'
df_p2v1['time:timestamp'] += pd.Timedelta(days=365*1)

df_p2v2['case:concept:name'] += '_p2v2'
df_p2v2['case:cluster'] = 'P2V2'
df_p2v2['time:timestamp'] += pd.Timedelta(days=365*2)

df_p2v1_sampled = random_filter_log(df_p2v1, perc=perc_p2v1)

df_all = pd.concat([
    df_p2v1_sampled,
    df_p2v2
])

log_all = pm4py.convert_to_event_log(df_all)
pm4py.write_xes(log_all, out_path)

print('done!')