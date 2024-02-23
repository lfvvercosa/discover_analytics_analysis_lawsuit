import pandas as pd

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe

from experiments.variant_analysis.exp5.MergeXes import MergeXes

path_p1_v1 = 'xes_files/test_variants/exp5/exp5_0.xes'
path_p1_v2 = 'xes_files/test_variants/exp5/exp5_1.xes'
path_p1_v3 = 'xes_files/test_variants/exp5/exp5_2.xes'
path_p2_v1 = 'xes_files/test_variants/exp5/exp5_3.xes'
path_p2_v2 = 'xes_files/test_variants/exp5/exp5_4.xes'

out_path = 'xes_files/test_variants/exp5/exp5.xes'

log_p1v1 = xes_importer.apply(path_p1_v1)
log_p1v2 = xes_importer.apply(path_p1_v2)
log_p1v3 = xes_importer.apply(path_p1_v3)
log_p2v1 = xes_importer.apply(path_p2_v1)
log_p2v2 = xes_importer.apply(path_p2_v2)

# df_p1v1 = 
#             df['case:concept:name'] += '_p1v' + str(count)
#             df['case:cluster'] = 'P1V' + str(count)
#             df['time:timestamp'] += pd.Timedelta(days=365*count_offset) 




# log_all = pm4py.convert_to_event_log(df_all)
# pm4py.write_xes(log_all, out_path)

# print('done!')