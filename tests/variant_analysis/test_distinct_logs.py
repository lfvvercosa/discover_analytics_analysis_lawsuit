import pandas as pd

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py import convert_to_dataframe


CASE_ID = 'case:concept:name'
ACT_LABEL = 'concept:name'

path_p1v1 = 'xes_files/test_variants/p1_v1.xes'
log_p1v1 = xes_importer.apply(path_p1v1)

df_p1v1 = convert_to_dataframe(log_p1v1)
df_p1v1 = df_p1v1.groupby(CASE_ID).agg({ACT_LABEL:list})
df_p1v1[ACT_LABEL] = df_p1v1[ACT_LABEL].astype(str)
df_p1v1 = df_p1v1.drop_duplicates([ACT_LABEL])
df_p1v1['log'] = 'P1V1'


path_p1v2 = 'xes_files/test_variants/p1_v2.xes'
log_p1v2 = xes_importer.apply(path_p1v2)

df_p1v2 = convert_to_dataframe(log_p1v2)
df_p1v2 = df_p1v2.groupby(CASE_ID).agg({ACT_LABEL:list})
df_p1v2[ACT_LABEL] = df_p1v2[ACT_LABEL].astype(str)
df_p1v2 = df_p1v2.drop_duplicates([ACT_LABEL])
df_p1v2['log'] = 'P1V2'


path_p1v3 = 'xes_files/test_variants/p1_v3.xes'
log_p1v3 = xes_importer.apply(path_p1v3)

df_p1v3 = convert_to_dataframe(log_p1v3)
df_p1v3 = df_p1v3.groupby(CASE_ID).agg({ACT_LABEL:list})
df_p1v3[ACT_LABEL] = df_p1v3[ACT_LABEL].astype(str)
df_p1v3 = df_p1v3.drop_duplicates([ACT_LABEL])
df_p1v3['log'] = 'P1V3'

path_p2v1 = 'xes_files/test_variants/p2_v1.xes'
log_p2v1 = xes_importer.apply(path_p2v1)

df_p2v1 = convert_to_dataframe(log_p2v1)
df_p2v1 = df_p2v1.groupby(CASE_ID).agg({ACT_LABEL:list})
df_p2v1[ACT_LABEL] = df_p2v1[ACT_LABEL].astype(str)
df_p2v1 = df_p2v1.drop_duplicates([ACT_LABEL])
df_p2v1['log'] = 'P2V1'

path_p2v2 = 'xes_files/test_variants/p2_v2.xes'
log_p2v2 = xes_importer.apply(path_p2v2)

df_p2v2 = convert_to_dataframe(log_p2v2)
df_p2v2 = df_p2v2.groupby(CASE_ID).agg({ACT_LABEL:list})
df_p2v2[ACT_LABEL] = df_p2v2[ACT_LABEL].astype(str)
df_p2v2 = df_p2v2.drop_duplicates([ACT_LABEL])
df_p2v2['log'] = 'P2V2'


df_all = pd.concat([
    df_p1v1,
    df_p1v2,
    df_p1v3,
    df_p2v1,
    df_p2v2,
])

df_temp = df_all.groupby(ACT_LABEL).agg({'log':list})


print(df_temp[df_temp['log'].str.len() > 1])
print()
