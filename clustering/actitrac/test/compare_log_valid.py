from pathlib import Path
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd
import pm4py


path = 'temp/actitrac/valid/results/'
path_valid = 'temp/actitrac/valid/log_valid.xes'
directory = Path(path)
xes_paths = [path + f.name for f in directory.iterdir() if f.is_file()]
df_all = []

for xes in xes_paths:
    log = xes_importer.apply(xes, variant=xes_importer.Variants.LINE_BY_LINE)
    df = pm4py.convert_to_dataframe(log)
    df_all.append(df)

df = pd.concat(df_all)

log_valid = xes_importer.apply(path_valid, variant=xes_importer.Variants.LINE_BY_LINE)
df_valid = pm4py.convert_to_dataframe(log_valid)

df_only_valid = df_valid[~df_valid['case:concept:name'].isin(df['case:concept:name'])]
df_only_result = df[~df['case:concept:name'].isin(df_valid['case:concept:name'])]

print('size of df_valid: ' + str(len(df_valid.index)))
print('size of df_result: ' + str(len(df.index)))
print('size of df_only_valid: ' + str(len(df_only_valid.index)))
print('size of df_only_result: ' + str(len(df_only_result.index)))