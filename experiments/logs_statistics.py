from pm4py.objects.log.importer.xes import importer as xes_importer
from features import LogFeatures

from os import listdir
from os.path import isfile, join
import pandas as pd


base_path = 'xes_files/candidates/'
out_path = 'experiments/results/'
all_files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

res = {
    'ID':[],
    'EVENT_LOG':[],
    '#TRACES':[],
    '#VARIANTS':[],
    '#ACTIVITIES':[],
    'AVG_TRACE_LENGTH':[],
    'AVG_DISTINCT_ACTS':[],
    'AVG_NON_OVERLAP_ACTS':[],
}
count = 1


for f in all_files:
    print('calc stats for ' + f + '...')
    log = xes_importer.apply(base_path + f)

    res['ID'].append(count)
    res['EVENT_LOG'].append(f)
    res['#TRACES'].append(len(log))
    res['#VARIANTS'].append(LogFeatures.number_unique_seqs(log))
    res['#ACTIVITIES'].append(LogFeatures.number_events_types(log))
    res['AVG_TRACE_LENGTH'].append(LogFeatures.avg_trace_size(log))
    res['AVG_DISTINCT_ACTS'].append(LogFeatures.avg_dist_act(log))
    res['AVG_NON_OVERLAP_ACTS'].\
        append(LogFeatures.avg_non_overlap_traces(log))

    count += 1


df_stats = pd.DataFrame.from_dict(res)
df_stats.to_csv(out_path + 'stats_logs_candidates.csv', sep='\t')

print('done!')
