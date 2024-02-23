from pm4py.objects.log.importer.xes import importer as xes_importer
from features import LogFeatures


log_path = 'xes_files/tests/log_pn_parallel.xes'
log_path2 = 'xes_files/tests/log_pn_parallel2.xes'
log_path3 = 'xes_files/tests/log_pn_parallel3.xes'

log = xes_importer.apply(log_path)
val = LogFeatures.avg_dist_act(log)
val2 = LogFeatures.avg_trace_size(log)

# ((3*2) + (2*3) + (1*1))/6
# 2.17

print('avg dist act: ' + str(val))

# ((3*3) + (3*2) + (1*2))/6
# 2.83

print('avg trace size: ' + str(val2))

log = xes_importer.apply(log_path2)
val = LogFeatures.avg_non_overlap_traces(log)
 
# 1 - ((3 * 3 * 1) + 2*(3 * 2 * 2/3) + (2 * 2 * 1)) / 25
# 0.16

print('avg non overlap traces: ' + str(val))

log = xes_importer.apply(log_path3)
val = LogFeatures.avg_non_overlap_traces(log)

# >> 0.16
print('avg non overlap traces2: ' + str(val))



