from utils.creation import create_filtered_logs

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.traces.generic.log import case_statistics


log_paths = [
    'xes_files/tests/log_pn_parallel.xes',
    'xes_files/tests/log_pn_parallel2.xes',
    'xes_files/tests/log_pn_parallel3.xes'
]
percent_remove = 0.5
seed = 42

for log_path in log_paths:
    log = xes_importer.apply(log_path)
    seqs = case_statistics.get_variant_statistics(log)

    print('variants log:')
    print(seqs)
    print()

    filt_log = create_filtered_logs.filter_log_fixed(log, percent_remove, seed)
    seqs = case_statistics.get_variant_statistics(filt_log)

    print('variants filtered log:')
    print(seqs)
    print()