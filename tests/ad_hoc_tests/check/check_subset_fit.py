from pm4py.objects.log.importer.xes import importer as xes_importer
from experiments.subset_conf import subset_alg
from pm4py.statistics.traces.generic.log import case_statistics

log_path = 'xes_files/3/BPI_Challenge_2012.xes.gz'

log = xes_importer.apply(log_path)
variants_count1 = case_statistics.get_variant_statistics(log)

log_25_freq = subset_alg.filter_variants_frequency(log, 0.25)
variants_count2 = case_statistics.get_variant_statistics(log_25_freq)

log_50_freq = subset_alg.filter_variants_frequency(log, 0.5)
variants_count3 = case_statistics.get_variant_statistics(log_50_freq)


print('')