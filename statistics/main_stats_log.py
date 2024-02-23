import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

from core.log_complexity.LogFeatures import LogFeatures

if __name__ == "__main__":
    log_path = 'dataset/tribunais_trabalho/TRT.xes'
    log_raw_path = 'dataset/tribunais_trabalho/TRT_raw.xes'

    log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
    log_raw = xes_importer.apply(log_raw_path, variant=xes_importer.Variants.LINE_BY_LINE)
    
    df_log = pm4py.convert_to_dataframe(log)
    df_log_raw = pm4py.convert_to_dataframe(log_raw)

    print('traces available only in processed log:')

    df_temp = df_log[~df_log['case:concept:name'].isin(df_log_raw['case:concept:name'])]

    print(str(len(df_temp.drop_duplicates('case:concept:name').index)))

    df_log_raw = df_log_raw[df_log_raw['case:concept:name'].isin(df_log['case:concept:name'])]
    log_raw = pm4py.convert_to_event_log(df_log_raw)
    
    print('Total traces log: ' + str(len(log)))
    print('Total traces log raw: ' + str(len(log_raw)))


    variants = pm4py.get_variants_as_tuples(log)
    variants_raw = pm4py.get_variants_as_tuples(log_raw)

    print('Total variants log: ' + str(len(variants)))
    print('Total variants log raw: ' + str(len(variants_raw)))
    
    log_complexity = LogFeatures(log)
    log_raw_complexity = LogFeatures(log_raw)

    print('Average Sequence Length log: ' + str(log_complexity.avg_trace_size()))
    print('Total variants log raw: ' + str(log_raw_complexity.avg_trace_size()))

    print('Min Sequence Length log: ' + str(log_complexity.min_sequence_length()))
    print('Min Sequence Length log raw: ' + str(log_raw_complexity.min_sequence_length()))

    print('Max Sequence Length log: ' + str(log_complexity.max_sequence_length()))
    print('Max Sequence Length log raw: ' + str(log_raw_complexity.max_sequence_length()))

    print('Average Sequence Length log: ' + str(log_complexity.avg_trace_size()))
    print('Average Sequence Length log: ' + str(log_raw_complexity.avg_trace_size()))

    print('Percent unique seqs log: ' + str(log_complexity.percent_unique_seqs()))
    print('Percent unique seqs log raw: ' + str(log_raw_complexity.percent_unique_seqs()))

    print('Number of distinct act log: ' + str(log_complexity.number_events_types()))
    print('Number of distinct act log raw: ' + str(log_raw_complexity.number_events_types()))

    print('Average non-overlaping traces log: ' + \
          str(log_complexity.avg_non_overlap_traces()))
    print('Average non-overlaping traces log raw: ' + \
          str(log_raw_complexity.avg_non_overlap_traces()))
    
    print('Average distinct activities log: ' + \
          str(log_complexity.avg_dist_act()))
    print('Average distinct activities log raw: ' + \
          str(log_raw_complexity.avg_dist_act()))


    print('Average levenshtein distance log: ' + \
          str(log_complexity.log_mean_edit_distance()))
    print('Average levenshtein distance log raw: ' + \
          str(log_raw_complexity.log_mean_edit_distance()))
    
    print('Variants entropy norm log: ' + \
          str(log_complexity.variant_entropy()[1]))
    print('Variants entropy norm log raw: ' + \
          str(log_raw_complexity.variant_entropy()[1]))
    
    print('Sequence entropy norm log: ' + \
          str(log_complexity.sequence_entropy()[1]))
    print('Sequence entropy norm log raw: ' + \
          str(log_raw_complexity.sequence_entropy()[1]))
    




    # print('done!')


