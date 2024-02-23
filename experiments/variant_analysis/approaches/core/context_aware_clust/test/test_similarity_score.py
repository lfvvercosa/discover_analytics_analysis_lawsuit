from experiments.variant_analysis.approaches.core.context_aware_clust.\
     SimilarityScore import SimilarityScore
from pm4py.objects.log.importer.xes import importer as xes_importer
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF
from pm4py import convert_to_dataframe


log_path = 'xes_files/test_variants/exp2/p1_v2_mini.xes'
log = xes_importer.apply(log_path, variant=xes_importer.Variants.LINE_BY_LINE)
df_log = convert_to_dataframe(log)
 
split_join = SplitJoinDF(df_log)
traces = split_join.split_df()
ids = split_join.ids


sim_score = SimilarityScore(log)

print()
