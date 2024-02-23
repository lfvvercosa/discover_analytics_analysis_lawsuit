from pm4py.algo.discovery.inductive.variants.im_f import algorithm as IMf
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py import convert_to_dataframe
from experiments.variant_analysis.utils.SplitJoinDF import SplitJoinDF


log_path = 'xes_files/test_variants/exp4/exp4_p2_v1.xes'
log = xes_importer.apply(log_path)

df = convert_to_dataframe(log)
split_join = SplitJoinDF(df)
traces = split_join.split_df()

for t in traces:
    print(t)
    
