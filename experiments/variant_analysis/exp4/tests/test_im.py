from pm4py.algo.discovery.inductive.variants.im_f import algorithm as IMf
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

import pm4py

thresh = 0.8


log_path = 'xes_files/test_variants/exp4/exp4_p2_v1.xes'
log = xes_importer.apply(log_path)

params_ind = {IMf.Parameters.NOISE_THRESHOLD:thresh}
# net1, im1, fm1 = IMf.apply(log, params_ind)
net1, im1, fm1 = pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.8)

gviz = pn_visualizer.apply(net1, im1, fm1)
pn_visualizer.view(gviz)