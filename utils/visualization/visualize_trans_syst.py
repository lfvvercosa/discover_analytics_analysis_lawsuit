import os
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer


# import log
# log = xes_importer.apply(os.path.join("simulations",
#                                       "source_files",
#                                       "logs",
#                                       "airline_log.xes"))

# discover petri-net
# net, initial_marking, final_marking = inductive_miner.apply(log)

my_file = 'petri_nets/tests/test_product_net.pnml'
# my_file = 'utils/converter/models/examples/' + \
#           'base_example3.pnml'

net, im, fm = pnml_importer.apply(my_file)


# visualize petri-net
# gviz = pn_visualizer.apply(net, 
#                            im, 
#                            fm)

# pn_visualizer.view(gviz) 

# convert to reachability graph
ts = reachability_graph.construct_reachability_graph(net, im)


# visualize reachability graph
# parameters = {ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"}
gviz = ts_visualizer.apply(ts)
ts_visualizer.view(gviz)

print('done!')