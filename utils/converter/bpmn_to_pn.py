import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer


my_path = '/home/vercosa/Insync/doutorado/artigos/artigo_alignment/'+\
          'split-miner-2.0/repair-model.bpmn'

# imports a BPMN file
bpmn_graph = pm4py.read_bpmn(my_path)

# converts to Petri-net
net, im, fm = pm4py.convert_to_petri_net(bpmn_graph)

gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.view(gviz)
