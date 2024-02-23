import pm4py
from pm4py.visualization.petri_net import visualizer as pn_visualizer


my_path = 'models/bpi2012_tree.ptml'
out_path = 'models/petri_nets/tests/'+\
           'bpi2012.pnml'

process_tree = pm4py.read_ptml(my_path)
net, im, fm = pm4py.convert_to_petri_net(process_tree)

gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.view(gviz)

pm4py.write_pnml(net, im, fm, out_path)

print('done!')