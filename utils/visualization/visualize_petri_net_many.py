from os import listdir
from os.path import isfile, join
from utils.visualization.visualize_petri_net import visualize_petri_net
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer


my_path = 'simul_qual_metr/petri_nets/heu_miner/'
files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

for f in files:
    visualize_petri_net(my_path + f)

