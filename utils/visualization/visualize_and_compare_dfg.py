from os import listdir
from os.path import isfile, join
from utils.visualization.visualize_dfg import visualize_dfg
from utils.converter.models.nx_graph_to_dfg import nx_graph_to_dfg
from simul_qual_metr.creation.dfg.pn_to_dfg import petri_net_to_dfg
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer



path_log = 'xes_files/real_processes/set_for_simulations/'
path_models = 'simul_qual_metr/petri_nets/'
folders = ['1/', '2/', '3/', '4/', '5/']

for fol in folders:
    my_path = path_log + fol
    files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

    for f in files:
        print('### File: ' + str(f))
        log = xes_importer.apply(my_path + f)
        dfg_log = dfg_discovery.apply(log)
        visualize_dfg(dfg_log)

        net, im, fm = \
            pnml_importer.apply(join("simul_qual_metr",
                                    "petri_nets",
                                    f + ".pnml"))
        
        G = petri_net_to_dfg(net, im, fm)
        dfg_model = nx_graph_to_dfg(G)
        visualize_dfg(dfg_model)


