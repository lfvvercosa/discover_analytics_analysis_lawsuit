from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer


if __name__ == '__main__':
    # log_path = 'logs/real/set_for_simulations/1/'+\
    #            '3a_VARA_DE_EXECUCOES_FISCAIS_DA_COMARCA_DE_FORTALEZA_-_TJCE.xes'
    # log = xes_importer.apply(log_path)
    my_file = 'petri_nets/IMf/'+\
              'BPI_Challenge_2014_Inspection.xes.gz.pnml'
    net, im, fm = pnml_importer.apply(my_file)

    TS = reachability_graph.construct_reachability_graph(net, im)


    gviz = ts_visualizer.apply(TS, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "png"})
    ts_visualizer.view(gviz)


