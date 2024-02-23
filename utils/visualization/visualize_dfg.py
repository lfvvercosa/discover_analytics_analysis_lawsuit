from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.visualization.dfg import visualizer as dfg_visualization
from metrics import networkx_graph
import networkx as nx
import matplotlib.pyplot as plt


def visualize_dfg(dfg):
    gviz = dfg_visualization.apply(dfg)
    dfg_visualization.view(gviz)
    plt.show()


if __name__ == '__main__':
    my_file = 'xes_files/real_processes/cnj_no_cluster/' + \
            'VARA_DA_FAZENDA_PUBLICA_DA_COMARCA_DE_PAULISTA-TJPE.xes'

    most_frequent = 0.7

    log = xes_importer.apply(my_file)
    dfg = dfg_discovery.apply(log)
    gviz = dfg_visualization.apply(dfg, 
                                log=log, 
                                variant=dfg_visualization.Variants.FREQUENCY)
    dfg_visualization.view(gviz)
    # plt.show()
