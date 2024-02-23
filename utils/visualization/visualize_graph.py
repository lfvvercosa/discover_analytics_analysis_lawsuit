from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.variants import variants_filter
from libraries import networkx_graph
import networkx as nx
import matplotlib.pyplot as plt


my_file = 'xes_files/real_processes/cnj_part1/' + \
          'TERCEIRA_VARA_C?VEL_DA_COMARCA_DE_BLUMENAU_-_TJSC.xes'

most_frequent = 0.7

log = xes_importer.apply(my_file)
log = variants_filter.\
            filter_log_variants_percentage(
                    log, 
                    percentage=most_frequent)
dfg = dfg_discovery.apply(log)
G = networkx_graph.create_graph(dfg)
nx.draw(G)

plt.show()
