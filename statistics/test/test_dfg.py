from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py

log = xes_importer.apply('clustering/test/test_dendrogram.xes',
                         variant=xes_importer.Variants.LINE_BY_LINE)

dfg, sa, ea = pm4py.discover_dfg(log)

print('done!')